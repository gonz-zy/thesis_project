"""
cyclegan_model.py -- CycleGAN (Zhu et al. 2017) for single-channel radiographs. Pure model code: no paths, no data.

Architectures, LSGAN loss, image pool and loss weights are the reusable parts of the friend's `CycleGAN v3.py`
(ResNet-9 generators, 70 x 70 PatchGAN, identity weight 5.0 = the reference implementation's 0.5 x lambda_cycle 10).

What differs from the friend's file, and why:
  * the image pool stores DETACHED copies (the friend's pool kept tensors attached to the autograd graph)
  * checkpoints also hold both LR schedulers, the pools and the sampler state, so a resumed run continues
    exactly where it stopped (the friend's `load` restarted the LR schedule from epoch 0)
  * the per-iteration losses stay on the device (no .item() sync every step); the trainer reads them once per epoch
  * no dataset, no config dataclass, no file logic here: train_cyclegan.py and clean_images.py own those

Domain A = artifact images, domain B = clean images.  G_AB: artifact -> clean (the cleaner), G_BA: clean -> artifact.
"""
import itertools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ building blocks
def _norm(dim):
    return nn.InstanceNorm2d(dim, affine=False, track_running_stats=False)


def init_weights(net, gain=0.02):
    """Normal(0, gain) for every conv, as in the reference implementation."""
    def fn(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    net.apply(fn)
    return net


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), _norm(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), _norm(dim))

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """c7s1-64, d128, d256, n_blocks x R256, u128, u64, c7s1-1, tanh. Works for any side that is a multiple of 4."""

    def __init__(self, ngf=64, n_blocks=9):
        super().__init__()
        m = [nn.ReflectionPad2d(3), nn.Conv2d(1, ngf, 7), _norm(ngf), nn.ReLU(True)]
        for i in range(2):                                   # two stride-2 downsamplings
            k = 2 ** i
            m += [nn.Conv2d(ngf * k, ngf * k * 2, 3, stride=2, padding=1), _norm(ngf * k * 2), nn.ReLU(True)]
        m += [ResnetBlock(ngf * 4) for _ in range(n_blocks)]
        for i in range(2):                                   # two fractionally strided upsamplings
            k = 2 ** (2 - i)
            m += [nn.ConvTranspose2d(ngf * k, ngf * k // 2, 3, stride=2, padding=1, output_padding=1),
                  _norm(ngf * k // 2), nn.ReLU(True)]
        m += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, 7), nn.Tanh()]
        self.model = nn.Sequential(*m)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """70 x 70 PatchGAN (n_layers=3): C64-C128-C256-C512, then a 1-channel patch map."""

    def __init__(self, ndf=64, n_layers=3):
        super().__init__()
        m = [nn.Conv2d(1, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            m += [nn.Conv2d(ndf * prev, ndf * mult, 4, stride=2, padding=1), _norm(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        m += [nn.Conv2d(ndf * prev, ndf * mult, 4, stride=1, padding=1), _norm(ndf * mult), nn.LeakyReLU(0.2, True),
              nn.Conv2d(ndf * mult, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*m)

    def forward(self, x):
        return self.model(x)


def lsgan(pred, target_is_real):
    """LSGAN: mean squared error to 1 (real) or 0 (fake)."""
    return F.mse_loss(pred, torch.full_like(pred, 1.0 if target_is_real else 0.0))


class ImagePool:
    """History of generated images (Shrivastava et al.): the discriminator sees a mix of current and older fakes.
    Stores DETACHED copies; the friend's version kept graph-attached tensors."""

    def __init__(self, size=50, seed=42):
        self.size, self.images, self.rng = size, [], random.Random(seed)

    def query(self, imgs):
        if self.size == 0:
            return imgs
        out = []
        for im in imgs:
            im = im.detach().unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(im)
                out.append(im)
            elif self.rng.random() > 0.5:
                j = self.rng.randint(0, self.size - 1)
                out.append(self.images[j].clone())
                self.images[j] = im
            else:
                out.append(im)
        return torch.cat(out, 0)

    def state(self):
        return {"images": [t.cpu() for t in self.images], "rng": self.rng.getstate()}

    def load(self, st, device):
        self.images = [t.to(device) for t in st["images"]]
        self.rng.setstate(st["rng"])


# ------------------------------------------------------------------ the model
def lr_rule(epochs, decay_start):
    """Reference schedule: constant for `decay_start` epochs, then linear decay towards zero over the rest.
    Epoch index e is 0-based:  1 - max(0, e + 1 - decay_start) / (epochs - decay_start + 1)."""
    return lambda e: 1.0 - max(0, e + 1 - decay_start) / float(epochs - decay_start + 1)


def set_requires_grad(nets, flag):
    for n in nets:
        for p in n.parameters():
            p.requires_grad = flag


class CycleGAN:
    def __init__(self, device, *, epochs=200, decay_start=100, lr=2e-4, betas=(0.5, 0.999), lambda_cycle=10.0,
                 lambda_identity=5.0, pool_size=50, n_blocks=9, ngf=64, ndf=64, n_layers_d=3, seed=42):
        self.device = torch.device(device)
        self.hp = dict(epochs=epochs, decay_start=decay_start, lr=lr, betas=list(betas), lambda_cycle=lambda_cycle,
                       lambda_identity=lambda_identity, pool_size=pool_size, n_blocks=n_blocks, ngf=ngf, ndf=ndf,
                       n_layers_d=n_layers_d)
        self.lambda_cycle, self.lambda_identity = lambda_cycle, lambda_identity
        mk_g = lambda: init_weights(ResnetGenerator(ngf, n_blocks)).to(self.device)
        mk_d = lambda: init_weights(PatchDiscriminator(ndf, n_layers_d)).to(self.device)
        self.G_AB, self.G_BA, self.D_A, self.D_B = mk_g(), mk_g(), mk_d(), mk_d()
        self.opt_g = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=lr, betas=betas)
        self.opt_d = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=lr, betas=betas)
        rule = lr_rule(epochs, decay_start)
        self.sched_g = torch.optim.lr_scheduler.LambdaLR(self.opt_g, rule)
        self.sched_d = torch.optim.lr_scheduler.LambdaLR(self.opt_d, rule)
        self.pool_A, self.pool_B = ImagePool(pool_size, seed), ImagePool(pool_size, seed + 1)

    @property
    def lr(self):
        return self.opt_g.param_groups[0]["lr"]

    def step_schedulers(self):
        self.sched_g.step()
        self.sched_d.step()

    def optimize(self, real_a, real_b):
        """One iteration (batch of 1 in the reference recipe). Returns detached loss tensors."""
        l1 = F.l1_loss
        fake_b = self.G_AB(real_a)
        rec_a = self.G_BA(fake_b)
        fake_a = self.G_BA(real_b)
        rec_b = self.G_AB(fake_a)

        # ---- generators (discriminators frozen)
        set_requires_grad([self.D_A, self.D_B], False)
        self.opt_g.zero_grad(set_to_none=True)
        idt_a = l1(self.G_BA(real_a), real_a) * self.lambda_identity   # G_BA should leave artifact images alone
        idt_b = l1(self.G_AB(real_b), real_b) * self.lambda_identity   # G_AB should leave clean images alone
        gan_ab = lsgan(self.D_B(fake_b), True)
        gan_ba = lsgan(self.D_A(fake_a), True)
        cyc_a = l1(rec_a, real_a) * self.lambda_cycle
        cyc_b = l1(rec_b, real_b) * self.lambda_cycle
        loss_g = gan_ab + gan_ba + cyc_a + cyc_b + idt_a + idt_b
        loss_g.backward()
        self.opt_g.step()

        # ---- discriminators
        set_requires_grad([self.D_A, self.D_B], True)
        self.opt_d.zero_grad(set_to_none=True)
        fb, fa = self.pool_B.query(fake_b.detach()), self.pool_A.query(fake_a.detach())
        d_b = 0.5 * (lsgan(self.D_B(real_b), True) + lsgan(self.D_B(fb.detach()), False))
        d_a = 0.5 * (lsgan(self.D_A(real_a), True) + lsgan(self.D_A(fa.detach()), False))
        d_b.backward()
        d_a.backward()
        self.opt_d.step()

        return {k: v.detach() for k, v in dict(G_total=loss_g, gan_AB=gan_ab, gan_BA=gan_ba, cycle_A=cyc_a,
                                              cycle_B=cyc_b, idt_A=idt_a, idt_B=idt_b, D_A=d_a, D_B=d_b).items()}

    @torch.no_grad()
    def clean(self, x):
        """G_AB on a [-1, 1] batch (N, 1, H, W). Instance norm keeps no running statistics, so train/eval are the same."""
        return self.G_AB(x.to(self.device))

    # ------------------------------------------------------------ checkpoint
    def state_dict(self):
        return dict(G_AB=self.G_AB.state_dict(), G_BA=self.G_BA.state_dict(), D_A=self.D_A.state_dict(),
                    D_B=self.D_B.state_dict(), opt_g=self.opt_g.state_dict(), opt_d=self.opt_d.state_dict(),
                    sched_g=self.sched_g.state_dict(), sched_d=self.sched_d.state_dict(),
                    pool_A=self.pool_A.state(), pool_B=self.pool_B.state(), hp=self.hp)

    def load_state_dict(self, st):
        for k in ("epochs", "decay_start", "lr", "lambda_cycle", "lambda_identity", "n_blocks", "ngf", "ndf"):
            assert st["hp"][k] == self.hp[k], f"checkpoint was trained with {k}={st['hp'][k]}, this run has {self.hp[k]}"
        for name in ("G_AB", "G_BA", "D_A", "D_B"):
            getattr(self, name).load_state_dict(st[name])
        self.opt_g.load_state_dict(st["opt_g"])
        self.opt_d.load_state_dict(st["opt_d"])
        self.sched_g.load_state_dict(st["sched_g"])
        self.sched_d.load_state_dict(st["sched_d"])
        self.pool_A.load(st["pool_A"], self.device)
        self.pool_B.load(st["pool_B"], self.device)


def load_generator(path, device="cpu"):
    """G_AB only, in eval mode, from a training checkpoint (what clean_images.py needs)."""
    st = torch.load(path, map_location=device, weights_only=True)
    hp = st["hp"]
    g = ResnetGenerator(hp["ngf"], hp["n_blocks"]).to(device)
    g.load_state_dict(st["G_AB"])
    return g.eval(), int(st["epoch"])
