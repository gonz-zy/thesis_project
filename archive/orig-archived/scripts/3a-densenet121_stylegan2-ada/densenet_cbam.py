"""
densenet_cbam.py -- DenseNet-121 + CBAM + clinical variables (Setup 1 classifier).

Architecture (manuscript, Developmental Method):
  ImageNet-pretrained DenseNet-121 -> CBAM after EACH of the four dense blocks -> global average pooling
  -> concatenate the 3 clinical values (sex, view, age) -> ONE fully connected unit -> sigmoid (applied in the loss).
  No dropout, no extra clinical layer: the head is "GAP then one FC", as the manuscript and Chehade describe.

cbam = "all"   CBAM after all four dense blocks  (thesis method, Setup 1)
cbam = "last"  CBAM only after the last block     (Chehade's placement; exploratory check only)
cbam = "none"  plain DenseNet-121                 (exploratory check only)

CBAM initialisation (manuscript: "small random weights"):
  every CBAM convolution ~ N(0, 0.01^2). The two gates then start near sigmoid(0) = 0.5, so a plain CBAM would
  shrink the pretrained features to about 0.5 x 0.5 = 0.25 at the start. The module output is therefore multiplied
  by a fixed factor of 4, which makes each CBAM start as an approximate identity (0.25 x 4 = 1) and keeps the
  pretrained features intact. Training then learns where to emphasise. The attention maths runs in float32.

Running this file prints the parameter counts (useful for the CBAM-cost sentence in the manuscript):
    python scripts\\3a-densenet121_stylegan2-ada\\densenet_cbam.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SMALL_STD = 0.01     # "small random weights"
CBAM_SCALE = 4.0     # cancels the 0.5 x 0.5 gate at initialisation -> near-identity start


class ChannelAttention(nn.Module):
    """Which channels matter: shared MLP on the average- and max-pooled channel descriptors (reduction r)."""

    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(c // r, c, 1, bias=False))

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x, 1))
        m = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(a + m)


class SpatialAttention(nn.Module):
    """Where to look: k x k conv on the channel-wise average and max maps."""

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018): channel attention, then spatial attention,
    with the near-identity start described in the header (output x CBAM_SCALE)."""

    def __init__(self, c, r=16, k=7, scale=CBAM_SCALE):
        super().__init__()
        self.ca, self.sa, self.scale = ChannelAttention(c, r), SpatialAttention(k), scale
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=SMALL_STD)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):  # attention in float32
            x32 = x.float()
            y = x32 * self.ca(x32)
            y = y * self.sa(y)
        return (self.scale * y).to(x.dtype)


class DenseNetVariant(nn.Module):
    def __init__(self, cbam="all", n_clinical=3, pretrained=True):
        super().__init__()
        assert cbam in ("none", "last", "all"), cbam
        self.cbam_mode = cbam
        try:
            w = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
            f = torchvision.models.densenet121(weights=w).features
        except AttributeError:  # older torchvision
            f = torchvision.models.densenet121(pretrained=pretrained).features
        self.stem = nn.Sequential(f.conv0, f.norm0, f.relu0, f.pool0)
        self.b1, self.t1 = f.denseblock1, f.transition1
        self.b2, self.t2 = f.denseblock2, f.transition2
        self.b3, self.t3 = f.denseblock3, f.transition3
        self.b4 = f.denseblock4
        # channels at the end of dense blocks 1-4 are 256, 512, 1024, 1024
        self.cbam1 = CBAM(256) if cbam == "all" else nn.Identity()
        self.cbam2 = CBAM(512) if cbam == "all" else nn.Identity()
        self.cbam3 = CBAM(1024) if cbam == "all" else nn.Identity()
        self.cbam4 = CBAM(1024) if cbam in ("last", "all") else nn.Identity()
        self.norm5 = f.norm5
        self.head = nn.Linear(1024 + n_clinical, 1)  # GAP features + clinical values -> one unit

    def forward(self, x, clin):
        x = self.stem(x)
        x = self.t1(self.cbam1(self.b1(x)))
        x = self.t2(self.cbam2(self.b2(x)))
        x = self.t3(self.cbam3(self.b3(x)))
        x = self.cbam4(self.b4(x))
        x = F.relu(self.norm5(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(torch.cat([x, clin], dim=1)).squeeze(1)  # logit (sigmoid is inside the loss)


if __name__ == "__main__":
    def n_params(m):
        return sum(p.numel() for p in m.parameters())
    base = DenseNetVariant("none", pretrained=False)
    for mode in ("none", "last", "all"):
        m = DenseNetVariant(mode, pretrained=False)
        extra = n_params(m) - n_params(base)
        print(f"cbam={mode:5s} parameters={n_params(m):,}  CBAM adds {extra:,} ({100 * extra / n_params(base):.2f}%)")
