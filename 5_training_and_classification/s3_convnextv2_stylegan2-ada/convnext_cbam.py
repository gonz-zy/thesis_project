"""
Setup 3 and Setup 4 model: ConvNeXt V2-Tiny (timm 'convnextv2_tiny.fcmae_ft_in1k', ImageNet weights downloaded
on first use) with CBAM after each of the four stages, global average pooling, the 3 clinical values
concatenated, one output unit (sigmoid in the loss). Drop-path rate 0.2 (fixed, not tuned).
Run this file to print the parameter counts.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE = "convnextv2_tiny.fcmae_ft_in1k"
DROP_PATH = 0.2
SMALL_STD = 0.01     # "small random weights"
CBAM_SCALE = 4.0     # cancels the 0.5 x 0.5 gate at initialisation -> near-identity start


def stage_channels(backbone):
    """Output channels of the four ConvNeXt stages, across timm versions."""
    fi = getattr(backbone, "feature_info", None)
    if fi is not None:
        if hasattr(fi, "channels"):                      # older timm: FeatureInfo object
            try:
                return list(fi.channels())
            except Exception:
                pass
        if isinstance(fi, (list, tuple)) and fi and isinstance(fi[0], dict):   # timm 1.0.x: list of dicts
            chans = [d.get("num_chs") for d in fi]
            if all(isinstance(c, int) for c in chans):
                return chans
    # last resort: measure them
    was_training = backbone.training
    backbone.eval()
    chans = []
    with torch.no_grad():
        x = torch.zeros(1, 3, 224, 224)
        x = backbone.stem(x)
        for stage in backbone.stages:
            x = stage(x)
            chans.append(int(x.shape[1]))
    backbone.train(was_training)
    return chans


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
    """Convolutional Block Attention Module (Woo et al., 2018), identical to the DenseNet-121 version."""

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


class ConvNeXtV2Variant(nn.Module):
    def __init__(self, cbam="all", n_clinical=3, pretrained=True, backbone=BACKBONE, drop_path=DROP_PATH):
        super().__init__()
        assert cbam in ("none", "last", "all"), cbam
        self.cbam_mode = cbam
        # num_classes=0 removes timm's classifier: the backbone returns the pooled 768-d feature vector
        # WEIGHTS: ImageNet-pretrained ConvNeXt V2-Tiny from timm (fine-tuned during training)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
        chans = stage_channels(self.backbone)  # tiny: [96, 192, 384, 768] = output of stages 1-4
        assert len(chans) == 4, f"expected 4 stages, got {chans}"
        assert len(self.backbone.stages) == 4, f"expected 4 stage modules, got {len(self.backbone.stages)}"
        self.stage_chans = chans
        self.cbams = nn.ModuleList([
            CBAM(c) if (cbam == "all" or (cbam == "last" and i == 3)) else nn.Identity()
            for i, c in enumerate(chans)])
        self.head = nn.Linear(self.backbone.num_features + n_clinical, 1)  # pooled features + clinical -> one unit

    def forward(self, x, clin):
        b = self.backbone
        x = b.stem(x)
        for stage, cbam in zip(b.stages, self.cbams):  # CBAM at the end of every stage
            x = cbam(stage(x))
        x = b.norm_pre(x)
        f = b.forward_head(x, pre_logits=True)  # global average pooling (+ ConvNeXt's final norm), flattened
        return self.head(torch.cat([f, clin], dim=1)).squeeze(1)  # logit (sigmoid is inside the loss)


if __name__ == "__main__":
    def n_params(m):
        return sum(p.numel() for p in m.parameters())
    print(f"timm {timm.__version__}")
    base = ConvNeXtV2Variant("none", pretrained=False)
    print(f"stage channels detected: {base.stage_chans}   (expected [96, 192, 384, 768])")
    for mode in ("none", "last", "all"):
        m = ConvNeXtV2Variant(mode, pretrained=False)
        extra = n_params(m) - n_params(base)
        print(f"cbam={mode:5s} parameters={n_params(m):,}  CBAM adds {extra:,} ({100 * extra / n_params(base):.2f}%)")
    # one forward pass, so a shape problem shows up here rather than inside a training run
    m = ConvNeXtV2Variant("all", pretrained=False)
    out = m(torch.zeros(2, 3, 224, 224), torch.zeros(2, 3))
    print(f"forward pass ok: input (2, 3, 224, 224) -> logits {tuple(out.shape)}")
