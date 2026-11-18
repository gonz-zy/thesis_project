"""
Setup 1: paths, run-folder naming, the image transforms (resize 224, standardize, augmentation) and the Dataset.

Input:  2_upsampling_branch/output/labels/combined_training_labels.csv (84,408 rows), validation and testing labels,
        4_image_transformation/output/train_stats_setup1_setup3.json
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# paths
ROOT = os.path.join("2_upsampling_branch", "output")
LABELS = os.path.join(ROOT, "labels")
SYN_DIR = os.path.join(ROOT, "generated-pneumonia")
COMBINED_CSV = os.path.join(LABELS, "combined_training_labels.csv")
STATS_JSON = os.path.join("4_image_transformation", "output", "train_stats_setup1_setup3.json")
RESULTS_DIR = os.path.join("5_training_and_classification", "s1_densenet121_stylegan2-ada", "output")
IMG = 224  # manuscript: 224 x 224 input


def run_paths(out, cbam, seed, trial=None):
    """File paths of one run: <out>/final/cbam_<cbam>/seed<seed>/ or <out>/search/cbam_<cbam>/trial<NNN>/."""
    tag = f"cbam-{cbam}"
    if trial is None:
        run_id = f"{tag}_seed{seed}"
        run_dir = os.path.join(out, "final", f"cbam_{cbam}", f"seed{seed}")
    else:
        run_id = f"search_{tag}_trial{trial:03d}"
        run_dir = os.path.join(out, "search", f"cbam_{cbam}", f"trial{trial:03d}")
    return dict(
        run_id=run_id, run_dir=run_dir,
        history=os.path.join(run_dir, f"history_{run_id}.csv"),
        best=os.path.join(run_dir, f"best_{run_id}.pt"),
        metrics=os.path.join(run_dir, f"metrics_{run_id}.json"),
        val_pred=os.path.join(run_dir, f"val_predictions_{run_id}.csv"),
        test_pred=os.path.join(run_dir, f"test_predictions_{run_id}.csv"),
    )


class ChehadeAugment:
    """Augmentation of Chehade et al. (2025), training images only: rotation 10 deg, horizontal flip, shift 0.1, zoom 0.1."""
    PAD = 64

    def __init__(self, degrees=10.0, shift=0.1, zoom=0.1):
        self.degrees, self.shift, self.zoom = degrees, shift, zoom

    @staticmethod
    def _u(a, b):
        return float(torch.empty(1).uniform_(a, b))

    def __call__(self, x):  # x: (1, H, W) tensor in [0, 1]
        h, w = x.shape[-2:]
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])  # horizontal flip (moves the heart to the right: stated limitation)
        angle = self._u(-self.degrees, self.degrees)
        tx = int(round(self._u(-self.shift, self.shift) * w))
        ty = int(round(self._u(-self.shift, self.shift) * h))
        scale = self._u(1 - self.zoom, 1 + self.zoom)
        p = self.PAD
        xp = F.pad(x.unsqueeze(0), (p, p, p, p), mode="replicate").squeeze(0)
        xp = TF.affine(xp, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
                       interpolation=T.InterpolationMode.BILINEAR)
        return TF.center_crop(xp, [h, w])


def make_transform(train, mean=None, std=None):
    """Resize to 224 x 224, scale to [0, 1], augment (training only), standardize with the training-set mean/std."""
    t = [T.Resize((IMG, IMG)), T.ToTensor()]
    if train:
        t.append(ChehadeAugment())
    if mean is not None:  # mean=None only while computing the statistics themselves
        t.append(T.Normalize([mean], [std]))
    return T.Compose(t)


class CXRDataset(Dataset):
    """Returns (image, clinical values [sex, view position, age], label, is_synthetic) for each row of df."""

    def __init__(self, df, img_root=None, train=False, mean=None, std=None):
        self.df = df.reset_index(drop=True)
        if "path" in self.df.columns:
            self.paths = self.df["path"].tolist()
        else:
            self.paths = [os.path.join(img_root, f) for f in self.df["Image Index"]]
        self.y = (self.df["Finding Labels"] == "Pneumonia").astype(np.float32).values
        self.synth = (self.df["is_synthetic"].values.astype(np.int64)
                      if "is_synthetic" in self.df.columns else np.zeros(len(self.df), dtype=np.int64))
        # clinical values: sex and view position as 0/1, age / 100
        sex = (self.df["Patient Sex"] == "M").astype(np.float32).values
        view = (self.df["View Position"] == "PA").astype(np.float32).values
        age = np.clip(self.df["Patient Age"].values.astype(np.float32), 0, 100) / 100.0
        self.clin = np.stack([sex, view, age], axis=1).astype(np.float32)
        self.tf = make_transform(train, mean, std)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("L"))
        x = x.repeat(3, 1, 1)  # grayscale -> 3 channels for the ImageNet-pretrained backbone
        return x, torch.from_numpy(self.clin[i]), torch.tensor(self.y[i]), torch.tensor(self.synth[i])
