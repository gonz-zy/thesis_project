"""
setup4_images.py -- the ONE place where Setup 4 reads a 256 image and takes the 256 -> 224 step.

train_cyclegan.py and clean_images.py both import this, so artifact images, clean images and the Setup 3 pipeline
cannot end up on different resize paths. A different path would let the classifier tell the two domains apart by
resampling blur alone, which is a class cue unrelated to the artifacts.

Chain (identical for every image):  1024 --LANCZOS--> 256 (already stored by Setup 3) --[G_AB if artifact]-->
8-bit --torchvision Resize((224, 224))--> 224 PNG.
"""
import os
import sys

import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from setup4_config import CLUSTERED_CSV, GAN_SIZE, IMG, SRC_IMG_1024, SRC_IMG_256

DOMAIN_COL = "domain"
SPLIT_ALIASES = {"train": "train", "training": "train", "validation": "validation", "val": "validation",
                 "valid": "validation", "test": "test", "testing": "test"}

# Same constructor call as cxr_data.make_transform(): T.Resize((IMG, IMG)) with the default (bilinear) interpolation.
# On a PIL image this is PIL's bilinear resize, which averages over the whole source footprint when shrinking.
_RESIZE_224 = T.Resize((IMG, IMG))


def read_domains(csv_path=CLUSTERED_CSV):
    """Rows with Image Index, split and a final `domain` (artifact | clean). Refuses to guess the domain."""
    df = pd.read_csv(csv_path)
    if DOMAIN_COL not in df.columns:
        raise SystemExit(f"{csv_path} has no '{DOMAIN_COL}' column. cluster_kmeans.py only writes 'provisional_domain'; "
                         f"run label_grid.py --score to write the final one. Not guessing.")
    bad = sorted(set(df[DOMAIN_COL]) - {"artifact", "clean"})
    if bad:
        raise SystemExit(f"{csv_path}: unexpected {DOMAIN_COL} values {bad}")
    for c in ("Image Index", "split"):
        if c not in df.columns:
            raise SystemExit(f"{csv_path}: missing column '{c}'")
    df = df.copy()
    df["split"] = df["split"].map(lambda s: SPLIT_ALIASES.get(str(s).lower(), s))
    unknown = sorted(set(df["split"]) - {"train", "validation", "test"})
    if unknown:
        raise SystemExit(f"{csv_path}: unknown split names {unknown}")
    if df["Image Index"].duplicated().any():
        raise SystemExit(f"{csv_path}: duplicated Image Index values")
    if "provisional_domain" in df.columns:
        n = int((df["provisional_domain"] != df[DOMAIN_COL]).sum())
        if n:
            print(f"NOTE: {n} images have a final domain different from the provisional one (the manual check overruled it)")
    return df


def load_256(row, allow_resize_from_1024=False):
    """uint8 (256, 256) array of one image. Reads Setup 3's own 256 file, so clean images are byte-identical to Setup 3's."""
    name = row["Image Index"]
    p = os.path.join(SRC_IMG_256[row["split"]], name)
    if os.path.exists(p):
        with Image.open(p) as im:
            im = im.convert("L")
            if im.size != (GAN_SIZE, GAN_SIZE):
                raise SystemExit(f"{p} is {im.size}, expected {(GAN_SIZE, GAN_SIZE)}")
            return np.asarray(im, dtype=np.uint8).copy()
    if not allow_resize_from_1024:
        raise FileNotFoundError(f"{p} not found. Setup 3's 256 images should hold all 676. If they really do not, "
                                f"rerun with --allow_resize_from_1024 (LANCZOS, the call the StyleGAN split script used; "
                                f"clean_images.py --verify_setup3_dir then checks it against the stored files).")
    with Image.open(os.path.join(SRC_IMG_1024, name)) as im:
        return np.asarray(im.convert("L").resize((GAN_SIZE, GAN_SIZE), Image.LANCZOS), dtype=np.uint8).copy()


def to_224(arr256):
    """uint8 (256, 256) -> uint8 (224, 224) through the exact transform CXRDataset applies."""
    return np.asarray(_RESIZE_224(Image.fromarray(arr256)), dtype=np.uint8)


def save_png(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


def to_model(u8):
    """uint8 array/tensor -> float in [-1, 1] (0 -> -1, 255 -> +1). Exactly invertible by from_model."""
    return u8.astype("float32") / 127.5 - 1.0 if isinstance(u8, np.ndarray) else u8.float() / 127.5 - 1.0


def from_model(x):
    """Tensor in [-1, 1] -> uint8 numpy, rounded (not truncated)."""
    return ((x.clamp(-1, 1) + 1.0) * 127.5).round().to("cpu").numpy().astype(np.uint8)
