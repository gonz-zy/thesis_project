"""
SUPPLEMENTARY TEST (no CycleGAN): writes the 224 x 224 images of the 676-image set WITHOUT any CycleGAN translation,
in one of two ways, each into its own folder:

  --variant 1024_to_224          histogram-equalized 1024 image -> 224 (one bilinear resize, the transform CXRDataset uses)
                                 -> 5_no_cyclegan_1024_to_224/images/224/
  --variant 1024_to_256_to_224   Setup 3's stored 256 image (made from the same 1024 image) -> 224, exactly the path the
                                 S2/S4B clean images took (clean_images.py: load_256 -> to_224), so every clean image is
                                 byte-identical to the main study's and artifact images simply skip the CycleGAN
                                 -> 5_no_cyclegan_1024_to_256_to_224/images/224/

Geometric augmentation is NOT applied here: as in every setup it is applied on the fly to the 224 training images.
`domain` (artifact | clean) is copied from the ORIGINAL clustering, only for the descriptive domain check.

Input:  3_downsampling_branch/output/clusters/clustered_676.csv, 1_global_pre-processing/output/normalized_images/,
        2_upsampling_branch/output/{training,validation,testing}/   (all read only)
Output: <variant folder>/images/224/*.png, <variant folder>/images/manifest.csv
Run:    python supplementary_test_for_downsampling/make_images_no_cyclegan.py --variant 1024_to_224
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORIG3 = os.path.join(ROOT, "3_downsampling_branch")
if ORIG3 not in sys.path:
    sys.path.append(ORIG3)

import setup4_config as C                      # original config, read only
from setup4_images import load_256, to_224     # the exact functions clean_images.py used

VARIANTS = {"1024_to_224": "5_no_cyclegan_1024_to_224", "1024_to_256_to_224": "5_no_cyclegan_1024_to_256_to_224"}
RESIZE_224 = T.Resize((C.IMG, C.IMG))           # = cxr_data.make_transform's resize (bilinear, antialias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    a = ap.parse_args()
    out_dir = os.path.join("supplementary_test_for_downsampling", VARIANTS[a.variant], "images")
    d224 = os.path.join(out_dir, "224")
    os.makedirs(d224, exist_ok=True)

    cl = pd.read_csv(C.CLUSTERED_CSV)
    assert len(cl) == C.TOTAL_676 and cl["Image Index"].is_unique
    rows = []
    for _, r in cl.iterrows():
        name = r["Image Index"]
        if a.variant == "1024_to_224":
            with Image.open(os.path.join(C.SRC_IMG_1024, name)) as im:
                im = im.convert("L")
                assert im.size == (1024, 1024), (name, im.size)
                arr = np.asarray(RESIZE_224(im), dtype=np.uint8)
        else:
            arr = to_224(load_256(r))
        assert arr.shape == (C.IMG, C.IMG) and arr.dtype == np.uint8
        p = os.path.join(d224, name)
        Image.fromarray(arr).save(p)
        rows.append(dict(**{k: r[k] for k in ("Image Index", "Finding Labels", "split")},
                         domain=r["domain"], translated=False, path=p))
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    print(f"{a.variant}: wrote {len(man)} images to {d224}")
    print(man.groupby(["split", "domain"]).size().to_string())

    if a.variant == "1024_to_256_to_224":       # check: clean images must equal the main study's cleaned/224 files
        orig = pd.read_csv(C.CLEANED_CSV)
        orig = orig[(orig["domain"] == "clean") & orig["Image Index"].isin(set(cl["Image Index"]))]
        bad = 0
        for _, r in orig.iterrows():
            o = np.asarray(Image.open(os.path.join(C.CLEAN_224_DIR, r["Image Index"])))
            n = np.asarray(Image.open(os.path.join(d224, r["Image Index"])))
            bad += int(not np.array_equal(o, n))
        print(f"clean images identical to the main study's cleaned/224: {len(orig) - bad}/{len(orig)}")
        assert bad == 0


if __name__ == "__main__":
    main()
