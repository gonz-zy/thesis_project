"""
Stage 3, check: FID and KID of artifact images before and after CycleGAN, against clean images.

Input:  output/clusters/clustered_all.csv, output/cleaned/cleaned_manifest.csv, cleaned/224/, 256 images
Output: supplementary_test_for_downsampling/3_new_downsampling_branch/output/fid_kid/fid_kid_<backbone>.csv, .json, feature_cache_<backbone>.npz
Run:    python supplementary_test_for_downsampling/3_new_downsampling_branch/fid_kid.py --backbone fid
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import setup4_config as C

OUT_DIR = os.path.join(C.ROOT, "fid_kid")
CLUSTERED_ALL = os.path.join(C.CLUSTERS_DIR, "clustered_all.csv")


# metrics
def fid_from_feats(x, y):
    """Frechet distance between Gaussians fitted to x and y (same value as pytorch-fid)."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    mu1, mu2 = x.mean(0), y.mean(0)
    xc, yc = x - mu1, y - mu2
    n1, n2 = len(x), len(y)
    tr1, tr2 = (xc ** 2).sum() / (n1 - 1), (yc ** 2).sum() / (n2 - 1)
    k = (xc @ yc.T) / np.sqrt((n1 - 1.0) * (n2 - 1.0))
    tr_covmean = np.linalg.svd(k, compute_uv=False).sum()
    d = mu1 - mu2
    return float(d.dot(d) + tr1 + tr2 - 2.0 * tr_covmean)


def kid_from_feats(x, y, degree=3, coef0=1.0):
    """Unbiased MMD^2 with the polynomial kernel k(a, b) = (a.b / d + 1)^3 (Binkowski et al. 2018)."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    d = x.shape[1]
    kxx = (x @ x.T / d + coef0) ** degree
    kyy = (y @ y.T / d + coef0) ** degree
    kxy = (x @ y.T / d + coef0) ** degree
    m, n = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                 - 2.0 * kxy.mean())


# images
def _clean_path(idx, man_paths):
    return man_paths[idx] if idx in man_paths else os.path.join(C.CLEAN_224_DIR, idx)


class ImgSet(Dataset):
    """kind 'after' = cleaned/224 file as is; kind 'before' = Setup 3's 256 image -> torchvision Resize((224, 224))."""

    def __init__(self, items):            # items: list of (kind, path)
        self.items = items
        self.rs = T.Compose([T.Resize((C.IMG, C.IMG)), T.ToTensor()])
        self.tt = T.ToTensor()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        kind, p = self.items[i]
        with Image.open(p) as im:
            im = im.convert("L")
            x = self.rs(im) if kind == "before" else self.tt(im)
        return x


def load_gray(kind, path):
    return ImgSet([(kind, path)])[0].numpy()


# feature extractors
class RandomNet(nn.Module):
    """Test-only stand-in: fixed random conv features, 128-d."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w = nn.Parameter(torch.randn(128, 3, 7, 7, generator=g) * 0.1, requires_grad=False)

    def forward(self, x):
        return torch.relu(nn.functional.conv2d(x, self.w, stride=8)).mean((2, 3))


def make_extractor(name, device):
    if name == "fid":
        from pytorch_fid.inception import InceptionV3
        net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True)
        f = lambda x: net(x)[0].squeeze(-1).squeeze(-1)
    elif name == "torchvision":
        import torchvision
        net = torchvision.models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
        net.fc = nn.Identity()
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        f = lambda x: net((nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
                           - mean) / std)
    elif name == "random":
        net = RandomNet()
        f = net
    else:
        raise ValueError(name)
    net.to(device).eval()
    return f


@torch.no_grad()
def extract(items, fn, device, batch=64, workers=4):
    dl = DataLoader(ImgSet(items), batch_size=batch, shuffle=False, num_workers=workers)
    out = []
    for x in dl:
        x = x.to(device).repeat(1, 3, 1, 1)                                 # grayscale -> 3 channels, [0, 1]
        out.append(fn(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=CLUSTERED_ALL, help="clustered_all.csv (needs split, domain, origin)")
    ap.add_argument("--cleaned_manifest", default=C.CLEANED_CSV, help="cleaned_manifest.csv (path column, optional)")
    ap.add_argument("--backbone", choices=["fid", "torchvision", "random"], default="fid")
    ap.add_argument("--pool", type=int, default=2000, help="max images drawn per group and domain (feature pool)")
    ap.add_argument("--n_per_set", type=int, default=1000, help="images per compared set in each repeat (max)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--min_n", type=int, default=30, help="skip a group whose compared sets would be smaller")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cl = pd.read_csv(a.manifest)
    cl["y"] = np.where(cl["Finding Labels"] == "Pneumonia", "Pneumonia", "No Finding")
    man_paths = {}
    if os.path.exists(a.cleaned_manifest):
        cm = pd.read_csv(a.cleaned_manifest)
        if "path" in cm.columns:
            man_paths = dict(zip(cm["Image Index"], cm["path"]))
    rng = np.random.default_rng(a.seed)

    groups = {                                       # name -> boolean mask over cl
        "seen_all (676)": cl["origin"] == "subset_676",
        "seen_NoFinding": (cl["origin"] == "subset_676") & (cl["y"] == "No Finding"),
        "seen_Pneumonia": (cl["origin"] == "subset_676") & (cl["y"] == "Pneumonia"),
        "unseen_NoFinding (G_AB never saw these)": cl["origin"] == "assigned",
    }

    # pools: fixed random draws, so every repeat subsamples the same extracted features
    pools = {}
    for g, m in groups.items():
        d = cl[m]
        art, cln = d[d["domain"] == "artifact"], d[d["domain"] == "clean"]
        take = lambda df: df.iloc[rng.permutation(len(df))[:a.pool]]
        pools[g] = (take(art), take(cln), len(art), len(cln))

    # resize check: for CLEAN-domain images the cleaned file must equal Setup 3's 224 image exactly
    chk_rows = pools["seen_all (676)"][1].head(40)
    n_eq = 0
    for _, r in chk_rows.iterrows():
        p_after = _clean_path(r["Image Index"], man_paths)
        p_before = os.path.join(C.SRC_IMG_256[r["split"]], r["Image Index"])
        if os.path.exists(p_after) and os.path.exists(p_before):
            n_eq += int(np.array_equal(load_gray("after", p_after), load_gray("before", p_before)))
    print(f"resize check: {n_eq} of {len(chk_rows)} clean-domain images identical before/after (expect all)")

    # features (cached per backbone)
    cache_f = os.path.join(a.out, f"feature_cache_{a.backbone}.npz")
    cache = {}
    if os.path.exists(cache_f):
        z = np.load(cache_f, allow_pickle=False)
        cache = dict(zip(z["keys"].tolist(), z["feats"]))
    need = {}                                                    # key -> (kind, path)
    for g, (art, cln, _, _) in pools.items():
        for _, r in art.iterrows():
            need[f"after|{r['Image Index']}"] = ("after", _clean_path(r["Image Index"], man_paths))
            need[f"before|{r['Image Index']}"] = ("before", os.path.join(C.SRC_IMG_256[r["split"]], r["Image Index"]))
        for _, r in cln.iterrows():                              # clean domain: before == after, one extraction
            need[f"after|{r['Image Index']}"] = ("after", _clean_path(r["Image Index"], man_paths))
    todo = [k for k in need if k not in cache]
    if todo:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"extracting {len(todo)} feature vectors with '{a.backbone}' on {dev} (cached: {len(cache)})")
        t0 = time.time()
        fn = make_extractor(a.backbone, dev)
        feats = extract([need[k] for k in todo], fn, dev, workers=a.workers)
        for k, v in zip(todo, feats):
            cache[k] = v
        np.savez(cache_f, keys=np.array(list(cache.keys())), feats=np.stack(list(cache.values())))
        print(f"  done in {(time.time() - t0) / 60:.1f} min")

    # comparisons
    rows = []
    for g, (art, cln, n_art, n_cln) in pools.items():
        n = min(a.n_per_set, len(art), len(cln) // 2)
        if n < a.min_n:
            print(f"[skip] {g}: only {len(art)} artifact / {len(cln)} clean images in the pool (need n >= {a.min_n})")
            continue
        fa_b = np.stack([cache[f"before|{i}"] for i in art["Image Index"]])
        fa_a = np.stack([cache[f"after|{i}"] for i in art["Image Index"]])
        fr = np.stack([cache[f"after|{i}"] for i in cln["Image Index"]])
        res = {k: [] for k in ("fid_before", "fid_after", "fid_floor", "kid_before", "kid_after", "kid_floor")}
        for _ in range(a.repeats):
            ia = rng.permutation(len(art))[:n]
            ir = rng.permutation(len(cln))[:2 * n]
            r1, r2 = fr[ir[:n]], fr[ir[n:2 * n]]
            res["fid_before"].append(fid_from_feats(fa_b[ia], r1))
            res["fid_after"].append(fid_from_feats(fa_a[ia], r1))
            res["fid_floor"].append(fid_from_feats(r2, r1))
            res["kid_before"].append(kid_from_feats(fa_b[ia], r1))
            res["kid_after"].append(kid_from_feats(fa_a[ia], r1))
            res["kid_floor"].append(kid_from_feats(r2, r1))
        r = {k: np.array(v) for k, v in res.items()}
        rows.append(dict(
            group=g, n_artifact_total=n_art, n_clean_total=n_cln, n_per_set=n, repeats=a.repeats,
            kid1000_before=1e3 * r["kid_before"].mean(), kid1000_before_sd=1e3 * r["kid_before"].std(ddof=1),
            kid1000_after=1e3 * r["kid_after"].mean(), kid1000_after_sd=1e3 * r["kid_after"].std(ddof=1),
            kid1000_floor=1e3 * r["kid_floor"].mean(), kid1000_floor_sd=1e3 * r["kid_floor"].std(ddof=1),
            kid_improved_in=f"{int((r['kid_after'] < r['kid_before']).sum())}/{a.repeats}",
            fid_before=r["fid_before"].mean(), fid_after=r["fid_after"].mean(), fid_floor=r["fid_floor"].mean(),
            fid_before_sd=r["fid_before"].std(ddof=1), fid_after_sd=r["fid_after"].std(ddof=1),
            fid_floor_sd=r["fid_floor"].std(ddof=1)))
    if not rows:
        raise SystemExit("no group had enough images")
    df = pd.DataFrame(rows)
    out_csv = os.path.join(a.out, f"fid_kid_{a.backbone}.csv")
    df.round(4).to_csv(out_csv, index=False)
    json.dump(dict(backbone=a.backbone, pool=a.pool, n_per_set=a.n_per_set, repeats=a.repeats, seed=a.seed,
                   resize_check=f"{n_eq}/{len(chk_rows)}",
                   group_sizes={g: dict(artifact=v[2], clean=v[3]) for g, v in pools.items()}),
              open(os.path.join(a.out, f"fid_kid_{a.backbone}.json"), "w"), indent=2)

    print(f"\nbackbone {a.backbone} | KID x1000 (mean +- sd over {a.repeats} repeats); floor = clean vs clean")
    for _, r in df.iterrows():
        print(f"\n{r['group']}  (artifact {r['n_artifact_total']}, clean {r['n_clean_total']}, n per set {r['n_per_set']})")
        print(f"  KID x1000  before {r['kid1000_before']:8.2f} +- {r['kid1000_before_sd']:.2f}   "
              f"after {r['kid1000_after']:8.2f} +- {r['kid1000_after_sd']:.2f}   "
              f"floor {r['kid1000_floor']:8.2f} +- {r['kid1000_floor_sd']:.2f}   improved in {r['kid_improved_in']}")
        print(f"  FID        before {r['fid_before']:8.2f}            after {r['fid_after']:8.2f}            "
              f"floor {r['fid_floor']:8.2f}")
    print(f"\nwritten: {out_csv}")


if __name__ == "__main__":
    main()
