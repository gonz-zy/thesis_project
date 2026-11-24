"""
Supplementary (moved synthetic): Step 3 leakage diagnostics.

1. Embeddings from a FIXED pretrained encoder (torchvision DenseNet-121, ImageNet weights, frozen; not one of the
   evaluated models): global-average-pooled 1,024-d features, L2-normalised. For every val/test image, cosine
   distance to its nearest training image, separately against real and synthetic training images.
   Threshold: the 1st (and 5th) percentile of the nearest-real-training distance of REAL val/test images. Real
   partitions share no patients, so that distribution is the empirical "different patient, nothing leaked" baseline;
   a synthetic val/test image closer to a training image than 99% of genuinely independent real images is flagged.
2. (age, sex, view) metadata tuples of val/test synthetic rows: distinct count, and how many are exact copies of
   training rows.

Input:  supplementary_moved_synthetic/output/manifests/*.csv
Output: supplementary_moved_synthetic/output/leakage/nn_distances.csv, nn_summary.csv, nn_thresholds.json,
        metadata_tuples.csv, metadata_summary.json, nn_distance_hist.png
Run:    python supplementary_moved_synthetic/leakage.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MAN = os.path.join("supplementary_moved_synthetic", "output", "manifests")
OUT = os.path.join("supplementary_moved_synthetic", "output", "leakage")
TUPLE = ["Patient Age", "Patient Sex", "View Position"]


class Imgs(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


@torch.no_grad()
def embed(paths, model, device, workers):
    dl = DataLoader(Imgs(paths), batch_size=128, shuffle=False, num_workers=workers, pin_memory=True)
    out = []
    for x in tqdm(dl, desc="embed"):
        f = model.features(x.to(device, non_blocking=True))
        f = torch.nn.functional.adaptive_avg_pool2d(torch.relu(f), 1).flatten(1)
        out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def nearest(q, ref, device, chunk=2048):
    """Cosine distance (1 - cos) from each query row to its nearest reference row, and that row's index."""
    r = ref.to(device)
    d, idx = [], []
    for i in range(0, len(q), chunk):
        s = q[i:i + chunk].to(device) @ r.T
        v, j = s.max(1)
        d.append((1 - v).cpu())
        idx.append(j.cpu())
    return torch.cat(d).numpy(), torch.cat(idx).numpy()


def metadata(tr, va, te):
    q = pd.concat([va.assign(partition="val"), te.assign(partition="test")])
    q = q[q["source"] == "synthetic"].copy()
    key = lambda d: set(map(tuple, d[TUPLE].astype(str).values))
    tr_real_pn = key(tr[(tr.source == "real") & (tr["Finding Labels"] == "Pneumonia")])
    tr_real_all = key(tr[tr.source == "real"])
    tr_syn = key(tr[tr.source == "synthetic"])
    t = list(map(tuple, q[TUPLE].astype(str).values))
    q["in_train_real_pneumonia"] = [x in tr_real_pn for x in t]
    q["in_train_real_any"] = [x in tr_real_all for x in t]
    q["in_train_synthetic"] = [x in tr_syn for x in t]
    tuples = q.groupby(TUPLE).agg(n_rows=("Image Index", "size"),
                                  in_train_real_pneumonia=("in_train_real_pneumonia", "first"),
                                  in_train_synthetic=("in_train_synthetic", "first")).reset_index()
    tuples.sort_values("n_rows", ascending=False).to_csv(os.path.join(OUT, "metadata_tuples.csv"), index=False)
    summ = dict(
        n_val_test_synthetic_rows=len(q),
        n_distinct_tuples=int(len(tuples)),
        n_distinct_tuples_in_232_train_real_pneumonia=len(tr_real_pn),
        rows_exact_copy_of_train_real_pneumonia_row=int(q["in_train_real_pneumonia"].sum()),
        rows_exact_copy_of_any_train_real_row=int(q["in_train_real_any"].sum()),
        rows_exact_copy_of_train_synthetic_row=int(q["in_train_synthetic"].sum()),
        distinct_tuples_exact_copy_of_train_real_pneumonia=int(tuples["in_train_real_pneumonia"].sum()),
        per_partition={p: dict(rows=int((q.partition == p).sum()),
                               distinct=int(q[q.partition == p].groupby(TUPLE).ngroups)) for p in ("val", "test")})
    json.dump(summ, open(os.path.join(OUT, "metadata_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only the first N rows of each manifest")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tr, va, te = (pd.read_csv(os.path.join(MAN, f)) for f in
                  ("combined_training_labels.csv", "validation_labels.csv", "testing_labels.csv"))
    metadata(tr, va, te)
    if a.limit:
        tr, va, te = (d.groupby("source", group_keys=False).head(a.limit) for d in (tr, va, te))

    torch.manual_seed(0)
    device = torch.device("cuda")
    model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    q = pd.concat([va.assign(partition="val"), te.assign(partition="test")], ignore_index=True)
    E_tr = embed(tr["path"].tolist(), model, device, a.workers)
    E_q = embed(q["path"].tolist(), model, device, a.workers)
    tr_real, tr_syn = (tr["source"] == "real").values, (tr["source"] == "synthetic").values
    d_r, i_r = nearest(E_q, E_tr[tr_real], device)
    d_s, i_s = nearest(E_q, E_tr[tr_syn], device)
    q["nn_dist_train_real"], q["nn_train_real"] = d_r, tr.loc[tr_real, "Image Index"].values[i_r]
    q["nn_dist_train_synthetic"], q["nn_train_synthetic"] = d_s, tr.loc[tr_syn, "Image Index"].values[i_s]
    q["nn_dist_train_any"] = np.minimum(d_r, d_s)
    q[["Image Index", "partition", "source", "Finding Labels", "nn_dist_train_real", "nn_train_real",
       "nn_dist_train_synthetic", "nn_train_synthetic", "nn_dist_train_any"]].to_csv(
        os.path.join(OUT, "nn_distances.csv"), index=False)

    base = q.loc[q.source == "real", "nn_dist_train_real"].to_numpy()  # independent-patient baseline
    thr = {"p1_real_baseline": float(np.percentile(base, 1)), "p5_real_baseline": float(np.percentile(base, 5))}
    # same idea within the synthetic family: how close are synthetic training images to EACH OTHER (leave-self-out)
    rng = np.random.default_rng(0)
    pick = rng.choice(int(tr_syn.sum()), size=min(5000, int(tr_syn.sum())), replace=False)
    Es = E_tr[tr_syn]
    s = (Es[pick].to(device) @ Es.to(device).T)
    s[torch.arange(len(pick)), torch.as_tensor(pick, device=device)] = -2  # drop self-match
    syn_self = (1 - s.max(1).values).cpu().numpy()
    thr["p1_synthetic_train_to_synthetic_train"] = float(np.percentile(syn_self, 1))
    json.dump(dict(encoder="torchvision densenet121 IMAGENET1K_V1, frozen, GAP 1024-d, L2-normalised",
                   distance="cosine distance = 1 - cosine similarity", thresholds=thr,
                   justification="p1/p5 of nearest-real-training distance of real val/test images, whose patients "
                                 "are disjoint from training by construction (independent baseline)"),
              open(os.path.join(OUT, "nn_thresholds.json"), "w"), indent=2)

    rows = []
    groups = {"real val/test": q.source == "real", "synthetic val": (q.source == "synthetic") & (q.partition == "val"),
              "synthetic test": (q.source == "synthetic") & (q.partition == "test"),
              "synthetic val+test": q.source == "synthetic"}
    for gname, m in groups.items():
        for col in ("nn_dist_train_real", "nn_dist_train_synthetic"):
            v = q.loc[m, col].to_numpy()
            rows.append(dict(group=gname, reference=col.replace("nn_dist_", ""), n=len(v), mean=v.mean(),
                             median=np.median(v), p1=np.percentile(v, 1), p5=np.percentile(v, 5),
                             p25=np.percentile(v, 25), p75=np.percentile(v, 75), min=v.min(),
                             frac_below_p1_real=float((v < thr["p1_real_baseline"]).mean()),
                             frac_below_p5_real=float((v < thr["p5_real_baseline"]).mean())))
    rows.append(dict(group="synthetic train (5,000 sampled, self excluded)", reference="train_synthetic",
                     n=len(syn_self), mean=syn_self.mean(), median=np.median(syn_self),
                     p1=np.percentile(syn_self, 1), p5=np.percentile(syn_self, 5), p25=np.percentile(syn_self, 25),
                     p75=np.percentile(syn_self, 75), min=syn_self.min(),
                     frac_below_p1_real=float((syn_self < thr["p1_real_baseline"]).mean()),
                     frac_below_p5_real=float((syn_self < thr["p5_real_baseline"]).mean())))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "nn_summary.csv"), index=False)
    print(json.dumps(thr, indent=2))
    print(summ.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for k, col in enumerate(("nn_dist_train_real", "nn_dist_train_synthetic")):
        for gname, m, c in (("real val/test", q.source == "real", "#4C72B0"),
                            ("synthetic val/test", q.source == "synthetic", "#DD8452")):
            ax[k].hist(q.loc[m, col], bins=100, density=True, alpha=0.55, label=gname, color=c)
        ax[k].axvline(thr["p1_real_baseline"], color="k", ls="--", lw=1, label="1st pct, real baseline")
        ax[k].set_title(f"Nearest {'real' if k == 0 else 'synthetic'} training image")
        ax[k].set_xlabel("cosine distance")
        ax[k].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nn_distance_hist.png"), dpi=150)


if __name__ == "__main__":
    main()
