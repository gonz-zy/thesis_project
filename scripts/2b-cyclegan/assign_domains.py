"""
assign_domains.py -- gives every validation and test image outside the 676 a domain (artifact | clean).

Why: Setup 4's classifier is trained on the 487 cleaned training images, but its full-size evaluation (run A: 5,985
validation images; primary comparison with Setup 3: 12,262 test images) needs every one of those images to go through
the SAME cleaning. Only 65 + 124 of them are among the 676 whose domains K-Means and the manual check already settled.

How: for each image the exact feature code of features.py (same mask, same HOG, same four scalars) -> the saved per-class
StandardScaler -> the saved per-class K-Means -> cluster -> domain, using the cluster -> domain map of the 676.
Per-class means the image's TRUE label picks which saved model is used, as in Chehade (they cluster per class). A test image's
cleaning therefore depends on its label. State this as a limitation. The two classes have similar artifact shares (44% / 46%).

Safety: before touching any new image the script runs the 676 through the same code and requires that it reproduces their
four scalars and every cluster in clustered_676.csv. If it does not, the extraction differs from the one that made the
clusters and the script stops. Only the cluster id and two distances are kept per image (the 580,648-value vectors are
never stored: 18,000 of them would be about 42 GB).

Reads   clusters\\clustered_676.csv, clusters\\kmeans_pneumonia.joblib, kmeans_no_finding.joblib, features\\scalars_676.csv,
        labels\\validation_labels.csv and testing_labels.csv (2A), the 1024 images in 1-normalized_images
Writes  clusters\\assigned_outside.csv   one row per new image (resume-safe: rerun continues where it stopped)
        clusters\\clustered_all.csv      the 676 plus the new images: Image Index, Finding Labels, split, cluster, domain, origin
        clusters\\assign_report.json

CPU only. Roughly 0.5 s per image per core: about 40 min with --workers 4 (Setup 3 running), about 20 min with 8.

Run (project root):
    venv\\Scripts\\python.exe scripts\\2b-cyclegan\\assign_domains.py --workers 4
    venv\\Scripts\\python.exe scripts\\2b-cyclegan\\assign_domains.py --selfcheck_only        (about 2 min)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from features import SCALAR_NAMES, four_scalars, hog_vector, load_1024
from setup4_config import CLUSTERED_CSV, CLUSTERS_DIR, FEATURES_DIR, SRC_IMG_1024, SRC_SPLIT_CSV

CLASSES = ["Pneumonia", "No Finding"]
SCALARS_CSV = os.path.join(FEATURES_DIR, "scalars_676.csv")
ASSIGNED_CSV = os.path.join(CLUSTERS_DIR, "assigned_outside.csv")
ALL_CSV = os.path.join(CLUSTERS_DIR, "clustered_all.csv")
REPORT_JSON = os.path.join(CLUSTERS_DIR, "assign_report.json")
_MODELS = {}


def model_path(cls):
    return os.path.join(CLUSTERS_DIR, f"kmeans_{cls.replace(' ', '_').lower()}.joblib")


def _init_worker():
    for cls in CLASSES:
        _MODELS[cls] = joblib.load(model_path(cls))


def work(job):
    """job = (key, path, class). Same vector as features.extract(): HOG then the four scalars."""
    key, path, cls = job
    img = load_1024(path)
    scal, fallback = four_scalars(img)
    h = hog_vector(img)
    vec = np.empty((1, len(h) + 4), dtype=np.float32)
    vec[0, :len(h)] = h
    vec[0, len(h):] = scal
    m = _MODELS[cls]
    d = m["kmeans"].transform(m["scaler"].transform(vec))[0]      # distances to the two centres, standardised space
    return key, int(np.argmin(d)), float(d[0]), float(d[1]), scal, bool(fallback)


def run_jobs(jobs, workers, on_batch=None, batch=200):
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for k, res in enumerate(pool.map(work, jobs, chunksize=2), 1):
            out.append(res)
            if k % batch == 0 or k == len(jobs):
                if on_batch:
                    on_batch(out)
                rate = (time.time() - t0) / k
                print(f"  {k}/{len(jobs)}  {rate:.2f} s/image wall  about {(len(jobs) - k) * rate / 60:.0f} min left", flush=True)
    return out


def domain_map(c676):
    """(class, cluster) -> domain, read from the final domains of the 676. Must be unambiguous."""
    m = {}
    for (cls, cl), g in c676.groupby(["Finding Labels", "cluster"]):
        doms = set(g["domain"])
        if len(doms) != 1:
            raise SystemExit(f"({cls}, cluster {cl}) has both domains {doms} in clustered_676.csv")
        m[(cls, int(cl))] = doms.pop()
    for cls in CLASSES:
        if {m.get((cls, 0)), m.get((cls, 1))} != {"artifact", "clean"}:
            raise SystemExit(f"{cls}: clusters 0/1 do not map to one artifact and one clean domain: {m}")
    return m


def path_of(row):
    p = row.get("src_path")
    return p if isinstance(p, str) and os.path.isfile(p) else os.path.join(SRC_IMG_1024, row["Image Index"])


def selfcheck(c676, workers):
    scal = pd.read_csv(SCALARS_CSV).set_index("Image Index")
    jobs = [(r["Image Index"], path_of(r), r["Finding Labels"]) for r in c676.to_dict("records")]
    print(f"self-check: {len(jobs)} images of the 676 through the same code")
    res = run_jobs(jobs, workers)
    got = pd.DataFrame([dict(name=k, cluster=c, **dict(zip(SCALAR_NAMES, s))) for k, c, _, _, s, _ in res]).set_index("name")
    ref = c676.set_index("Image Index")
    scalar_diff = float(np.abs(got[SCALAR_NAMES].to_numpy() - scal.loc[got.index, SCALAR_NAMES].to_numpy()).max())
    bad = got.index[got["cluster"].to_numpy() != ref.loc[got.index, "cluster"].to_numpy()]
    print(f"  max |scalar difference| vs scalars_676.csv: {scalar_diff:.2e}")
    print(f"  clusters reproduced: {len(got) - len(bad)}/{len(got)}")
    if scalar_diff > 1e-5 or len(bad):
        raise SystemExit(f"SELF-CHECK FAILED ({len(bad)} clusters differ, scalar difference {scalar_diff:.2e}). The extraction "
                         f"here is not the one that made clusters; do NOT assign new images. First differing images: {list(bad[:5])}")
    print("  self-check passed: the new images will be assigned by exactly the code that made the 676 clusters")
    return dict(n=len(got), clusters_reproduced=len(got) - len(bad), max_scalar_diff=scalar_diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4, help="4 while Setup 3 trains, 8 when the machine is free")
    ap.add_argument("--selfcheck_only", action="store_true")
    ap.add_argument("--skip_selfcheck", action="store_true")
    ap.add_argument("--restart", action="store_true", help="ignore assigned_outside.csv and start over")
    ap.add_argument("--limit", type=int, default=None, help="tests only: first N outside images")
    args = ap.parse_args()

    c676 = pd.read_csv(CLUSTERED_CSV)
    for c in ("Image Index", "Finding Labels", "cluster", "domain", "split"):
        if c not in c676.columns:
            raise SystemExit(f"{CLUSTERED_CSV}: missing column {c} (run label_grid.py --score for the final domain)")
    dmap = domain_map(c676)
    print("cluster -> domain:", {f"{k[0]}|{k[1]}": v for k, v in dmap.items()})
    report = {"cluster_to_domain": {f"{k[0]}|{k[1]}": v for k, v in dmap.items()}}

    if not args.skip_selfcheck:
        report["selfcheck"] = selfcheck(c676, args.workers)
    if args.selfcheck_only:
        return

    known = set(c676["Image Index"])
    frames = []
    for split in ("validation", "test"):
        d = pd.read_csv(SRC_SPLIT_CSV[split])[["Image Index", "Finding Labels"]]
        bad = sorted(set(d["Finding Labels"]) - set(CLASSES))
        if bad:
            raise SystemExit(f"{SRC_SPLIT_CSV[split]}: labels other than Pneumonia / No Finding: {bad[:5]}")
        d = d[~d["Image Index"].isin(known)].copy()
        d["split"] = split
        frames.append(d)
    todo = pd.concat(frames, ignore_index=True)
    if args.limit:
        todo = todo.head(args.limit)
    print(f"images outside the 676: {len(todo)} ({(todo.split == 'validation').sum()} validation, {(todo.split == 'test').sum()} test)")

    done = pd.DataFrame(columns=["Image Index", "cluster", "dist0", "dist1", "mask_fallback"])
    if os.path.exists(ASSIGNED_CSV) and not args.restart:
        done = pd.read_csv(ASSIGNED_CSV)
        print(f"resuming: {len(done)} already assigned")
    todo_rest = todo[~todo["Image Index"].isin(set(done["Image Index"]))]

    def save(partial):
        new = pd.DataFrame([dict(zip(["Image Index", "cluster", "dist0", "dist1", "scal", "mask_fallback"], r)) for r in partial])
        pd.concat([done, new.drop(columns="scal")], ignore_index=True).to_csv(ASSIGNED_CSV, index=False)

    if len(todo_rest):
        jobs = [(r["Image Index"], os.path.join(SRC_IMG_1024, r["Image Index"]), r["Finding Labels"])
                for r in todo_rest.to_dict("records")]
        missing = [j[1] for j in jobs if not os.path.isfile(j[1])]
        if missing:
            raise SystemExit(f"{len(missing)} 1024 images missing, e.g. {missing[:3]}")
        run_jobs(jobs, args.workers, on_batch=save)
    assigned = pd.read_csv(ASSIGNED_CSV)
    assert len(assigned) >= len(todo) and assigned["Image Index"].is_unique

    new = todo.merge(assigned, on="Image Index", how="left")
    assert new["cluster"].notna().all()
    new["domain"] = [dmap[(c, int(k))] for c, k in zip(new["Finding Labels"], new["cluster"])]
    new["origin"] = "assigned"
    old = c676[["Image Index", "Finding Labels", "split", "cluster", "domain"]].copy()
    old["origin"] = "subset_676"
    full = pd.concat([old, new[["Image Index", "Finding Labels", "split", "cluster", "domain", "origin"]]], ignore_index=True)
    full["cluster"] = full["cluster"].astype(int)
    assert full["Image Index"].is_unique
    full.to_csv(ALL_CSV, index=False)

    print("\ndomain by split and class (artifact share):")
    tab = full.groupby(["origin", "split", "Finding Labels"])["domain"].apply(lambda s: f"{(s == 'artifact').sum()}/{len(s)} = {100 * (s == 'artifact').mean():.0f}%")
    print(tab.to_string())
    share = new.groupby("Finding Labels")["domain"].apply(lambda s: (s == "artifact").mean())
    ref = c676.groupby("Finding Labels")["domain"].apply(lambda s: (s == "artifact").mean())
    for cls in CLASSES:
        if cls in share and abs(share[cls] - ref[cls]) > 0.15:
            print(f"WARNING: {cls}: artifact share {100 * share[cls]:.0f}% outside the 676 vs {100 * ref[cls]:.0f}% inside; "
                  f"look at a few images before cleaning")
    report.update(n_new=len(new), n_total=len(full), mask_fallbacks=int(new["mask_fallback"].sum()),
                  artifact_share_new={k: float(v) for k, v in share.items()})
    json.dump(report, open(REPORT_JSON, "w"), indent=2)
    print(f"\nmask fell back to the whole image for {int(new['mask_fallback'].sum())} new images")
    print(f"wrote {ALL_CSV} ({len(full)} rows)\nNext, after CycleGAN is trained: clean_images.py --manifest {ALL_CSV}")


if __name__ == "__main__":
    main()
