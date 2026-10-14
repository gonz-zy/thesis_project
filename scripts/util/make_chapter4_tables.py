"""
make_chapter4_tables.py -- builds the Chapter 4 tables and figures for the FINISHED parts of the thesis:
    A  data splits           (NIH metadata, the StyleGAN split files, the 676-image Setup 4 subset)
    B  StyleGAN2-ADA training (options, FID curve, loss / ADA curves, samples, synthetic set)
    C  K-clustering          (cluster sizes, quality indices, manual check, final domains, pool purity)

Read-only on your data. Everything is written to  results\\chapter4\\  :
    tables\\*.csv        one CSV per table  (open in Excel; paste into the manuscript)
    figures\\*.png       figures
    chapter4_notes.md   what was produced, what was skipped and why, and the assumptions

Run from the project root (second terminal if Setup 3 is training; this is CPU-only and takes seconds):
    venv\\Scripts\\python.exe scripts\\util\\make_chapter4_tables.py
    venv\\Scripts\\python.exe scripts\\util\\make_chapter4_tables.py --only C        (one part)
    venv\\Scripts\\python.exe scripts\\util\\make_chapter4_tables.py --root D:\\path\\to\\thesis_project

Every step is independent: a missing file or an unexpected column makes THAT step print SKIPPED with the
reason and the script carries on. Nothing is typed by hand; every number comes from your files.
Needs pandas and numpy; figures need matplotlib and Pillow (pip install matplotlib pillow).
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:                                    # tables still work without it
    plt = None
try:
    from PIL import Image
except Exception:
    Image = None

CHEHADE = dict(
    sizes={"Pneumonia": (152, 170), "No Finding": (133, 221)},       # (artifact, clean), their Table 1
    agreement={"Pneumonia": 83.54, "No Finding": 82.49},             # their Fig. 5
    total=(285, 391),
)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
ap.add_argument("--only", default="ABC", help="which parts to run, e.g. A, B, C, AC")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--entry", default=None, help="path to the NIH Data_Entry csv if the script cannot find it")
ARGS = ap.parse_args()

ROOT = os.path.abspath(ARGS.root)
DATA = os.path.join(ROOT, "data")
RAW_CSV = os.path.join(DATA, "1-raw", "Data_Entry_2017_v020.csv")      # default; entry() searches if absent
A_DIR = os.path.join(DATA, "2-processed", "2A-input_for_stylegan2ada")
B_DIR = os.path.join(DATA, "2-processed", "2B-input_for_cyclegan")
SG_DIR = os.path.join(ROOT, "results", "stylegan2-ada")
OUT = os.path.join(ROOT, "results", "chapter4")
TABLES, FIGS = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")
for d in (TABLES, FIGS):
    os.makedirs(d, exist_ok=True)

LOG = []


def log(status, name, msg=""):
    LOG.append((status, name, msg))
    print(f"[{status:<7}] {name}  {msg}")


def table(df, name, index=False):
    df.to_csv(os.path.join(TABLES, name + ".csv"), index=index)
    log("table", name, f"{len(df)} rows")
    return df


def save_fig(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log("figure", name)


def rd(path, **kw):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kw)


def rj(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def norm_class(s):
    s = s.astype(str).str.strip().str.lower()
    out = pd.Series("Other", index=s.index)
    out[s.str.contains("no finding") | s.isin(["0", "0.0", "normal", "no_finding"])] = "No Finding"
    out[(s.str.contains("pneumonia")) | s.isin(["1", "1.0"])] = "Pneumonia"
    return out


def class_of(df):
    for c in ["Finding Labels", "Label", "label", "class", "Class", "target", "Target", "Finding"]:
        if c in df.columns:
            return norm_class(df[c])
    raise KeyError(f"no class column found in {list(df.columns)}")


def pid_of(df):
    if "Patient ID" not in df.columns:
        raise KeyError(f"no 'Patient ID' column in {list(df.columns)}")
    return df["Patient ID"].astype(str).str.strip()


_ENTRY = {}


def find_entry_csv():
    if ARGS.entry:
        return ARGS.entry
    if os.path.isfile(RAW_CSV):
        return RAW_CSV
    hits = glob.glob(os.path.join(DATA, "**", "Data_Entry*.csv"), recursive=True) + \
        glob.glob(os.path.join(ROOT, "**", "Data_Entry*.csv"), recursive=True)
    hits = [h for h in hits if "venv" not in h]
    if not hits:
        raise FileNotFoundError("no Data_Entry*.csv under data\\ or the project root; pass --entry <path>")
    return hits[0]


def entry():
    if "df" not in _ENTRY:
        path = find_entry_csv()
        log("NOTE", "NIH metadata file", os.path.relpath(path, ROOT) if os.path.isabs(path) and path.startswith(ROOT) else path)
        _ENTRY["df"] = rd(path)
    return _ENTRY["df"]


def with_meta(df):
    """Add age / sex / view position from the NIH file when the split file does not carry them."""
    df = df.copy()
    need = [c for c in ["Patient Age", "Patient Gender", "View Position"] if c not in df.columns]
    if "Patient Sex" in df.columns and "Patient Gender" in need:
        need.remove("Patient Gender")                 # the 676 files call it Patient Sex
    if need and "Image Index" in df.columns:
        e = entry().drop_duplicates("Image Index").set_index("Image Index")
        for c in need:
            if c in e.columns:
                df[c] = df["Image Index"].map(e[c])
    return df


def demog(df, label):
    df = with_meta(df)
    row = {"set": label, "images": len(df)}
    if "Patient ID" in df.columns:
        row["patients"] = df["Patient ID"].astype(str).nunique()
    if "Patient Age" in df.columns:
        a = pd.to_numeric(df["Patient Age"], errors="coerce")
        row.update(age_mean=round(a.mean(), 1), age_sd=round(a.std(), 1), age_median=a.median(),
                   age_min=a.min(), age_max=a.max())
    gcol = next((c for c in ["Patient Gender", "Patient Sex"] if c in df.columns), None)
    if gcol:
        row["male_pct"] = round(100 * (df[gcol].astype(str).str.upper().str.startswith("M")).mean(), 1)
    if "View Position" in df.columns:
        v = df["View Position"].astype(str).str.upper()
        row["PA_pct"] = round(100 * (v == "PA").mean(), 1)
        row["AP_pct"] = round(100 * (v == "AP").mean(), 1)
    return row


STEPS = []


def step(part):
    def deco(fn):
        STEPS.append((part, fn))
        return fn
    return deco


# =====================================================================================================
# A. DATA SPLITS
# =====================================================================================================
def split_frames():
    return {k: rd(os.path.join(A_DIR, "labels", f)) for k, f in
            [("train", "training_labels.csv"), ("validation", "validation_labels.csv"),
             ("test", "testing_labels.csv")]}


@step("A")
def a1_dataset_flow():
    e = entry()
    lab = e["Finding Labels"].astype(str)
    pneu_only = e[lab == "Pneumonia"]
    nf = e[lab == "No Finding"]
    rows = [
        ("NIH ChestX-ray14, all images", len(e), e["Patient ID"].nunique()),
        ("... label is exactly 'Pneumonia'", len(pneu_only), pneu_only["Patient ID"].nunique()),
        ("... label is 'No Finding'", len(nf), nf["Patient ID"].nunique()),
        ("... any label that contains Pneumonia (multi-label, not used if the pipeline kept single-label only)",
         int(lab.str.contains("Pneumonia").sum()), e.loc[lab.str.contains("Pneumonia"), "Patient ID"].nunique()),
    ]
    fp = os.path.join(DATA, "2-processed", "filtered_labels.csv")
    if os.path.isfile(fp):
        f = pd.read_csv(fp)
        c = class_of(f)
        rows.append(("filtered_labels.csv (what the pipeline kept)", len(f),
                     f["Patient ID"].astype(str).nunique() if "Patient ID" in f else np.nan))
        for k in ["Pneumonia", "No Finding"]:
            rows.append((f"   ... {k}", int((c == k).sum()),
                         f.loc[c == k, "Patient ID"].astype(str).nunique() if "Patient ID" in f else np.nan))
    table(pd.DataFrame(rows, columns=["step", "images", "patients"]), "a1_dataset_flow")


@step("A")
def a2_split_counts():
    sp = split_frames()
    rows = []
    for k, df in sp.items():
        c = class_of(df)
        rows.append(dict(split=k, images=len(df), patients=pid_of(df).nunique(),
                         Pneumonia=int((c == "Pneumonia").sum()), No_Finding=int((c == "No Finding").sum()),
                         pneumonia_pct=round(100 * (c == "Pneumonia").mean(), 2)))
    t = pd.DataFrame(rows)
    tot = dict(split="total (patients counted once)", images=t["images"].sum(),
               patients=pd.concat([pid_of(d) for d in sp.values()]).nunique(),
               Pneumonia=t["Pneumonia"].sum(), No_Finding=t["No_Finding"].sum())
    tot["pneumonia_pct"] = round(100 * tot["Pneumonia"] / tot["images"], 2)
    table(pd.concat([t, pd.DataFrame([tot])]), "a2_split_counts")

    cp = os.path.join(A_DIR, "labels", "combined_training_labels.csv")
    if os.path.isfile(cp):
        cb = pd.read_csv(cp)
        c = class_of(cb)
        syn = pid_of(cb).str.upper().str.startswith("SYNTH")
        sp_path = os.path.join(DATA, "2-processed", "synthetic_labels.csv")
        if os.path.isfile(sp_path):
            s = pd.read_csv(sp_path)
            if "Image Index" in s.columns and "Image Index" in cb.columns:
                syn = syn | cb["Image Index"].isin(set(s["Image Index"]))
        r = []
        for name, m in [("real", ~syn), ("synthetic", syn), ("combined", pd.Series(True, index=cb.index))]:
            r.append(dict(part=name, images=int(m.sum()), Pneumonia=int(((c == "Pneumonia") & m).sum()),
                          No_Finding=int(((c == "No Finding") & m).sum())))
        t2 = pd.DataFrame(r)
        t2["pneumonia_pct"] = (100 * t2["Pneumonia"] / t2["images"]).round(2)
        table(t2, "a2b_combined_training_set")


@step("A")
def a3_leakage():
    sp = split_frames()
    ids = {k: set(pid_of(v)) for k, v in sp.items()}
    imgs = {k: set(v["Image Index"]) for k, v in sp.items()} if all("Image Index" in v for v in sp.values()) else {}
    rows = []
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        r = dict(pair=f"{a} / {b}", patient_overlap=len(ids[a] & ids[b]))
        if imgs:
            r["image_overlap"] = len(imgs[a] & imgs[b])
        rows.append(r)
    table(pd.DataFrame(rows), "a3_patient_leakage")


@step("A")
def a4_demographics():
    sp = split_frames()
    rows = [demog(df, k) for k, df in sp.items()]
    table(pd.DataFrame(rows), "a4_demographics")


@step("A")
def a5_setup4_subset():
    sub = rd(os.path.join(B_DIR, "labels", "subset_676.csv"))
    c = class_of(sub)
    sub["_c"] = c
    sp_col = "split" if "split" in sub.columns else None
    rows = []
    groups = sub.groupby(sp_col) if sp_col else [("all", sub)]
    for k, g in groups:
        rows.append(dict(split=k, images=len(g), patients=pid_of(g).nunique() if "Patient ID" in g else np.nan,
                         Pneumonia=int((g["_c"] == "Pneumonia").sum()), No_Finding=int((g["_c"] == "No Finding").sum())))
    rows.append(dict(split="total", images=len(sub), patients=pid_of(sub).nunique() if "Patient ID" in sub else np.nan,
                     Pneumonia=int((c == "Pneumonia").sum()), No_Finding=int((c == "No Finding").sum())))
    t = pd.DataFrame(rows)
    table(t, "a5_setup4_subset")

    extra = []
    if "Patient ID" in sub.columns and sp_col:
        s = {k: set(pid_of(g)) for k, g in sub.groupby(sp_col)}
        ks = list(s)
        ov = sum(len(s[a] & s[b]) for i, a in enumerate(ks) for b in ks[i + 1:])
        extra.append(("patients shared across splits", ov))
        mixed = sub.groupby(pid_of(sub))["_c"].nunique()
        extra.append(("patients with both classes", int((mixed > 1).sum())))
    for f in ["test_balanced_124.csv", "validation_small_65.csv"]:
        p = os.path.join(B_DIR, "labels", f)
        if os.path.isfile(p):
            d = pd.read_csv(p)
            cc = class_of(d)
            extra.append((f"{f}: images / pneumonia / no finding",
                          f"{len(d)} / {(cc == 'Pneumonia').sum()} / {(cc == 'No Finding').sum()}"))
    if extra:
        table(pd.DataFrame(extra, columns=["check", "value"]), "a5b_setup4_checks")
    dm = [demog(g, k) for k, g in groups] if sp_col else [demog(sub, "all")]
    table(pd.DataFrame(dm), "a5c_setup4_demographics")


@step("A")
def a6_train_stats():
    js = rj(os.path.join(DATA, "2-processed", "train_stats.json"))
    rows = []

    def walk(prefix, o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(f"{prefix}{k}.", v)
        else:
            rows.append((prefix.rstrip("."), o))
    walk("", js)
    table(pd.DataFrame(rows, columns=["item", "value"]), "a6_train_stats")


# =====================================================================================================
# B. STYLEGAN2-ADA TRAINING
# =====================================================================================================
def find_one(name, root=SG_DIR):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(os.path.join(root, "**", name))
    return max(hits, key=os.path.getmtime)


def jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def _val(rec, key):
    v = rec.get(key)
    if isinstance(v, dict):
        v = v.get("mean")
    return v


@step("B")
def b1_options():
    js = rj(find_one("training_options.json"))
    rows = []

    def walk(prefix, o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(f"{prefix}{k}.", v)
        else:
            rows.append((prefix.rstrip("."), o))
    walk("", js)
    t = pd.DataFrame(rows, columns=["option", "value"])
    table(t, "b1_stylegan_options_full")
    keep = ["total_kimg", "batch_size", "batch_gpu", "ema_kimg", "ada_target", "random_seed", "num_gpus",
            "G_kwargs.channel_base", "G_kwargs.channel_max", "G_opt_kwargs.lr", "D_opt_kwargs.lr",
            "loss_kwargs.r1_gamma", "training_set_kwargs.resolution", "training_set_kwargs.use_labels",
            "training_set_kwargs.xflip", "training_set_kwargs.max_size", "augment_kwargs.p", "metrics"]
    k = t[t["option"].apply(lambda x: any(x == p or x.endswith(p) for p in keep))]
    table(k, "b1_stylegan_options_key")


@step("B")
def b2_fid():
    recs = [r for r in jsonl(find_one("metric-fid50k_full.jsonl")) if "fid50k_full" in (r.get("results") or {})]
    if not recs:
        raise ValueError("no fid50k_full entries in the metric file")
    rows = []
    for r in recs:
        m = re.search(r"network-snapshot-(\d+)", str(r.get("snapshot_pkl", "")))
        rows.append(dict(snapshot=r.get("snapshot_pkl"), kimg=int(m.group(1)) if m else np.nan,
                         fid50k_full=round(float(r["results"]["fid50k_full"]), 3),
                         eval_minutes=round(float(r.get("total_time", np.nan)) / 60, 1) if r.get("total_time") else np.nan))
    t = pd.DataFrame(rows).sort_values("kimg").reset_index(drop=True)
    t["best"] = t["fid50k_full"] == t["fid50k_full"].min()
    table(t, "b2_fid_by_snapshot")
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(t["kimg"], t["fid50k_full"], marker="o", ms=3, lw=1.4, color="#2b6cb0")
        b = t.loc[t["fid50k_full"].idxmin()]
        ax.scatter([b["kimg"]], [b["fid50k_full"]], color="#c53030", zorder=3,
                   label=f"best: {b['fid50k_full']:.2f} at {int(b['kimg'])} kimg")
        ax.set_xlabel("training progress (kimg)")
        ax.set_ylabel("FID (50k, full)")
        ax.set_title("StyleGAN2-ADA: FID by snapshot")
        ax.grid(alpha=.3)
        ax.legend()
        save_fig(fig, "b2_fid_curve")


@step("B")
def b3_training_curves():
    recs = jsonl(find_one("stats.jsonl"))
    kimg = [_val(r, "Progress/kimg") for r in recs]
    series = {"G loss": "Loss/G/loss", "D loss": "Loss/D/loss", "ADA p": "Progress/augment"}
    data = {k: [_val(r, v) for r in recs] for k, v in series.items()}
    df = pd.DataFrame({"kimg": kimg, **data}).dropna(subset=["kimg"])
    last = recs[-1]
    summ = [("ticks logged", len(recs)),
            ("total kimg", df["kimg"].max()),
            ("final ADA p", round(float(df["ADA p"].dropna().iloc[-1]), 3) if df["ADA p"].notna().any() else np.nan),
            ("mean ADA p, last 20% of training",
             round(float(df["ADA p"].tail(max(1, len(df) // 5)).mean()), 3) if df["ADA p"].notna().any() else np.nan)]
    th = _val(last, "Timing/total_hours")
    if th is not None:
        summ.append(("training time (hours)", round(float(th), 2)))
    table(pd.DataFrame(summ, columns=["item", "value"]), "b3_training_summary")
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, k in zip(axes, series):
            ax.plot(df["kimg"], df[k], lw=1.1)
            ax.set_title(k)
            ax.set_xlabel("kimg")
            ax.grid(alpha=.3)
        fig.suptitle("StyleGAN2-ADA training curves", y=1.03)
        save_fig(fig, "b3_training_curves")


@step("B")
def b4_log_excerpt():
    lines = open(find_one("log.txt"), errors="ignore").read().splitlines()
    keys = ("Number of GPUs", "Training duration", "Number of images", "Image resolution", "Conditional model",
            "Dataset x-flips", "Output directory", "Training data", "Training options", "GPU", "Tick 0 ")
    keep = [ln for ln in lines if any(k in ln for k in keys)][:40]
    if len(keep) < 3:                                   # unfamiliar log layout: keep the head and tail instead
        keep = lines[:25] + ["..."] + lines[-10:]
    with open(os.path.join(TABLES, "b4_log_excerpt.txt"), "w") as f:
        f.write("\n".join(keep))
    log("table", "b4_log_excerpt.txt", f"{len(keep)} lines")


def _imgs(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg")))


@step("B")
def b5_synthetic_set_and_samples():
    gen = _imgs(os.path.join(A_DIR, "generated-pneumonia"))
    real = _imgs(os.path.join(A_DIR, "training_pneumonia_only"))
    table(pd.DataFrame([("real pneumonia images used to train StyleGAN2-ADA (training_pneumonia_only)", len(real)),
                        ("generated pneumonia images (generated-pneumonia)", len(gen))],
                       columns=["item", "images"]), "b5_synthetic_set")
    if Image is not None and plt is not None and gen and real:
        rng = np.random.default_rng(ARGS.seed)
        n = 8
        pick_g = [gen[i] for i in rng.choice(len(gen), min(n, len(gen)), replace=False)]
        pick_r = [real[i] for i in rng.choice(len(real), min(n, len(real)), replace=False)]
        fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
        for row, (picks, title) in enumerate([(pick_r, "real"), (pick_g, "generated")]):
            for j in range(n):
                ax = axes[row, j]
                ax.axis("off")
                if j < len(picks):
                    ax.imshow(Image.open(picks[j]).convert("L"), cmap="gray")
            axes[row, 0].set_title(title, loc="left", fontsize=10)
        save_fig(fig, "b5_real_vs_generated")
    reals = glob.glob(os.path.join(SG_DIR, "**", "reals.png"), recursive=True)
    if reals:
        shutil.copy(reals[0], os.path.join(FIGS, "b5_snapshot_reals.png"))
        log("figure", "b5_snapshot_reals.png", f"copied from {os.path.relpath(reals[0], ROOT)}")
    fakes = []
    for h in glob.glob(os.path.join(SG_DIR, "**", "fakes*.png"), recursive=True):
        m = re.search(r"fakes(\d+)\.png$", os.path.basename(h))
        if m:                                           # numbered grids only; fakes_init.png is the untrained generator
            fakes.append((int(m.group(1)), h))
    if fakes:
        k, src = max(fakes)
        shutil.copy(src, os.path.join(FIGS, "b5_snapshot_fakes_last.png"))
        log("figure", "b5_snapshot_fakes_last.png", f"kimg {k}, copied from {os.path.relpath(src, ROOT)}")
        first = min(fakes)
        shutil.copy(first[1], os.path.join(FIGS, "b5_snapshot_fakes_first.png"))
    else:
        log("NOTE", "b5 fakes", "no numbered fakesNNNNNN.png found (only fakes_init.png?)")


# =====================================================================================================
# C. K-CLUSTERING
# =====================================================================================================
CLASSES = ["Pneumonia", "No Finding"]


@step("C")
def c1_sizes_quality():
    r = rj(os.path.join(B_DIR, "clusters", "cluster_report.json"))
    rows, qrows, srows = [], [], []
    for cls in CLASSES:
        p = r["per_class"][cls]
        sizes = {int(k): v for k, v in p["sizes"].items()}
        art = p["provisional_artifact_cluster"]
        rows.append(dict(**{"class": cls}, n=p["n"], cluster0=sizes.get(0), cluster1=sizes.get(1),
                         provisional_artifact_cluster=art,
                         ours_artifact=sizes.get(art), ours_clean=p["n"] - sizes.get(art, 0),
                         chehade_artifact=p.get("chehade_artifact"), chehade_clean=p.get("chehade_clean")))
        q = p.get("quality") or {}
        qrows.append({"class": cls, "silhouette": q.get("silhouette"), "calinski_harabasz": q.get("calinski_harabasz"),
                      "davies_bouldin": q.get("davies_bouldin"),
                      "scalar_only_agreement_pct": round(100 * r.get("scalar_only_agreement", {}).get(cls, np.nan), 1),
                      "inertia": p.get("inertia")})
        for k, d in p["scalar_means"].items():
            srows.append({"class": cls, "cluster": int(k), **{a: round(b, 5) for a, b in d.items()}})
    t = pd.DataFrame(rows)
    tot = dict(**{"class": "total"}, n=t["n"].sum(), ours_artifact=t["ours_artifact"].sum(),
               ours_clean=t["ours_clean"].sum(), chehade_artifact=t["chehade_artifact"].sum(),
               chehade_clean=t["chehade_clean"].sum())
    table(pd.concat([t, pd.DataFrame([tot])]), "c1_cluster_sizes")
    q = pd.DataFrame(qrows)
    table(q, "c2_cluster_quality")
    if q[["silhouette", "calinski_harabasz", "davies_bouldin"]].isna().all().all():
        log("NOTE", "c2_cluster_quality", "quality is empty: rerun cluster_kmeans.py --quality, then label_grid.py --score")
    table(pd.DataFrame(srows), "c3_scalar_means_per_cluster")
    t2 = [("standardisation", r.get("standardisation")), ("seed", r.get("seed"))]
    fr = os.path.join(B_DIR, "features", "feature_report.json")
    if os.path.isfile(fr):
        for k, v in rj(fr).items():
            t2.append((k, v))
    table(pd.DataFrame(t2, columns=["setting", "value"]), "c9_feature_and_cluster_settings")


@step("C")
def c4_manual_check():
    r = rj(os.path.join(B_DIR, "label_grids", "label_report.json"))
    rows = []
    for cls in CLASSES:
        p = r["per_class"][cls]
        share = {int(k): v for k, v in p["artifact_share_per_cluster"].items()}
        ct = p["contingency"]                       # rows = clusters sorted, cols = [artifact, clean]
        lo, hi = p["agreement_ci95"]
        rows.append({"class": cls, "n_marked": p["n_marked"], "artifact_cluster": p["artifact_cluster"],
                     "cluster0_artifact_pct": round(100 * share.get(0, np.nan), 1),
                     "cluster1_artifact_pct": round(100 * share.get(1, np.nan), 1),
                     "cluster0_artifact_n": f"{ct[0][0]} of {ct[0][0] + ct[0][1]}",
                     "cluster1_artifact_n": f"{ct[1][0]} of {ct[1][0] + ct[1][1]}",
                     "fisher_p": round(p["fisher_p"], 4),
                     "agreement_pct": round(100 * p["agreement"], 1),
                     "agreement_ci95": f"{100 * lo:.0f}-{100 * hi:.0f}",
                     "chehade_agreement_pct": p["chehade_agreement"],
                     "overturned_provisional": p["overturned"]})
    table(pd.DataFrame(rows), "c4_manual_check")
    m = rd(os.path.join(B_DIR, "label_grids", "manual_labels.csv"))
    m["m"] = m["manual"].astype(str).str.strip().str.lower()
    g = m.groupby(["round", "cls"]).agg(shown=("m", "size"), marked=("m", lambda s: int(s.isin(["a", "c"]).sum())),
                                        artifact=("m", lambda s: int((s == "a").sum()))).reset_index()
    table(g, "c5_marks_by_round")

    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
        for ax, cls in zip(axes, CLASSES):
            p = r["per_class"][cls]
            ct = p["contingency"]
            vals, errs = [], []
            for k in (0, 1):
                a, c = ct[k]
                lo, hi = wilson(a, a + c)
                vals.append(100 * a / (a + c))
                errs.append([100 * a / (a + c) - 100 * lo, 100 * hi - 100 * a / (a + c)])
            ax.bar(["cluster 0", "cluster 1"], vals, yerr=np.array(errs).T, capsize=4,
                   color=["#9ec5e8", "#e8a09e"])
            ax.set_title(f"{cls}  (Fisher p = {p['fisher_p']:.3f})")
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=.3)
        axes[0].set_ylabel("images you marked as artifact (%)")
        fig.suptitle("Manual check: artifact share per cluster, 95% Wilson CI", y=1.02)
        save_fig(fig, "c4_artifact_share_per_cluster")


@step("C")
def c6_domains_and_purity():
    d = rd(os.path.join(B_DIR, "clusters", "clustered_676.csv"))
    ct = pd.crosstab(d["Finding Labels"], d["domain"])
    ct["artifact_pct"] = (100 * ct["artifact"] / (ct["artifact"] + ct["clean"])).round(1)
    ct["chehade_artifact_pct"] = [round(100 * CHEHADE["sizes"][k][0] / sum(CHEHADE["sizes"][k]), 1) for k in ct.index]
    table(ct.reset_index(), "c6_final_domains")
    if "split" in d.columns:
        t = pd.crosstab(d["split"], d["domain"]).reset_index()
        table(t, "c7_domains_by_split")

    m = rd(os.path.join(B_DIR, "label_grids", "manual_labels.csv"))
    m["m"] = m["manual"].astype(str).str.strip().str.lower()
    m = m[m["m"].isin(["a", "c"])].merge(d[["Image Index", "cluster", "domain"]], left_on="image",
                                         right_on="Image Index", how="left", suffixes=("_old", ""))
    rows = []
    for (cls, k), g in m.groupby(["cls", "cluster"]):
        n_total = int(((d["Finding Labels"] == cls) & (d["cluster"] == k)).sum())
        a = int((g["m"] == "a").sum())
        lo, hi = wilson(a, len(g))
        dom = d.loc[(d["Finding Labels"] == cls) & (d["cluster"] == k), "domain"].iloc[0]
        rows.append(dict(**{"class": cls}, cluster=int(k), domain=dom, cluster_images=n_total, marked=len(g),
                         marked_artifact=a, artifact_share_pct=round(100 * a / len(g), 1),
                         ci95=f"{100 * lo:.0f}-{100 * hi:.0f}",
                         est_true_artifacts=round(a / len(g) * n_total, 1)))
    t = pd.DataFrame(rows)
    pool = t.groupby("domain").agg(images=("cluster_images", "sum"), est_true_artifacts=("est_true_artifacts", "sum")).reset_index()
    pool["est_artifact_share_pct"] = (100 * pool["est_true_artifacts"] / pool["images"]).round(1)
    pool["est_true_artifacts"] = pool["est_true_artifacts"].round(0)
    table(t, "c8_purity_per_cluster")
    table(pool, "c8b_purity_per_domain_pool")
    log("NOTE", "c8_purity", "estimates scale 50-image samples per cluster up to the whole cluster; ballpark only")


@step("C")
def c10_figures():
    src = os.path.join(B_DIR, "label_grids")
    dst = os.path.join(FIGS, "cluster_grids")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(src, "grid_*.png"))):
        shutil.copy(p, dst)
        n += 1
    log("figure", "cluster_grids/", f"{n} contact sheets copied")
    if Image is not None and plt is not None:
        ims = sorted(glob.glob(os.path.join(src, "masks", "mask_*.png")))[:20]
        if ims:
            cols = 5
            rows = int(np.ceil(len(ims) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
            for ax in np.ravel(axes):
                ax.axis("off")
            for ax, p in zip(np.ravel(axes), ims):
                ax.imshow(Image.open(p))
            fig.suptitle("Lung-mask overlays (red = mask), random sample of the 676 images", y=1.0)
            save_fig(fig, "c10_lung_mask_overlays")


def main():
    t0 = time.time()
    if plt is None:
        log("NOTE", "figures", "matplotlib missing: tables only (pip install matplotlib pillow)")
    for part, fn in STEPS:
        if part not in ARGS.only.upper():
            continue
        try:
            fn()
        except Exception as e:
            log("SKIPPED", fn.__name__, f"{type(e).__name__}: {e}")

    ok = [x for x in LOG if x[0] in ("table", "figure")]
    skipped = [x for x in LOG if x[0] == "SKIPPED"]
    notes = [x for x in LOG if x[0] == "NOTE"]
    with open(os.path.join(OUT, "chapter4_notes.md"), "w") as f:
        f.write(f"# Chapter 4 tables and figures (parts {ARGS.only.upper()})\n\n"
                f"Generated {time.strftime('%Y-%m-%d %H:%M')} by make_chapter4_tables.py from the files under {ROOT}\n\n")
        f.write(f"## Produced ({len(ok)})\n")
        f.writelines(f"- {s}: {n} {m}\n" for s, n, m in ok)
        f.write(f"\n## Skipped ({len(skipped)})\n")
        f.writelines(f"- {n}: {m}\n" for s, n, m in skipped)
        f.write(f"\n## Notes\n")
        f.writelines(f"- {n}: {m}\n" for s, n, m in notes)
        f.write("\n## Assumptions to state in the thesis\n"
                "- Class labels are read from the split files; 'Pneumonia' and 'No Finding' are matched by name.\n"
                "- Purity figures (c8) scale 50-image mark samples to whole clusters; they are estimates.\n"
                "- Chehade's cluster sizes and agreement are typed in once at the top of this script from their Table 1 and Fig. 5.\n"
                "- Any table that is missing was skipped because its source file or column was not found (see above).\n")
    print(f"\n{len(ok)} outputs, {len(skipped)} skipped, {time.time() - t0:.1f} s.  See {os.path.relpath(OUT, ROOT)}\\chapter4_notes.md")


if __name__ == "__main__":
    main()
