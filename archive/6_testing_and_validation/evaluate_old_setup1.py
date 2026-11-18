"""
evaluate.py -- statistics on the SAVED test predictions. No training, no GPU, takes minutes.

For every setup (a folder holding seed*/test_predictions_*.csv from run_experiments.py --eval_test) it reports:
  1. per-seed test metrics                      (AUC, accuracy, precision, recall, specificity, F1)
  2. the 5-seed ENSEMBLE (mean probability) with a STRATIFIED bootstrap: 1,000 resamples, pneumonia and
     No Finding images drawn separately so every resample keeps the real class counts (59 / 12,203).
     CI = estimate +/- 1.96 x SD of the resamples (manuscript). Every resample's metrics are saved.
  3. DeLong test for AUC between every pair of setups (ensemble predictions), Holm-corrected p-values
  4. ANOVA, both versions the manuscript reports:
       (a) across the 5 SEEDS of each setup  (+ Tukey post-hoc when 3 or more setups) -- training randomness
       (b) across the 1,000 BOOTSTRAP resamples, with eta-squared -- test-set sampling noise.
           Resamples overlap heavily, so p-values here are tiny for small differences: read eta-squared and the
           CIs, not just p.
  5. Chehade et al. (2025) printed beside the results as a DESCRIPTIVE reference only (they published no
     per-image predictions, so neither DeLong nor ANOVA can include them).

Threshold: Youden's J on VALIDATION predictions only (the ensemble uses the mean of the seeds' validation
probabilities; if val_predictions_*.csv are missing, the mean of the seeds' thresholds is used and a note is printed).
All setups are aligned by Image Index; the script stops if their test cases or labels differ (DeLong would be invalid).
The prediction columns may be named  Image Index|image_id,  label,  prob_pneumonia|prob.

Run (project root):
    python scripts\\3a-densenet121_stylegan2-ada\\evaluate.py ^
        --setup S1_densenet121=results\\densenet121_stylegan2-ada\\final\\cbam_all ^
        --setup S3_convnextv2=results\\convnextv2_stylegan2-ada\\final\\cbam_all
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import roc_auc_score, roc_curve

METRICS = ["auc", "accuracy", "precision", "recall", "specificity", "f1"]
ALIASES = {"image_id": "Image Index", "prob": "prob_pneumonia"}
Z95 = float(ss.norm.ppf(0.975))  # 1.96


# ------------------------------------------------------------------ loading
def read_pred(path):
    d = pd.read_csv(path).rename(columns=ALIASES)
    for c in ("Image Index", "label", "prob_pneumonia"):
        assert c in d.columns, f"{path}: missing column {c}"
    return d[["Image Index", "label", "prob_pneumonia"]].sort_values("Image Index").reset_index(drop=True)


def load_setup(folder):
    """Returns per-seed prediction frames, thresholds, and validation frames (or None)."""
    files = sorted(f for f in glob.glob(os.path.join(folder, "**", "test_predictions_*.csv"), recursive=True))
    assert files, f"no test_predictions_*.csv under {folder} (did the runs use --eval_test?)"
    seeds, preds, thr, vals = [], [], [], []
    for f in files:
        d = os.path.dirname(f)
        run_id = os.path.basename(f)[len("test_predictions_"):-4]
        seed = int(run_id.split("seed")[-1])
        seeds.append(seed)
        preds.append(read_pred(f))
        mj = os.path.join(d, f"metrics_{run_id}.json")
        t = None
        if os.path.exists(mj):
            m = json.load(open(mj))
            t = m.get("threshold_from_val", m.get("threshold"))
        thr.append(t)
        vp = os.path.join(d, f"val_predictions_{run_id}.csv")
        vals.append(read_pred(vp) if os.path.exists(vp) else None)
    return seeds, preds, thr, vals


# ------------------------------------------------------------------ metrics
def point_metrics(y, p, thr):
    out = {"auc": float(roc_auc_score(y, p))}
    if thr is None:
        out.update({k: np.nan for k in METRICS[1:]})
        return out
    pr = p >= thr
    tp, fp = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum())
    fn, tn = int((~pr & (y == 1)).sum()), int((~pr & (y == 0)).sum())
    out.update(accuracy=(tp + tn) / len(y), precision=tp / (tp + fp) if tp + fp else 0.0,
               recall=tp / (tp + fn), specificity=tn / (tn + fp), f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return out


def stratified_indices(y, B, seed):
    """B resamples; positives and negatives are drawn separately so class counts never change."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return [(rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)) for _ in range(B)]


def bootstrap_metrics(y, p, thr, resamples):
    rows = []
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    pred = None if thr is None else (p >= thr)
    for pi, ni in resamples:
        idx = np.concatenate([pi, ni])
        r = {"auc": float(roc_auc_score(y[idx], p[idx]))}
        if pred is None:
            r.update({k: np.nan for k in METRICS[1:]})
        else:
            tp, fp = int(pred[pi].sum()), int(pred[ni].sum())
            fn, tn = n_pos - tp, n_neg - fp
            r.update(accuracy=(tp + tn) / (n_pos + n_neg), precision=tp / (tp + fp) if tp + fp else 0.0,
                     recall=tp / n_pos, specificity=tn / n_neg, f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ DeLong (fast algorithm, Sun & Xu 2014)
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, pa, pb):
    order = np.argsort(-y, kind="stable")  # positives first
    m = int(y.sum())
    preds = np.vstack([pa, pb])[:, order]
    n = preds.shape[1] - m
    k = 2
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r], ty[r], tz[r] = _midrank(preds[r, :m]), _midrank(preds[r, m:]), _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01, v10 = (tz[:, :m] - tx) / n, 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    diff = float(aucs[0] - aucs[1])
    z = diff / np.sqrt(var) if var > 0 else 0.0
    half = Z95 * np.sqrt(var) if var > 0 else 0.0
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), diff=diff, ci_lower=diff - half, ci_upper=diff + half,
                z=float(z), p_value=float(2 * ss.norm.sf(abs(z))))


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="append", required=True, metavar="NAME=FOLDER")
    ap.add_argument("--out", default=os.path.join("results", "densenet121_stylegan2-ada", "evaluation"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--boot_seed", type=int, default=42)
    ap.add_argument("--chehade_auc", type=float, default=0.8387)
    ap.add_argument("--chehade_sd", type=float, default=0.014)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    S = {}
    for item in args.setup:
        name, folder = item.split("=", 1)
        S[name] = load_setup(folder)

    # ---- the same test cases and labels in every setup (DeLong and the shared bootstrap need this)
    ref_name = next(iter(S))
    ref = S[ref_name][1][0]
    for name, (seeds, preds, _, _) in S.items():
        for d in preds:
            assert d["Image Index"].equals(ref["Image Index"]) and d["label"].equals(ref["label"]), \
                f"{name}: test cases or labels differ from {ref_name}; setups must be scored on the same test set"
    y = ref["label"].to_numpy().astype(int)
    print(f"Test set: {len(y)} images, {int(y.sum())} pneumonia / {int((1 - y).sum())} No Finding")

    # ---- 1. per-seed metrics
    seed_rows = []
    for name, (seeds, preds, thr, _) in S.items():
        for sd, d, t in zip(seeds, preds, thr):
            seed_rows.append(dict(setup=name, seed=sd, threshold=t, **point_metrics(y, d["prob_pneumonia"].to_numpy(), t)))
    per_seed = pd.DataFrame(seed_rows).sort_values(["setup", "seed"])
    per_seed.to_csv(os.path.join(args.out, "per_seed_test_metrics.csv"), index=False)
    print("\n=== PER-SEED TEST METRICS ===")
    print(per_seed.round(4).to_string(index=False))

    # ---- 2. ensemble + stratified bootstrap
    resamples = stratified_indices(y, args.boot, args.boot_seed)  # same resamples for every setup
    ens_p, boots, summary = {}, {}, []
    for name, (seeds, preds, thr, vals) in S.items():
        p = np.mean([d["prob_pneumonia"].to_numpy() for d in preds], axis=0)
        ens_p[name] = p
        if all(v is not None for v in vals):  # threshold of the ensemble: Youden on the mean VALIDATION probability
            vy = vals[0]["label"].to_numpy()
            vp = np.mean([v["prob_pneumonia"].to_numpy() for v in vals], axis=0)
            fpr, tpr, th = roc_curve(vy, vp)
            t = float(th[np.argmax(tpr - fpr)])
        elif all(x is not None for x in thr):
            t = float(np.mean(thr))
            print(f"NOTE {name}: no validation predictions found; ensemble threshold = mean of the seeds' thresholds")
        else:
            t = None
            print(f"NOTE {name}: no thresholds found; only AUC is reported for this setup")
        est = point_metrics(y, p, t)
        b = bootstrap_metrics(y, p, t, resamples)
        b.to_csv(os.path.join(args.out, f"bootstrap_samples_{name}.csv"), index=False)
        boots[name] = b
        for k in METRICS:
            sd = float(b[k].std(ddof=1))
            summary.append(dict(setup=name, metric=k, n_seeds=len(seeds), threshold=t, estimate=est[k], sd=sd,
                                ci_lower=max(0.0, est[k] - Z95 * sd), ci_upper=min(1.0, est[k] + Z95 * sd)))  # metrics live in [0, 1]
    ens = pd.DataFrame(summary)
    ens.to_csv(os.path.join(args.out, "ensemble_metrics.csv"), index=False)
    print(f"\n=== ENSEMBLE (mean of seeds), stratified bootstrap x{args.boot}, CI = estimate +/- 1.96 SD ===")
    print(ens.round(4).to_string(index=False))
    print(f"\nReference (descriptive only, no per-image predictions published): Chehade et al. (2025) "
          f"AUC {args.chehade_auc:.4f} +/- {args.chehade_sd:.3f} (mean +/- SD over ten random splits, balanced ~135-image test sets)")

    # ---- 3. DeLong between setups (ensemble predictions), Holm-corrected
    names = list(S)
    if len(names) >= 2:
        rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = delong(y, ens_p[names[i]], ens_p[names[j]])
                rows.append(dict(setup_a=names[i], setup_b=names[j], **r))
        dl = pd.DataFrame(rows)
        dl["p_holm"] = holm(dl["p_value"].to_numpy())
        dl["significant_holm_0.05"] = dl["p_holm"] < 0.05
        dl.to_csv(os.path.join(args.out, "delong.csv"), index=False)
        print("\n=== DELONG (ensemble AUC, pairwise, Holm correction) ===")
        print(dl.round(4).to_string(index=False))

        # ---- 4a. ANOVA across seeds
        rows, tuk = [], []
        for k in METRICS:
            groups = [per_seed.loc[per_seed.setup == n, k].dropna().to_numpy() for n in names]
            if any(len(g) < 2 for g in groups):
                continue
            F, p = ss.f_oneway(*groups)
            rows.append(dict(metric=k, F=F, p_value=p, **{f"mean_{n}": g.mean() for n, g in zip(names, groups)}))
            if len(names) >= 3 and p < 0.05:
                res = ss.tukey_hsd(*groups)
                for a in range(len(names)):
                    for b in range(a + 1, len(names)):
                        tuk.append(dict(metric=k, a=names[a], b=names[b], p_tukey=float(res.pvalue[a, b])))
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "anova_seeds.csv"), index=False)
        print("\n=== ANOVA ACROSS SEEDS (training randomness; n = seeds per setup) ===")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        if tuk:
            pd.DataFrame(tuk).to_csv(os.path.join(args.out, "anova_seeds_tukey.csv"), index=False)
            print(pd.DataFrame(tuk).round(4).to_string(index=False))

        # ---- 4b. ANOVA across bootstrap resamples (+ eta-squared)
        rows = []
        for k in METRICS:
            groups = [boots[n][k].dropna().to_numpy() for n in names]
            if any(len(g) < 2 for g in groups):
                continue
            F, p = ss.f_oneway(*groups)
            allv = np.concatenate(groups)
            ss_between = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
            rows.append(dict(metric=k, F=F, p_value=p, eta_squared=ss_between / ((allv - allv.mean()) ** 2).sum(),
                             **{f"mean_{n}": g.mean() for n, g in zip(names, groups)}))
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "anova_bootstrap.csv"), index=False)
        print("\n=== ANOVA ACROSS BOOTSTRAP RESAMPLES (test-set sampling noise; read eta-squared, resamples overlap) ===")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nFiles written to {args.out}")


if __name__ == "__main__":
    main()
