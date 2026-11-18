"""
Stage 3, step 4: manual check that decides which cluster is the artifact cluster.

Input:  output/clusters/clustered_676.csv
Output: 3_downsampling_branch/output/label_grids/grid_*.png, manual_labels.csv, label_report.json, masks/
        adds the final `domain` column (artifact | clean) to clustered_676.csv (--score)
Run:    python 3_downsampling_branch/label_grid.py            make the grids, then mark manual_labels.csv
        python 3_downsampling_branch/label_grid.py --more     another round of images
        python 3_downsampling_branch/label_grid.py --score    read the marks and fix the artifact cluster
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import scipy.stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from features import lung_mask, load_1024
from setup4_config import CLUSTERS_DIR, LABELGRID_DIR, LABEL_GRID_PER_CLUSTER, SUBSET_SEED, make_dirs

CLUSTERED_CSV = os.path.join(CLUSTERS_DIR, "clustered_676.csv")
MANUAL_CSV = os.path.join(LABELGRID_DIR, "manual_labels.csv")
REPORT_JSON = os.path.join(LABELGRID_DIR, "label_report.json")
MASK_DIR = os.path.join(LABELGRID_DIR, "masks")
CLASSES = ["Pneumonia", "No Finding"]
THUMB = 320          # thumbnail size in the contact sheet
COLS = 5
CHEHADE_AGREEMENT = {"Pneumonia": 83.54, "No Finding": 82.49}


def thumbnail(path, size=THUMB):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((size, size), dtype=np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def contact_sheet(paths, numbers, cols=COLS, size=THUMB):
    """Numbered grid. The number is drawn twice (black then white) so it is readable on any background."""
    rows = int(np.ceil(len(paths) / cols))
    sheet = np.full((rows * size, cols * size), 255, dtype=np.uint8)
    for i, (p, num) in enumerate(zip(paths, numbers)):
        r, c = divmod(i, cols)
        t = thumbnail(p, size)
        cv2.rectangle(t, (0, 0), (size - 1, size - 1), 255, 2)
        cv2.putText(t, str(num), (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 7, cv2.LINE_AA)
        cv2.putText(t, str(num), (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 2, cv2.LINE_AA)
        sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = t
    return sheet


def mask_overlay(path):
    """Red where the lung mask is, so the Otsu assumption can be judged by eye."""
    img = load_1024(path)
    mask, fallback = lung_mask(img)
    u8 = (img * 255).astype(np.uint8)
    rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(rgb)
    red[..., 2] = 255
    rgb[mask] = (0.65 * rgb[mask] + 0.35 * red[mask]).astype(np.uint8)
    return cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA), fallback


def make(args):
    if not os.path.exists(CLUSTERED_CSV):
        raise SystemExit(f"{CLUSTERED_CSV} not found: run cluster_kmeans.py first")
    df = pd.read_csv(CLUSTERED_CSV)
    os.makedirs(MASK_DIR, exist_ok=True)

    # a plain run would overwrite manual_labels.csv and erase the marks
    if not args.more and os.path.exists(MANUAL_CSV) and not args.force:
        old = pd.read_csv(MANUAL_CSV)
        n_marked = int(old["manual"].astype(str).str.strip().str.lower().isin(["a", "c"]).sum())
        if n_marked:
            raise SystemExit(f"{MANUAL_CSV} already holds {n_marked} marks and this mode would overwrite it.\n"
                             f"  more images to mark -> add --more\n"
                             f"  mask pictures only  -> add --masks\n"
                             f"  really start over   -> add --force (back the file up first)")

    # round 2 (--more): keep the marks already made and draw a FRESH sample of unseen images
    already, start_num, prev = set(), 1, None
    rnd = 1
    if args.more:
        if not os.path.exists(MANUAL_CSV):
            raise SystemExit(f"{MANUAL_CSV} not found: run without --more first")
        prev = pd.read_csv(MANUAL_CSV)
        already = set(prev["image"])
        start_num = int(prev["number"].max()) + 1
        rnd = int(prev["round"].max()) + 1 if "round" in prev.columns else 2
        if "round" not in prev.columns:
            prev["round"] = 1
        marked = int(prev["manual"].astype(str).str.strip().str.lower().isin(["a", "c"]).sum())
        print(f"round {rnd}: {len(already)} images already shown ({marked} marked); drawing fresh ones\n")
    # a different seed per round, so round 2 is an independent draw, not a reshuffle of round 1
    rng = np.random.default_rng(args.seed + 1000 * rnd)

    rows, num = [], start_num
    for cls in CLASSES:
        for k in sorted(df.loc[df["Finding Labels"] == cls, "cluster"].unique()):
            sub = df[(df["Finding Labels"] == cls) & (df["cluster"] == k)
                     & (~df["Image Index"].isin(already))]
            take = min(args.per_cluster, len(sub))
            if take == 0:
                print(f"  {cls} cluster {k}: every image has already been shown")
                continue
            if take < args.per_cluster:
                print(f"  {cls} cluster {k}: only {take} unseen images left")
            pick = sub.iloc[rng.choice(len(sub), size=take, replace=False)].reset_index(drop=True)
            numbers = list(range(num, num + take))
            num += take

            suffix = "" if rnd == 1 else f"_round{rnd}"
            name = f"grid_{cls.replace(' ', '_').lower()}_cluster{k}{suffix}.png"
            cv2.imwrite(os.path.join(LABELGRID_DIR, name),
                        contact_sheet(pick["src_path"].tolist(), numbers))
            print(f"  {name}  {take} images (numbers {numbers[0]}-{numbers[-1]}) of "
                  f"{len(sub) + len(already & set(sub['Image Index']))} in this cluster")

            for n, (_, r) in zip(numbers, pick.iterrows()):
                rows.append(dict(number=n, image=r["Image Index"], cls=cls, cluster=int(k),
                                 provisional=r["provisional_domain"], manual="", note="", round=rnd))

    labels = pd.DataFrame(rows)
    # CSV rows are shuffled so the cluster is hidden while marking
    labels = labels.sample(frac=1.0, random_state=args.seed).sort_values("number").reset_index(drop=True)
    if prev is not None:
        labels = pd.concat([prev, labels], ignore_index=True)
    labels.to_csv(MANUAL_CSV, index=False)

    # lung-mask overlays
    sample = df.sample(n=min(args.n_masks, len(df)), random_state=args.seed)
    n_fallback = 0
    for _, r in sample.iterrows():
        ov, fb = mask_overlay(r["src_path"])
        n_fallback += int(fb)
        cv2.imwrite(os.path.join(MASK_DIR, f"mask_{r['Image Index']}"), ov)

    new = len(rows)
    print(f"\nwrote 4 contact sheets ({new} new images) and {len(sample)} mask overlays"
          + (f"  ({n_fallback} fell back to the whole image)" if n_fallback else ""))
    print(f"\nNEXT:")
    print(f"  1. open the new grid_*.png files in {LABELGRID_DIR}"
          + (f" (the ones ending _round{rnd})" if rnd > 1 else ""))
    print(f"  2. open {MANUAL_CSV} in Excel; the marks you already made are kept")
    print(f"  3. fill in the 'manual' column for numbers {start_num}-{start_num + new - 1}: "
          f"a (artifact) or c (clean);")
    print(f"     leave it blank if you cannot tell. Do not sort or delete rows.")
    print(f"  4. glance at the overlays in {MASK_DIR}: red should cover the lung fields, not the whole chest")
    print(f"  5. save as CSV, then run:  python 3_downsampling_branch/label_grid.py --score")
    print(f"\nArtifact = pacemaker, implant, ECG leads, wires, catheter, clips, burned-in text or position")
    print(f"markers, or a clearly bad capture (heavy blur, extreme exposure, large cut-off).")


def masks_only(args):
    """Rewrite the lung-mask overlays and print mask statistics. Never touches manual_labels.csv."""
    if not os.path.exists(CLUSTERED_CSV):
        raise SystemExit(f"{CLUSTERED_CSV} not found: run cluster_kmeans.py first")
    df = pd.read_csv(CLUSTERED_CSV)
    os.makedirs(MASK_DIR, exist_ok=True)
    sample = df.sample(n=min(args.n_masks, len(df)), random_state=args.seed)
    fracs, n_fb = [], 0
    for _, r in sample.iterrows():
        img = load_1024(r["src_path"])
        mask, fb = lung_mask(img)
        fracs.append(float(mask.mean()))
        n_fb += int(fb)
        ov, _ = mask_overlay(r["src_path"])
        cv2.imwrite(os.path.join(MASK_DIR, f"mask_{r['Image Index']}"), ov)
    fracs = np.array(fracs)
    print(f"wrote {len(sample)} overlays to {MASK_DIR}")
    print(f"mask area as share of the image: median {np.median(fracs):.2f}, "
          f"min {fracs.min():.2f}, max {fracs.max():.2f}   (two lung fields: about 0.15-0.35)")
    print(f"fell back to the whole image: {n_fb} of {len(sample)}")
    print("open a few overlays: red should sit on the two lung fields and NOT on the black frame.")


def score(args):
    if not os.path.exists(MANUAL_CSV):
        raise SystemExit(f"{MANUAL_CSV} not found: run without --score first")
    lab = pd.read_csv(MANUAL_CSV)
    lab["manual"] = lab["manual"].astype(str).str.strip().str.lower()
    bad = set(lab.loc[lab["manual"] != "", "manual"]) - {"a", "c", "nan"}
    if bad:
        raise SystemExit(f"unexpected values in the 'manual' column: {bad} (use a or c)")

    df = pd.read_csv(CLUSTERED_CSV)

    # take the current cluster from clustered_676.csv (the one in manual_labels.csv can be stale)
    cur = df.set_index("Image Index")["cluster"]
    missing = sorted(set(lab["image"]) - set(cur.index))
    if missing:
        raise SystemExit(f"{len(missing)} marked images are not in {CLUSTERED_CSV}, e.g. {missing[:3]}")
    lab["cluster_at_labeling"] = lab["cluster"]
    lab["cluster"] = lab["image"].map(cur).astype(int)
    changed = int((lab["cluster"] != lab["cluster_at_labeling"]).sum())
    print(f"cluster re-merged from {os.path.basename(CLUSTERED_CSV)} by Image Index: "
          f"{changed} of {len(lab)} marked images carry a different cluster number than at labelling time"
          + ("  (numbering and/or membership changed)" if changed else "") + "\n")

    done = lab[lab["manual"].isin(["a", "c"])]
    print(f"{len(done)} of {len(lab)} images marked ({len(lab) - len(done)} left blank)\n")
    if len(done) < 20:
        raise SystemExit("too few marks to decide the direction; mark more images")

    report, final = {}, {}
    for cls in CLASSES:
        d = done[done["cls"] == cls]
        if d.empty:
            print(f"{cls}: nothing marked, cannot decide")
            continue
        ks = sorted(d["cluster"].unique())
        share = {int(k): float((d.loc[d["cluster"] == k, "manual"] == "a").mean()) for k in ks}
        artifact = int(max(share, key=share.get))

        # agreement = how often the clustering, named this way, matches your mark
        pred = np.where(d["cluster"].to_numpy() == artifact, "a", "c")
        agreement = float((pred == d["manual"].to_numpy()).mean())
        prov = int(df.loc[df["Finding Labels"] == cls, "cluster"].iloc[0])
        prov_artifact = json.load(open(os.path.join(CLUSTERS_DIR, "cluster_report.json"))
                                  )["per_class"][cls]["provisional_artifact_cluster"]

        # is the difference between the two clusters more than sampling noise? (2x2 Fisher exact)
        table = [[int((d.loc[d["cluster"] == k, "manual"] == "a").sum()),
                  int((d.loc[d["cluster"] == k, "manual"] == "c").sum())] for k in ks]
        p_fisher = float(ss.fisher_exact(table)[1]) if len(ks) == 2 else float("nan")
        # Wilson 95% CI for the agreement, so n is visible in the number
        n_ag = len(d)
        z, phat = 1.96, agreement
        denom = 1 + z * z / n_ag
        centre = (phat + z * z / (2 * n_ag)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n_ag + z * z / (4 * n_ag * n_ag)) / denom

        print(f"{cls}  ({len(d)} marked)")
        for k, t in zip(ks, table):
            print(f"    cluster {k}: {100 * share[k]:>5.1f}% marked artifact  "
                  f"({t[0]} of {t[0] + t[1]})" + ("   <- ARTIFACT" if k == artifact else ""))
        print(f"    agreement with your labels: {100 * agreement:.1f}%  "
              f"[95% CI {100 * (centre - half):.0f}-{100 * (centre + half):.0f}]   "
              f"(Chehade {CHEHADE_AGREEMENT[cls]}%, chance 50%)")
        print(f"    difference between clusters: Fisher exact p = {p_fisher:.3f}"
              + ("  (real)" if p_fisher < 0.05 else "  (not distinguishable from noise)"))
        if artifact != prov_artifact:
            print(f"    NOTE: this OVERTURNS the provisional guess (cluster {prov_artifact}); your labels win")
        if p_fisher >= 0.05:
            print(f"    WARNING: these two clusters are not separating artifacts in your sample. The direction")
            print(f"             CycleGAN would clean is close to arbitrary. Report this rather than hide it.")
        final[cls] = artifact
        report[cls] = dict(n_marked=len(d), artifact_cluster=artifact, agreement=agreement,
                           agreement_ci95=[float(centre - half), float(centre + half)],
                           fisher_p=p_fisher, contingency=table,
                           artifact_share_per_cluster=share, provisional_was=prov_artifact,
                           overturned=bool(artifact != prov_artifact),
                           chehade_agreement=CHEHADE_AGREEMENT[cls])
        print()

    if len(final) < len(CLASSES):
        raise SystemExit("both classes must be decided before CycleGAN can be trained")

    df["domain"] = ["artifact" if r["cluster"] == final[r["Finding Labels"]] else "clean"
                    for _, r in df.iterrows()]
    df.to_csv(CLUSTERED_CSV, index=False)
    json.dump(dict(per_class=report, final_artifact_cluster=final,
                   n_marked=int(len(done)), n_shown=int(len(lab))),
              open(REPORT_JSON, "w"), indent=2)

    ct = pd.crosstab(df["Finding Labels"], df["domain"])
    print("FINAL DOMAINS (Chehade Table 1: 152/133 artifact, 170/221 clean)")
    print(ct.to_string())
    print(f"\ntotal artifact {int((df['domain'] == 'artifact').sum())} of {len(df)} "
          f"(Chehade 285 of 676)")
    print(f"\nwrote the 'domain' column to {CLUSTERED_CSV}\n      {REPORT_JSON}")
    print("\nCycleGAN will translate artifact -> clean. Next: the CycleGAN scripts (GPU, after Setup 3).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="read the filled-in CSV and fix the direction")
    ap.add_argument("--more", action="store_true",
                    help="draw another round of UNSEEN images, keeping the marks already made")
    ap.add_argument("--per_cluster", type=int, default=LABEL_GRID_PER_CLUSTER)
    ap.add_argument("--masks", action="store_true",
                    help="only rewrite the lung-mask overlays and print mask stats; marks are not touched")
    ap.add_argument("--force", action="store_true",
                    help="allow a plain run to overwrite a manual_labels.csv that already has marks")
    ap.add_argument("--n_masks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SUBSET_SEED)
    args = ap.parse_args()
    make_dirs()
    os.makedirs(LABELGRID_DIR, exist_ok=True)
    if args.score:
        score(args)
    elif args.masks:
        masks_only(args)
    else:
        make(args)


if __name__ == "__main__":
    main()
