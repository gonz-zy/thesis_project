"""
compare_clusters.py -- how much did the clustering move after the lung-mask fix? Read-only.

    python scripts\\util\\compare_clusters.py <old clustered_676.csv> <new clustered_676.csv>

K-Means cluster numbers are arbitrary, so within each class the new numbering is matched to the old one
(either as-is or swapped, whichever agrees more) before counting images that changed cluster.
"""
import sys

import pandas as pd


def main(old_path, new_path):
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    m = old[["Image Index", "Finding Labels", "cluster"]].merge(
        new[["Image Index", "cluster"]], on="Image Index", suffixes=("_old", "_new"))
    if len(m) != len(old) or len(m) != len(new):
        raise SystemExit(f"image sets differ: old {len(old)}, new {len(new)}, matched {len(m)}")
    print(f"{len(m)} images compared\n")
    total_changed = 0
    for cls, g in m.groupby("Finding Labels"):
        same = float((g["cluster_old"] == g["cluster_new"]).mean())
        swapped = float((g["cluster_old"] == 1 - g["cluster_new"]).mean())
        best, flip = (same, False) if same >= swapped else (swapped, True)
        changed = int(round((1 - best) * len(g)))
        total_changed += changed
        so = g["cluster_old"].value_counts().sort_index().tolist()
        sn = g["cluster_new"].value_counts().sort_index().tolist()
        print(f"{cls:<11} n={len(g)}  old sizes {so}  new sizes {sn}  "
              f"numbering {'SWAPPED' if flip else 'same'}  "
              f"unchanged {100 * best:.1f}%  ({changed} images moved)")
    print(f"\ntotal images that changed cluster: {total_changed} of {len(m)}")
    print("If this is small, the earlier labelling result carries over almost unchanged; "
          "rerun label_grid.py --score to confirm.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
