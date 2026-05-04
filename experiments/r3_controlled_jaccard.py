"""
R3 (NeurIPS prep) — Controlled Jaccard reanalysis.

Methodological note:
  The prior reference compares within-arch Jaccard at 200x size gap (smallest to largest)
  against cross-arch Jaccard at ~1x size gap (matched sizes). That is an uncontrolled
  comparison — of course the 200x gap shows more redistribution.

  Here we bucket all pairs by log(size ratio) and compare within-arch vs cross-arch
  at matched buckets. Produces the corrected Table 5.

Output:
  - results/cifar100/controlled_jaccard.json  (all pair data)
  - figures/controlled_jaccard_table.txt      (NeurIPS Table 5)
  - figures/fig_controlled_jaccard.pdf        (matched-bucket comparison plot)
"""
import json
import os
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import sys
BASE = Path(__file__).parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cifar100"
RESULTS = BASE / "results" / DATASET
FIG_DIR = BASE.parent / "paper" / "neurips2026" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n[DATASET={DATASET}]\n")


def jaccard(mask_a, mask_b):
    """Jaccard overlap of error sets (mask = True where correct)."""
    err_a = ~np.asarray(mask_a, dtype=bool)
    err_b = ~np.asarray(mask_b, dtype=bool)
    inter = int(np.logical_and(err_a, err_b).sum())
    union = int(np.logical_or(err_a, err_b).sum())
    return inter / union if union > 0 else float("nan")


def load_runs():
    runs = []
    for fn in sorted(os.listdir(RESULTS)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(RESULTS / fn) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        if "correct_mask" not in d or d.get("arch") is None:
            continue
        runs.append(d)
    return runs


def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} runs with correct_mask")

    # All pairs with N_i < N_j
    pairs = []
    for r_i, r_j in combinations(runs, 2):
        n_i, n_j = r_i["num_params"], r_j["num_params"]
        if n_i == n_j:
            continue
        if n_i > n_j:
            r_i, r_j = r_j, r_i
            n_i, n_j = n_j, n_i
        ratio = n_j / n_i
        same_arch = r_i["arch"] == r_j["arch"]
        same_seed = r_i["seed"] == r_j["seed"]
        j = jaccard(r_i["correct_mask"], r_j["correct_mask"])
        pairs.append({
            "arch_i": r_i["arch"], "arch_j": r_j["arch"],
            "sp_i": r_i["size_param"], "sp_j": r_j["size_param"],
            "n_i": n_i, "n_j": n_j,
            "ratio": ratio, "log_ratio": np.log2(ratio),
            "same_arch": same_arch, "same_seed": same_seed,
            "seed_i": r_i["seed"], "seed_j": r_j["seed"],
            "jaccard": j,
        })
    print(f"Total pairs: {len(pairs)}")

    # Bucket by size ratio
    buckets = [
        (1.0, 1.5, "1.0-1.5x"),
        (1.5, 2.5, "1.5-2.5x"),
        (2.5, 5.0, "2.5-5x"),
        (5.0, 10.0, "5-10x"),
        (10.0, 50.0, "10-50x"),
        (50.0, 1e9, ">50x"),
    ]

    print()
    print("=" * 80)
    print("CONTROLLED JACCARD: within-arch vs cross-arch at MATCHED size-ratio buckets")
    print("=" * 80)
    print(f"{'Bucket':>10}  {'Within-arch':>20}  {'Cross-arch':>20}  {'Delta':>8}  {'n_within':>8}  {'n_cross':>8}")
    print("-" * 80)

    table_rows = []
    for lo, hi, label in buckets:
        within = [p["jaccard"] for p in pairs
                  if p["same_arch"] and lo <= p["ratio"] < hi]
        cross = [p["jaccard"] for p in pairs
                 if not p["same_arch"] and lo <= p["ratio"] < hi]
        if len(within) == 0 and len(cross) == 0:
            continue
        w_mean = np.mean(within) if within else float("nan")
        w_std = np.std(within) if within else float("nan")
        c_mean = np.mean(cross) if cross else float("nan")
        c_std = np.std(cross) if cross else float("nan")
        delta = w_mean - c_mean if within and cross else float("nan")
        print(f"{label:>10}  {w_mean:>7.3f} +/- {w_std:>5.3f} ({len(within):>3})  "
              f"{c_mean:>7.3f} +/- {c_std:>5.3f} ({len(cross):>3})  "
              f"{delta:>+7.3f}")
        table_rows.append({
            "bucket": label, "lo": lo, "hi": hi,
            "within_mean": w_mean, "within_std": w_std, "within_n": len(within),
            "cross_mean": c_mean, "cross_std": c_std, "cross_n": len(cross),
            "delta": delta,
        })

    print()
    print("KEY QUESTION: in each matched bucket, is within-arch > cross-arch?")
    for row in table_rows:
        if np.isnan(row["delta"]):
            continue
        sign = "WITHIN > CROSS" if row["delta"] > 0 else "CROSS > WITHIN"
        print(f"  {row['bucket']:>10}: delta = {row['delta']:+.3f}   {sign}")

    # Also produce the original uncontrolled comparison (for the reframe section of the paper)
    print()
    print("=" * 80)
    print("UNCONTROLLED (original prior framing, for reference)")
    print("=" * 80)
    cross_all = [p["jaccard"] for p in pairs if not p["same_arch"] and abs(p["log_ratio"]) < 1]
    print(f"  Cross-arch, matched size (ratio < 2x): {np.mean(cross_all):.3f} +/- {np.std(cross_all):.3f} (n={len(cross_all)})")
    for arch in ["scalecnn", "mobilenet"]:
        within_adj = [p["jaccard"] for p in pairs
                      if p["same_arch"] and p["arch_i"] == arch and 1 <= p["ratio"] < 2.5]
        within_far = [p["jaccard"] for p in pairs
                      if p["same_arch"] and p["arch_i"] == arch and p["ratio"] > 50]
        print(f"  Within {arch}, adjacent (1-2.5x): {np.mean(within_adj):.3f} (n={len(within_adj)})")
        if within_far:
            print(f"  Within {arch}, distant (>50x): {np.mean(within_far):.3f} (n={len(within_far)})")

    # Save machine-readable
    out = {
        "buckets": table_rows,
        "n_pairs": len(pairs),
        "n_runs": len(runs),
    }
    out_path = RESULTS / f"controlled_jaccard_{DATASET}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")

    # Plot
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(table_rows))
        width = 0.4
        w_means = [r["within_mean"] for r in table_rows]
        w_stds = [r["within_std"] for r in table_rows]
        c_means = [r["cross_mean"] for r in table_rows]
        c_stds = [r["cross_std"] for r in table_rows]
        labels = [r["bucket"] for r in table_rows]

        ax.bar(x - width/2, w_means, width, yerr=w_stds, label="Within-architecture",
               color="#1a5276", capsize=3, alpha=0.85)
        ax.bar(x + width/2, c_means, width, yerr=c_stds, label="Cross-architecture",
               color="#e87422", capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Size ratio bucket  $N_j / N_i$")
        ax.set_ylabel("Jaccard error-set overlap")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        out_fig = FIG_DIR / f"fig_controlled_jaccard_{DATASET}.pdf"
        ax.set_title(f"Controlled Jaccard ({DATASET.upper()}): within- vs cross-architecture at matched size ratios")
        plt.savefig(out_fig, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_fig}")

    print()
    print("=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("If WITHIN > CROSS in every matched bucket:")
    print("  -> architecture DOES matter more than scale (at matched size ratios).")
    print("  -> the prior headline claim is WRONG as stated.")
    print("  -> Reframe: 'Scale rotates error sets monotonically; architecture adds an")
    print("     additional orthogonal rotation on top.'")
    print()
    print("If WITHIN ~ CROSS in matched buckets:")
    print("  -> prior headline is SALVAGEABLE with controlled comparison.")


if __name__ == "__main__":
    main()
