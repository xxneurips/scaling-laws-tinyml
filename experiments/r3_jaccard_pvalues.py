"""
R3 NeurIPS — Per-bucket bootstrap p-values for the crossover Δ = within - cross.

We note that the crossover magnitude is presented without
per-bucket significance tests. This script computes the 95% bootstrap CI and
two-sided p-value (against Δ = 0) for each size-ratio bucket on each dataset.

Usage:
    python r3_jaccard_pvalues.py <dataset>
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np

BASE = Path(__file__).parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cifar100"
RESULTS = BASE / "results" / DATASET

N_BOOT = 10000
rng = np.random.default_rng(2026)


def jaccard(mask_a, mask_b):
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


def build_pairs(runs):
    pairs = []
    for a, b in combinations(runs, 2):
        ni, nj = a["num_params"], b["num_params"]
        if ni == nj:
            continue
        if ni > nj:
            a, b = b, a
            ni, nj = nj, ni
        ratio = nj / ni
        same_arch = a["arch"] == b["arch"]
        j = jaccard(a["correct_mask"], b["correct_mask"])
        pairs.append({"ratio": ratio, "same_arch": same_arch, "jaccard": j})
    return pairs


def main():
    runs = load_runs()
    pairs = build_pairs(runs)
    print(f"[{DATASET}] {len(runs)} runs, {len(pairs)} pairs")

    buckets = [
        (1.0, 1.5, "1.0-1.5x"),
        (1.5, 2.5, "1.5-2.5x"),
        (2.5, 5.0, "2.5-5x"),
        (5.0, 10.0, "5-10x"),
        (10.0, 50.0, "10-50x"),
        (50.0, 1e9, ">50x"),
    ]

    print(f"\n{'Bucket':>10} {'within (n)':>14} {'cross (n)':>14} {'Delta':>8} {'95% CI':>22} {'p':>12}")
    print("-" * 90)

    results = []
    for lo, hi, label in buckets:
        within = np.array([p["jaccard"] for p in pairs
                           if p["same_arch"] and lo <= p["ratio"] < hi])
        cross = np.array([p["jaccard"] for p in pairs
                          if not p["same_arch"] and lo <= p["ratio"] < hi])
        if len(within) < 2 or len(cross) < 2:
            print(f"{label:>10} (insufficient data)")
            continue

        delta_obs = within.mean() - cross.mean()

        # Bootstrap: resample within each set, recompute delta
        deltas = np.empty(N_BOOT)
        for b in range(N_BOOT):
            w = rng.choice(within, size=len(within), replace=True)
            c = rng.choice(cross, size=len(cross), replace=True)
            deltas[b] = w.mean() - c.mean()
        ci = np.percentile(deltas, [2.5, 97.5])
        # Two-sided p-value against delta = 0 (permutation-style)
        # Approximate via bootstrap centered on zero
        pooled = np.concatenate([within, cross])
        null_deltas = np.empty(N_BOOT)
        n_w = len(within)
        for b in range(N_BOOT):
            perm = rng.permutation(pooled)
            null_deltas[b] = perm[:n_w].mean() - perm[n_w:].mean()
        p_two = (np.abs(null_deltas) >= abs(delta_obs)).mean()

        print(f"{label:>10} {within.mean():>7.3f} ({len(within):>3}) "
              f"{cross.mean():>7.3f} ({len(cross):>3}) "
              f"{delta_obs:>+8.3f}  [{ci[0]:>+.3f}, {ci[1]:>+.3f}]  {p_two:>11.5f}")

        results.append({
            "bucket": label, "lo": lo, "hi": hi,
            "within_mean": float(within.mean()), "within_n": len(within),
            "cross_mean": float(cross.mean()), "cross_n": len(cross),
            "delta": float(delta_obs),
            "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
            "p_two_sided": float(p_two),
        })

    out_path = RESULTS / f"jaccard_pvalues_{DATASET}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
