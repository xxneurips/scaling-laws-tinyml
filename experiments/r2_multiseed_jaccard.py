"""
R2 Comment 1: Multi-seed Jaccard overlap for cross-architecture table.
Computes mean ± std of Jaccard across all 5 seeds instead of seed-0 only.
"""
import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "cifar100"


def compute_jaccard(correct_a, correct_b):
    err_a = ~np.array(correct_a, dtype=bool)
    err_b = ~np.array(correct_b, dtype=bool)
    intersection = (err_a & err_b).sum()
    union = (err_a | err_b).sum()
    return float(intersection / union) if union > 0 else 1.0


def load_runs():
    runs = []
    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        if json_file.stem.endswith("_summary") or json_file.stem.endswith("_results"):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "arch" in data and "correct_mask" in data:
                runs.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return runs


def main():
    runs = load_runs()

    # Group by (arch, size_param, seed)
    by_key = {}
    for r in runs:
        key = (r["arch"], r["size_param"])
        seed = r.get("seed", 0)
        by_key.setdefault(key, {})[seed] = r

    # Cross-architecture pairs (from Table 5 in paper)
    # ScaleCNN configs matched to MobileNetV2 configs
    cross_pairs = [
        (("scalecnn", 12), ("mobilenet", 0.1)),
        (("scalecnn", 16), ("mobilenet", 0.15)),
        (("scalecnn", 16), ("mobilenet", 0.25)),
        (("scalecnn", 24), ("mobilenet", 0.35)),
        (("scalecnn", 24), ("mobilenet", 0.50)),
        (("scalecnn", 32), ("mobilenet", 0.75)),
        (("scalecnn", 48), ("mobilenet", 1.00)),
        (("scalecnn", 48), ("mobilenet", 1.50)),
        (("scalecnn", 64), ("mobilenet", 1.00)),
        (("scalecnn", 64), ("mobilenet", 1.50)),
        (("scalecnn", 64), ("mobilenet", 2.00)),
    ]

    print("=" * 90)
    print("Cross-Architecture Jaccard: Multi-seed (mean ± std across 5 seeds)")
    print("=" * 90)
    print(f"{'ScaleCNN':>10} {'MobNetV2':>10} {'CNN Params':>12} {'Mob Params':>12} "
          f"{'Seed0 J':>8} {'Mean J':>8} {'Std J':>8} {'Seeds':>6}")
    print("-" * 90)

    all_means = []
    all_stds = []

    for cnn_key, mob_key in cross_pairs:
        cnn_seeds = by_key.get(cnn_key, {})
        mob_seeds = by_key.get(mob_key, {})

        if not cnn_seeds or not mob_seeds:
            print(f"  Missing data for {cnn_key} or {mob_key}")
            continue

        # Compute Jaccard for each seed
        seed_jaccards = []
        seed0_j = None
        for seed in sorted(set(cnn_seeds.keys()) & set(mob_seeds.keys())):
            j = compute_jaccard(cnn_seeds[seed]["correct_mask"], mob_seeds[seed]["correct_mask"])
            seed_jaccards.append(j)
            if seed == 0:
                seed0_j = j

        mean_j = np.mean(seed_jaccards)
        std_j = np.std(seed_jaccards)
        n_seeds = len(seed_jaccards)
        all_means.append(mean_j)
        all_stds.append(std_j)

        cnn_params = cnn_seeds[0]["num_params"]
        mob_params = mob_seeds[0]["num_params"]
        cnn_label = f"c={cnn_key[1]}"
        mob_label = f"m={mob_key[1]:.2f}"

        print(f"{cnn_label:>10} {mob_label:>10} {cnn_params:>12,} {mob_params:>12,} "
              f"{seed0_j:>8.3f} {mean_j:>8.3f} {std_j:>8.3f} {n_seeds:>6}")

    print("-" * 90)
    print(f"{'Overall mean':>35} {'':<12} {'':<12} "
          f"{np.mean(all_means):>8.3f} {np.mean(all_stds):>8.3f}")

    # Also compute within-architecture adjacent Jaccard across all seeds
    print("\n\nWithin-Architecture Adjacent Jaccard (multi-seed):")
    print("=" * 70)

    for arch_name in ["scalecnn", "mobilenet"]:
        sizes = sorted(set(k[1] for k in by_key if k[0] == arch_name))
        adj_seed_means = []

        for i in range(len(sizes) - 1):
            key_a = (arch_name, sizes[i])
            key_b = (arch_name, sizes[i+1])
            seeds_a = by_key.get(key_a, {})
            seeds_b = by_key.get(key_b, {})

            seed_js = []
            for seed in sorted(set(seeds_a.keys()) & set(seeds_b.keys())):
                j = compute_jaccard(seeds_a[seed]["correct_mask"], seeds_b[seed]["correct_mask"])
                seed_js.append(j)

            if seed_js:
                adj_seed_means.append(np.mean(seed_js))

        if adj_seed_means:
            label = "ScaleCNN" if arch_name == "scalecnn" else "MobileNetV2"
            print(f"  {label}: mean adjacent Jaccard = {np.mean(adj_seed_means):.3f} "
                  f"± {np.std(adj_seed_means):.3f}")

    # Output LaTeX table format
    print("\n\n% LaTeX table (for paper update):")
    print("% Cross-architecture Jaccard overlap at matched parameter counts")
    print("% (mean ± std across 5 seeds, ratio < 2×)")
    print()

    for cnn_key, mob_key in cross_pairs:
        cnn_seeds = by_key.get(cnn_key, {})
        mob_seeds = by_key.get(mob_key, {})
        if not cnn_seeds or not mob_seeds:
            continue

        seed_jaccards = []
        for seed in sorted(set(cnn_seeds.keys()) & set(mob_seeds.keys())):
            j = compute_jaccard(cnn_seeds[seed]["correct_mask"], mob_seeds[seed]["correct_mask"])
            seed_jaccards.append(j)

        mean_j = np.mean(seed_jaccards)
        std_j = np.std(seed_jaccards)
        cnn_params = cnn_seeds[0]["num_params"]
        mob_params = mob_seeds[0]["num_params"]

        print(f"$c={cnn_key[1]}$ & $m={mob_key[1]:.2f}$ & {cnn_params//1000}K & "
              f"{mob_params//1000}K & ${mean_j:.3f} \\pm {std_j:.3f}$ \\\\")


if __name__ == "__main__":
    main()
