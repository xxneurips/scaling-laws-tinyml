"""
Stage 3D: Cross-architecture Jaccard overlap analysis.
========================================================
At matched parameter counts, compute pairwise Jaccard overlap across all 3
architecture families (ScaleCNN, MobileNetV2, EfficientNet-Lite).

Key question: "Does scale drive error set composition more than architecture?"

Method:
  1. For each pair of architectures, find parameter counts that are close (<2x ratio)
  2. Compute Jaccard overlap of error sets at matched sizes
  3. Compare cross-architecture Jaccard to within-architecture Jaccard (adjacent sizes)
  4. If cross-arch Jaccard > within-arch Jaccard at matched N, scale dominates architecture

Usage:
    python cross_arch_jaccard.py                          # All datasets
    python cross_arch_jaccard.py --dataset cifar100       # Specific dataset
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR.parent / "paper" / "figures"

ARCH_LABELS = {"scalecnn": "ScaleCNN", "mobilenet": "MobileNetV2", "effnet": "EfficientNet-Lite"}
PAIR_COLORS = {
    ("scalecnn", "mobilenet"): "#7c3aed",
    ("scalecnn", "effnet"): "#0891b2",
    ("mobilenet", "effnet"): "#ea580c",
}


# =============================================================================
# Jaccard computation
# =============================================================================

def compute_jaccard(correct_a, correct_b):
    """Jaccard index of error sets (samples that both get wrong)."""
    err_a = ~np.array(correct_a, dtype=bool)
    err_b = ~np.array(correct_b, dtype=bool)
    intersection = (err_a & err_b).sum()
    union = (err_a | err_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_overlap_ratio(correct_a, correct_b):
    """
    Fraction of the smaller error set contained in the larger.
    Measures error subset relationship.
    """
    err_a = ~np.array(correct_a, dtype=bool)
    err_b = ~np.array(correct_b, dtype=bool)
    n_a = err_a.sum()
    n_b = err_b.sum()
    intersection = (err_a & err_b).sum()
    smaller = min(n_a, n_b)
    if smaller == 0:
        return 1.0
    return float(intersection / smaller)


# =============================================================================
# Data loading
# =============================================================================

def load_runs(dataset_name):
    """Load all JSON results for a dataset."""
    results_dir = BASE_DIR / "results" / dataset_name
    if not results_dir.exists():
        return []

    runs = []
    for json_file in sorted(results_dir.glob("*.json")):
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


# =============================================================================
# Cross-architecture matching
# =============================================================================

def find_matched_pairs(runs_a, runs_b, max_ratio=2.0):
    """
    Find pairs of runs from different architectures with similar parameter counts.
    Returns list of (run_a, run_b, param_ratio) tuples.
    """
    pairs = []
    for ra in runs_a:
        for rb in runs_b:
            ratio = max(ra["num_params"], rb["num_params"]) / min(ra["num_params"], rb["num_params"])
            if ratio <= max_ratio:
                pairs.append((ra, rb, ratio))
    return pairs


# =============================================================================
# Analysis
# =============================================================================

def analyze_dataset(dataset_name):
    """Full cross-architecture Jaccard analysis for one dataset."""
    print(f"\n{'='*70}")
    print(f"  Cross-Architecture Jaccard: {dataset_name}")
    print(f"{'='*70}")

    runs = load_runs(dataset_name)
    if not runs:
        print(f"  No results found for {dataset_name}")
        return {}

    # Group by architecture, seed=0 only for consistency
    by_arch = defaultdict(list)
    for r in runs:
        if r.get("seed", 0) == 0:
            by_arch[r["arch"]].append(r)

    archs = sorted(by_arch.keys())
    if len(archs) < 2:
        print(f"  Need at least 2 architectures, found: {archs}")
        return {}

    results = {}

    # --- Full Jaccard matrices per architecture (within-arch) ---
    print("\n  Within-Architecture Jaccard (seed=0, all size pairs):")
    within_arch_jaccards = {}

    for arch in archs:
        label = ARCH_LABELS.get(arch, arch)
        arch_runs = sorted(by_arch[arch], key=lambda r: r["num_params"])

        n = len(arch_runs)
        jac_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                jac_matrix[i, j] = compute_jaccard(
                    arch_runs[i]["correct_mask"], arch_runs[j]["correct_mask"]
                )

        within_arch_jaccards[arch] = {
            "matrix": jac_matrix.tolist(),
            "sizes": [r["size_param"] for r in arch_runs],
            "params": [r["num_params"] for r in arch_runs],
        }

        # Adjacent-size Jaccard (measures within-arch overlap at similar scales)
        adj_jacs = [jac_matrix[i, i+1] for i in range(n-1)]
        if adj_jacs:
            print(f"    {label}: adjacent-size J = {np.mean(adj_jacs):.3f} "
                  f"+/- {np.std(adj_jacs):.3f} (n={len(adj_jacs)} pairs)")

    results["within_arch"] = within_arch_jaccards

    # --- Cross-architecture Jaccard at matched parameter counts ---
    print("\n  Cross-Architecture Jaccard (matched params, ratio < 2x):")
    cross_arch_results = {}

    for arch_a, arch_b in combinations(archs, 2):
        label_a = ARCH_LABELS.get(arch_a, arch_a)
        label_b = ARCH_LABELS.get(arch_b, arch_b)
        pair_key = f"{arch_a}_vs_{arch_b}"

        pairs = find_matched_pairs(by_arch[arch_a], by_arch[arch_b], max_ratio=2.0)

        if not pairs:
            print(f"    {label_a} vs {label_b}: no matched pairs found")
            continue

        pair_data = []
        jaccards = []
        for ra, rb, ratio in pairs:
            j = compute_jaccard(ra["correct_mask"], rb["correct_mask"])
            overlap = compute_overlap_ratio(ra["correct_mask"], rb["correct_mask"])
            jaccards.append(j)
            pair_data.append({
                f"{arch_a}_size": ra["size_param"],
                f"{arch_a}_params": ra["num_params"],
                f"{arch_b}_size": rb["size_param"],
                f"{arch_b}_params": rb["num_params"],
                "param_ratio": ratio,
                "jaccard": j,
                "overlap_ratio": overlap,
            })

        cross_arch_results[pair_key] = pair_data

        print(f"    {label_a} vs {label_b}: J = {np.mean(jaccards):.3f} "
              f"+/- {np.std(jaccards):.3f} (n={len(jaccards)} matched pairs)")

    results["cross_arch"] = cross_arch_results

    # --- Key comparison: cross-arch vs within-arch at matched scale ---
    print("\n  Scale vs Architecture (key test):")
    for pair_key, pair_data in cross_arch_results.items():
        cross_j = np.mean([p["jaccard"] for p in pair_data])
        within_js = []
        for arch in archs:
            waj = within_arch_jaccards.get(arch, {})
            matrix = np.array(waj.get("matrix", []))
            if matrix.size > 0:
                n = matrix.shape[0]
                adj = [matrix[i, i+1] for i in range(n-1)]
                within_js.extend(adj)

        if within_js:
            within_j = np.mean(within_js)
            dominance = "SCALE dominates" if cross_j > within_j else "ARCHITECTURE dominates"
            print(f"    {pair_key}: cross-arch J={cross_j:.3f} vs within-arch adj J={within_j:.3f} -> {dominance}")

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_jaccard_matrices(results, dataset_name, save_path):
    """Plot within-architecture Jaccard heatmaps."""
    if not HAS_MPL:
        return

    within = results.get("within_arch", {})
    n_archs = len(within)
    if n_archs == 0:
        return

    fig, axes = plt.subplots(1, n_archs, figsize=(6 * n_archs, 5))
    if n_archs == 1:
        axes = [axes]

    for ax, (arch, data) in zip(axes, sorted(within.items())):
        matrix = np.array(data["matrix"])
        sizes = data["sizes"]
        label = ARCH_LABELS.get(arch, arch)

        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(sizes)))
        ax.set_yticks(range(len(sizes)))
        if arch == "scalecnn":
            tick_labels = [f"c={int(s)}" for s in sizes]
        else:
            tick_labels = [f"w={s:.2f}" for s in sizes]
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_title(f"{label}")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Jaccard")

        # Annotate values
        for i in range(len(sizes)):
            for j in range(len(sizes)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                       fontsize=6, color="black" if matrix[i,j] > 0.5 else "white")

    plt.suptitle(f"Within-Architecture Error Overlap — {dataset_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Jaccard matrices saved: {save_path}")


def plot_cross_arch_scatter(results, dataset_name, save_path):
    """Scatter plot: cross-arch Jaccard vs geometric mean of params."""
    if not HAS_MPL:
        return

    cross = results.get("cross_arch", {})
    if not cross:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for pair_key, pair_data in cross.items():
        archs = pair_key.split("_vs_")
        color = PAIR_COLORS.get(tuple(archs), "gray")
        label_a = ARCH_LABELS.get(archs[0], archs[0])
        label_b = ARCH_LABELS.get(archs[1], archs[1])

        geom_means = []
        jaccards = []
        for p in pair_data:
            params_a = p[f"{archs[0]}_params"]
            params_b = p[f"{archs[1]}_params"]
            geom_means.append(np.sqrt(params_a * params_b))
            jaccards.append(p["jaccard"])

        ax.scatter(geom_means, jaccards, color=color, s=60, alpha=0.7,
                  label=f"{label_a} vs {label_b}")

    ax.set_xscale("log")
    ax.set_xlabel("Geometric Mean of Parameters")
    ax.set_ylabel("Cross-Architecture Jaccard")
    ax.set_title(f"Cross-Architecture Error Overlap — {dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Cross-arch scatter saved: {save_path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-architecture Jaccard overlap analysis")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar100", "cifar10", "tinyimagenet", "speechcommands"])
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else ["cifar100", "cifar10", "tinyimagenet", "speechcommands"]

    for dataset_name in datasets:
        results_dir = BASE_DIR / "results" / dataset_name
        if not results_dir.exists():
            print(f"[SKIP] No results for {dataset_name}")
            continue

        results = analyze_dataset(dataset_name)

        if results:
            # Save
            output_path = results_dir / "cross_arch_jaccard_results.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n[SAVED] {output_path}")

            # Plots
            plot_jaccard_matrices(results, dataset_name,
                                FIGURES_DIR / f"jaccard_matrices_{dataset_name}.pdf")
            plot_cross_arch_scatter(results, dataset_name,
                                  FIGURES_DIR / f"cross_arch_jaccard_{dataset_name}.pdf")


if __name__ == "__main__":
    main()
