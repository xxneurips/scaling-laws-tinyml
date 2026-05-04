"""
Stage 3B: Full calibration analysis suite.
============================================
Loads saved softmax .npz files and computes:
  1. Reliability diagrams (per-bin accuracy vs confidence)
  2. Temperature scaling (learn T on validation split, report post-calibration ECE)
  3. Adaptive ECE (equal-mass bins)
  4. NLL and Brier score (from extended JSON)
  5. Test whether inverted-U ECE pattern persists after temperature scaling

Usage:
    python calibration_analysis.py                        # All datasets
    python calibration_analysis.py --dataset cifar100     # Specific dataset
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.optimize import minimize_scalar
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR.parent / "paper" / "figures"

ARCH_COLORS = {"scalecnn": "#2563eb", "mobilenet": "#dc2626", "effnet": "#16a34a"}
ARCH_MARKERS = {"scalecnn": "o", "mobilenet": "s", "effnet": "D"}
ARCH_LABELS = {"scalecnn": "ScaleCNN", "mobilenet": "MobileNetV2", "effnet": "EfficientNet-Lite"}


# =============================================================================
# Calibration metrics
# =============================================================================

def compute_ece(confidences, correct, n_bins=15):
    """Expected Calibration Error with uniform-width bins."""
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    bin_data = []

    for i in range(n_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_data.append({"center": (lo + hi) / 2, "acc": 0, "conf": 0, "count": 0})
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += (count / total) * abs(bin_acc - bin_conf)
        bin_data.append({"center": (lo + hi) / 2, "acc": bin_acc, "conf": bin_conf, "count": count})

    return float(ece), bin_data


def compute_adaptive_ece(confidences, correct, n_bins=15):
    """
    Adaptive ECE with equal-mass (equal-count) bins.
    Each bin has approximately the same number of samples.
    """
    sorted_indices = np.argsort(confidences)
    sorted_conf = confidences[sorted_indices]
    sorted_correct = correct[sorted_indices]

    n = len(confidences)
    bin_size = n // n_bins
    ece = 0.0

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        bin_conf = sorted_conf[start:end]
        bin_correct = sorted_correct[start:end]
        bin_acc = bin_correct.mean()
        bin_mean_conf = bin_conf.mean()
        ece += (len(bin_conf) / n) * abs(bin_acc - bin_mean_conf)

    return float(ece)


def compute_nll(probs, targets):
    """Negative log-likelihood (mean cross-entropy loss)."""
    n = len(targets)
    # Clip for numerical stability
    eps = 1e-10
    log_probs = np.log(np.clip(probs[np.arange(n), targets], eps, 1.0))
    return float(-log_probs.mean())


def compute_brier_score(probs, targets, num_classes):
    """Multi-class Brier score: (1/N) * sum_i sum_c (p_ic - y_ic)^2."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(targets)), targets] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


# =============================================================================
# Temperature scaling
# =============================================================================

def temperature_scale(logits, T):
    """Apply temperature scaling to logits."""
    scaled = logits / T
    # Softmax
    exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def learn_temperature(probs, targets, n_bins=15):
    """
    Learn optimal temperature T that minimizes NLL on the given data.
    Since we have probabilities, convert back to logits first.
    Returns optimal T.
    """
    # Convert probs to logits
    eps = 1e-10
    logits = np.log(np.clip(probs, eps, 1.0))

    def nll_at_T(T):
        scaled_probs = temperature_scale(logits, T)
        return compute_nll(scaled_probs, targets)

    if HAS_SCIPY:
        result = minimize_scalar(nll_at_T, bounds=(0.1, 10.0), method='bounded')
        return result.x
    else:
        # Grid search fallback
        best_T, best_nll = 1.0, float('inf')
        for T in np.arange(0.1, 10.0, 0.05):
            nll = nll_at_T(T)
            if nll < best_nll:
                best_nll = nll
                best_T = T
        return best_T


def apply_temperature_scaling(probs, targets, val_fraction=0.3):
    """
    Split data into val/test, learn T on val, return scaled metrics on test.
    Returns (T, pre_ece, post_ece, post_probs, test_targets, test_correct).
    """
    n = len(targets)
    n_val = int(n * val_fraction)

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    val_idx = indices[:n_val]
    test_idx = indices[n_val:]

    val_probs = probs[val_idx]
    val_targets = targets[val_idx]
    test_probs = probs[test_idx]
    test_targets = targets[test_idx]

    # Learn T on validation
    T = learn_temperature(val_probs, val_targets)

    # Apply to test
    eps = 1e-10
    test_logits = np.log(np.clip(test_probs, eps, 1.0))
    scaled_probs = temperature_scale(test_logits, T)

    # Compute ECE before and after
    test_conf_pre = test_probs.max(axis=1)
    test_preds_pre = test_probs.argmax(axis=1)
    test_correct = (test_preds_pre == test_targets).astype(float)
    pre_ece, _ = compute_ece(test_conf_pre, test_correct)

    test_conf_post = scaled_probs.max(axis=1)
    post_ece, _ = compute_ece(test_conf_post, test_correct)

    return T, pre_ece, post_ece, scaled_probs, test_targets, test_correct


# =============================================================================
# Load data
# =============================================================================

def load_runs(dataset_name):
    """Load all JSON results and matching .npz prob files for a dataset."""
    results_dir = BASE_DIR / "results" / dataset_name
    probs_dir = results_dir / "probs"

    runs = []
    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.stem.endswith("_summary") or json_file.stem.endswith("_results"):
            continue
        with open(json_file) as f:
            data = json.load(f)

        # Load softmax probs if available
        npz_path = probs_dir / f"{json_file.stem}_probs.npz"
        probs = None
        if npz_path.exists():
            probs = np.load(npz_path)["probs"]

        data["_probs"] = probs
        runs.append(data)

    return runs


# =============================================================================
# Analysis
# =============================================================================

def analyze_dataset(dataset_name):
    """Run full calibration analysis for one dataset."""
    print(f"\n{'='*70}")
    print(f"  Calibration Analysis: {dataset_name}")
    print(f"{'='*70}")

    runs = load_runs(dataset_name)
    if not runs:
        print(f"  No results found for {dataset_name}")
        return {}

    # Group by architecture and size
    by_arch = defaultdict(list)
    for r in runs:
        by_arch[r["arch"]].append(r)

    all_results = {}

    for arch, arch_runs in sorted(by_arch.items()):
        label = ARCH_LABELS.get(arch, arch)
        print(f"\n  --- {label} ---")

        # Aggregate by size
        by_size = defaultdict(list)
        for r in arch_runs:
            by_size[r["size_param"]].append(r)

        arch_results = []
        for size_param in sorted(by_size.keys()):
            size_runs = by_size[size_param]
            num_params = size_runs[0]["num_params"]
            num_classes = size_runs[0].get("num_classes", 100)

            # Existing metrics from JSON
            ece_values = [r["ece"] for r in size_runs]
            nll_values = [r.get("nll", r.get("test_ce_loss", None)) for r in size_runs]
            brier_values = [r.get("brier_score", None) for r in size_runs]

            # Compute adaptive ECE and temperature scaling from .npz
            adap_ece_values = []
            temp_values = []
            pre_ece_values = []
            post_ece_values = []

            for r in size_runs:
                probs = r.get("_probs")
                if probs is None:
                    continue
                targets = np.array(r["targets"])
                preds = probs.argmax(axis=1)
                correct = (preds == targets).astype(float)
                conf = probs.max(axis=1)

                # Adaptive ECE
                adap_ece = compute_adaptive_ece(conf, correct)
                adap_ece_values.append(adap_ece)

                # Temperature scaling
                T, pre_ece, post_ece, _, _, _ = apply_temperature_scaling(probs, targets)
                temp_values.append(T)
                pre_ece_values.append(pre_ece)
                post_ece_values.append(post_ece)

            entry = {
                "size_param": size_param,
                "num_params": num_params,
                "num_classes": num_classes,
                "n_seeds": len(size_runs),
                "ece_mean": float(np.mean(ece_values)),
                "ece_std": float(np.std(ece_values)),
            }

            if nll_values and nll_values[0] is not None:
                entry["nll_mean"] = float(np.mean([v for v in nll_values if v is not None]))
            if brier_values and brier_values[0] is not None:
                entry["brier_mean"] = float(np.mean([v for v in brier_values if v is not None]))
            if adap_ece_values:
                entry["adaptive_ece_mean"] = float(np.mean(adap_ece_values))
                entry["adaptive_ece_std"] = float(np.std(adap_ece_values))
            if temp_values:
                entry["temperature_mean"] = float(np.mean(temp_values))
                entry["pre_tempscale_ece_mean"] = float(np.mean(pre_ece_values))
                entry["post_tempscale_ece_mean"] = float(np.mean(post_ece_values))
                entry["post_tempscale_ece_std"] = float(np.std(post_ece_values))

            arch_results.append(entry)

            # Print summary
            size_label = f"c={int(size_param)}" if arch == "scalecnn" else f"w={size_param:.2f}"
            line = f"    {size_label:>8} | params={num_params:>10,} | ECE={entry['ece_mean']:.4f}"
            if "adaptive_ece_mean" in entry:
                line += f" | AdaECE={entry['adaptive_ece_mean']:.4f}"
            if "temperature_mean" in entry:
                line += f" | T={entry['temperature_mean']:.2f} | post-ECE={entry['post_tempscale_ece_mean']:.4f}"
            if "nll_mean" in entry:
                line += f" | NLL={entry['nll_mean']:.4f}"
            if "brier_mean" in entry:
                line += f" | Brier={entry['brier_mean']:.4f}"
            print(line)

        all_results[arch] = arch_results

    return all_results


# =============================================================================
# Plotting
# =============================================================================

def plot_reliability_diagrams(dataset_name, runs, save_path):
    """Plot reliability diagrams for representative models (smallest, middle, largest per arch)."""
    if not HAS_MPL:
        return

    by_arch = defaultdict(list)
    for r in runs:
        if r.get("_probs") is not None and r.get("seed", 0) == 0:
            by_arch[r["arch"]].append(r)

    n_archs = len(by_arch)
    if n_archs == 0:
        return

    fig, axes = plt.subplots(n_archs, 3, figsize=(14, 4 * n_archs))
    if n_archs == 1:
        axes = axes[np.newaxis, :]

    for row, (arch, arch_runs) in enumerate(sorted(by_arch.items())):
        arch_runs.sort(key=lambda r: r["num_params"])
        # Pick smallest, middle, largest
        indices = [0, len(arch_runs) // 2, -1]
        label = ARCH_LABELS.get(arch, arch)

        for col, idx in enumerate(indices):
            r = arch_runs[idx]
            probs = r["_probs"]
            targets = np.array(r["targets"])
            conf = probs.max(axis=1)
            correct = (probs.argmax(axis=1) == targets).astype(float)

            _, bin_data = compute_ece(conf, correct, n_bins=15)

            ax = axes[row, col]
            centers = [b["center"] for b in bin_data if b["count"] > 0]
            accs = [b["acc"] for b in bin_data if b["count"] > 0]
            confs = [b["conf"] for b in bin_data if b["count"] > 0]

            ax.bar(centers, accs, width=1/15 * 0.8, alpha=0.6, color=ARCH_COLORS.get(arch, "gray"),
                   label="Accuracy")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            sp = r["size_param"]
            sp_label = f"c={int(sp)}" if arch == "scalecnn" else f"w={sp:.2f}"
            ax.set_title(f"{label} {sp_label} ({r['num_params']:,} params)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

    plt.suptitle(f"Reliability Diagrams — {dataset_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Reliability diagrams saved: {save_path}")


def plot_calibration_comparison(all_results, dataset_name, save_path):
    """
    Plot ECE, post-temp-scale ECE, adaptive ECE, NLL, Brier vs params.
    Shows whether inverted-U persists after temperature scaling.
    """
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics = [
        ("ece_mean", "ece_std", "ECE (15 bins)", axes[0, 0]),
        ("adaptive_ece_mean", "adaptive_ece_std", "Adaptive ECE", axes[0, 1]),
        ("post_tempscale_ece_mean", "post_tempscale_ece_std", "Post-TempScale ECE", axes[0, 2]),
        ("nll_mean", None, "NLL", axes[1, 0]),
        ("brier_mean", None, "Brier Score", axes[1, 1]),
        ("temperature_mean", None, "Learned Temperature $T$", axes[1, 2]),
    ]

    for key, std_key, title, ax in metrics:
        for arch, results in all_results.items():
            params = [r["num_params"] for r in results if key in r]
            values = [r[key] for r in results if key in r]
            if not params:
                continue

            color = ARCH_COLORS.get(arch, "gray")
            marker = ARCH_MARKERS.get(arch, "o")
            label = ARCH_LABELS.get(arch, arch)

            if std_key and any(std_key in r for r in results):
                stds = [r.get(std_key, 0) for r in results if key in r]
                ax.errorbar(params, values, yerr=stds, fmt=f"{marker}-", capsize=3,
                           color=color, markersize=7, linewidth=1.5, label=label)
            else:
                ax.plot(params, values, f"{marker}-", color=color, markersize=7,
                       linewidth=1.5, label=label)

        ax.set_xscale("log")
        ax.set_xlabel("Parameters ($N$)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    plt.suptitle(f"Full Calibration Suite — {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Calibration comparison saved: {save_path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full calibration analysis")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar100", "cifar10", "tinyimagenet", "speechcommands"],
                        help="Specific dataset (default: all available)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else ["cifar100", "cifar10", "tinyimagenet", "speechcommands"]

    for dataset_name in datasets:
        results_dir = BASE_DIR / "results" / dataset_name
        if not results_dir.exists():
            print(f"[SKIP] No results for {dataset_name}")
            continue

        all_results = analyze_dataset(dataset_name)

        if all_results:
            # Save calibration results
            output_path = results_dir / "calibration_results.json"
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n[SAVED] {output_path}")

            # Generate plots
            runs = load_runs(dataset_name)
            plot_reliability_diagrams(dataset_name, runs,
                                     FIGURES_DIR / f"reliability_{dataset_name}.pdf")
            plot_calibration_comparison(all_results, dataset_name,
                                       FIGURES_DIR / f"calibration_suite_{dataset_name}.pdf")


if __name__ == "__main__":
    main()
