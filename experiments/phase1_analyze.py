"""
Stage 1 Analysis: scaling-law extraction
==========================================
Reads Stage 1 training results for both ScaleCNN and MobileNetV2 and produces:
  1. Scaling law — log-log error vs params, both architectures, fit lines
  2. Local exponents — alpha_local vs params, Kaplan reference line
  3. Error overlap — Jaccard vs params (reference = largest model)
  4. Per-class heatmap — ScaleCNN classes sorted by difficulty
  5. Class triage — Gini coefficient + bottom-5/10/20 accuracy vs size
  6. Calibration — ECE vs params, both architectures

Usage:
    python phase1_analyze.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found. Plots will be skipped.")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not found. Using numpy for fits.")


BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "phase1"
FIGURES_DIR = BASE_DIR.parent / "paper" / "figures"

# Publication styling
COLORS = {
    "cnn": "#2563eb",       # blue
    "mob": "#dc2626",       # red
    "cnn_light": "#93bbfd",
    "mob_light": "#fca5a5",
    "kaplan": "#6b7280",    # gray
}
MARKERS = {"cnn": "o", "mob": "s"}
LABELS = {"cnn": "ScaleCNN", "mob": "MobileNetV2"}


# =============================================================================
# Data Loading
# =============================================================================

def load_results():
    """Load all Stage 1 JSON results (both cnn_*.json and mob_*.json)."""
    results = {"cnn": [], "mob": []}
    for f in sorted(RESULTS_DIR.glob("cnn_*.json")):
        with open(f) as fh:
            results["cnn"].append(json.load(fh))
    for f in sorted(RESULTS_DIR.glob("mob_*.json")):
        with open(f) as fh:
            results["mob"].append(json.load(fh))
    return results


def aggregate_by_size(results_list):
    """Group results by size_param, compute mean and std across seeds."""
    groups = defaultdict(list)
    for r in results_list:
        groups[r["size_param"]].append(r)

    summary = []
    for size in sorted(groups.keys()):
        runs = groups[size]
        per_class_accs = np.array([r["per_class_acc"] for r in runs])
        entry = {
            "size_param": size,
            "arch": runs[0]["arch"],
            "num_params": runs[0]["num_params"],
            "n_seeds": len(runs),
            "top1_acc_mean": np.mean([r["top1_acc"] for r in runs]),
            "top1_acc_std": np.std([r["top1_acc"] for r in runs]),
            "top5_acc_mean": np.mean([r["top5_acc"] for r in runs]),
            "top5_acc_std": np.std([r["top5_acc"] for r in runs]),
            "ece_mean": np.mean([r["ece"] for r in runs]),
            "ece_std": np.std([r["ece"] for r in runs]),
            "mean_error_severity_mean": np.mean([r["mean_error_severity"] for r in runs]),
            "converged_all": all(r["converged"] for r in runs),
            "per_class_acc_mean": np.mean(per_class_accs, axis=0).tolist(),
            "per_class_acc_std": np.std(per_class_accs, axis=0).tolist(),
        }
        summary.append(entry)

    return summary


# =============================================================================
# Fitting
# =============================================================================

def fit_power_law(params, accuracies):
    """Fit: accuracy = C * params^alpha on log-log scale."""
    log_n = np.log10(params)
    log_acc = np.log10(accuracies)

    if HAS_SCIPY:
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(log_n, log_acc)
    else:
        slope, intercept, r_value, p_value, std_err = _numpy_linregress(log_n, log_acc)

    predicted = slope * log_n + intercept
    residuals = log_acc - predicted

    return {
        "alpha": slope,
        "C": 10 ** intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
        "residuals": residuals.tolist(),
        "log_params": log_n.tolist(),
        "log_acc": log_acc.tolist(),
    }


def fit_power_law_error(params, error_rates):
    """Fit: error = C * params^(-alpha) on log-log scale. alpha returned positive."""
    log_n = np.log10(params)
    log_err = np.log10(error_rates)

    if HAS_SCIPY:
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(log_n, log_err)
    else:
        slope, intercept, r_value, p_value, std_err = _numpy_linregress(log_n, log_err)

    predicted = slope * log_n + intercept
    residuals = log_err - predicted

    return {
        "alpha": -slope,  # positive exponent
        "C": 10 ** intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err if std_err else 0.0,
        "residuals": residuals.tolist(),
    }


def _numpy_linregress(x, y):
    """Fallback linear regression using numpy."""
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    predicted = slope * x + intercept
    ss_res = ((y - predicted) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_value = np.sqrt(1 - ss_res / ss_tot) if ss_tot > 0 else 0
    std_err = np.sqrt(ss_res / (n - 2) / (sxx - sx * sx / n)) if n > 2 else 0
    return slope, intercept, r_value, None, std_err


def compute_local_exponents(summary):
    """Compute local finite-difference scaling exponent between adjacent sizes."""
    exponents = []
    for i in range(1, len(summary)):
        n1, n2 = summary[i - 1]["num_params"], summary[i]["num_params"]
        e1 = 1.0 - summary[i - 1]["top1_acc_mean"]
        e2 = 1.0 - summary[i]["top1_acc_mean"]
        if e1 > 0 and e2 > 0 and n1 != n2:
            alpha_local = -(np.log10(e2) - np.log10(e1)) / (np.log10(n2) - np.log10(n1))
            midpoint_params = np.sqrt(n1 * n2)  # geometric mean
            exponents.append({
                "params_mid": midpoint_params,
                "alpha_local": alpha_local,
                "n1": n1, "n2": n2,
                "label": f"{summary[i-1]['size_param']}->{summary[i]['size_param']}",
            })
    return exponents


def compute_gini(per_class_acc):
    """Compute Gini coefficient of per-class accuracy distribution."""
    x = np.sort(np.array(per_class_acc))
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))) if np.sum(x) > 0 else 0.0


# =============================================================================
# Error Overlap (Jaccard)
# =============================================================================

def compute_jaccard_matrix(results_list):
    """Compute pairwise Jaccard overlap of error sets (seed=0 only)."""
    by_size = {}
    for r in results_list:
        if r["seed"] == 0:
            by_size[r["size_param"]] = r

    sizes = sorted(by_size.keys())
    n = len(sizes)
    jaccard = np.zeros((n, n))

    for i, s1 in enumerate(sizes):
        c1 = np.array(by_size[s1]["correct_mask"], dtype=bool)
        e1 = ~c1
        for j, s2 in enumerate(sizes):
            c2 = np.array(by_size[s2]["correct_mask"], dtype=bool)
            e2 = ~c2
            intersection = (e1 & e2).sum()
            union = (e1 | e2).sum()
            jaccard[i, j] = intersection / union if union > 0 else 1.0

    return sizes, jaccard


def compute_jaccard_vs_reference(results_list):
    """Jaccard overlap vs largest model (seed=0)."""
    by_size = {}
    for r in results_list:
        if r["seed"] == 0:
            by_size[r["size_param"]] = r

    sizes = sorted(by_size.keys())
    ref = sizes[-1]
    ref_correct = np.array(by_size[ref]["correct_mask"], dtype=bool)
    ref_errors = ~ref_correct

    overlaps = {}
    for s in sizes:
        correct = np.array(by_size[s]["correct_mask"], dtype=bool)
        errors = ~correct
        intersection = (errors & ref_errors).sum()
        union = (errors | ref_errors).sum()
        overlaps[s] = float(intersection / union) if union > 0 else 1.0

    return overlaps, ref, by_size[ref]["num_params"]


# =============================================================================
# Plotting — 6 Publication Figures
# =============================================================================

def _setup_style():
    """Configure publication-ready plot style."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_fig1_scaling_law(cnn_summary, mob_summary, cnn_fit, mob_fit, save_path):
    """Figure 1: Log-log error vs parameters, both architectures with fit lines."""
    if not HAS_MPL:
        return
    _setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel (a): Scaling curves ---
    for summary, fit, arch in [(cnn_summary, cnn_fit, "cnn"), (mob_summary, mob_fit, "mob")]:
        params = np.array([s["num_params"] for s in summary])
        err_mean = 1.0 - np.array([s["top1_acc_mean"] for s in summary])
        err_std = np.array([s["top1_acc_std"] for s in summary])

        ax1.errorbar(params, err_mean, yerr=err_std, fmt=f"{MARKERS[arch]}-",
                     capsize=3, color=COLORS[arch], markersize=7, linewidth=1.8,
                     label=f"{LABELS[arch]} (data)")

        # Power-law fit line
        x_fit = np.logspace(np.log10(params.min() * 0.5), np.log10(params.max() * 2), 100)
        y_fit = fit["C"] * x_fit ** (-fit["alpha"])
        ax1.plot(x_fit, y_fit, "--", color=COLORS[arch], linewidth=1.2, alpha=0.7,
                 label=rf"{LABELS[arch]}: $\alpha$={fit['alpha']:.3f}, $R^2$={fit['r_squared']:.3f}")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Parameters ($N$)")
    ax1.set_ylabel("Top-1 Error Rate")
    ax1.set_title("(a) Scaling Law: Error vs. Model Size")
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(True, alpha=0.3, which="both")

    # --- Panel (b): Residuals ---
    for fit, arch in [(cnn_fit, "cnn"), (mob_fit, "mob")]:
        residuals = np.array(fit["residuals"])
        # Use params from the fit
        if arch == "cnn":
            params = np.array([s["num_params"] for s in cnn_summary])
        else:
            params = np.array([s["num_params"] for s in mob_summary])
        ax2.scatter(params, residuals, color=COLORS[arch], s=50, marker=MARKERS[arch],
                    label=LABELS[arch], zorder=3)

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Parameters ($N$)")
    ax2.set_ylabel("Residual (log$_{10}$ scale)")
    ax2.set_title("(b) Residuals from Power-Law Fit")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Figure 1 saved: {save_path}")


def plot_fig2_local_exponents(cnn_exponents, mob_exponents, save_path):
    """Figure 2: Local scaling exponent vs parameters, Kaplan reference."""
    if not HAS_MPL:
        return
    _setup_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    for exponents, arch in [(cnn_exponents, "cnn"), (mob_exponents, "mob")]:
        params = [e["params_mid"] for e in exponents]
        alphas = [e["alpha_local"] for e in exponents]
        ax.plot(params, alphas, f"{MARKERS[arch]}-", color=COLORS[arch],
                markersize=8, linewidth=1.8, label=LABELS[arch])

    # Reference lines
    ax.axhline(0.076, color=COLORS["kaplan"], linestyle="--", linewidth=1.2, alpha=0.7,
               label=r"Kaplan et al. $\alpha \approx 0.076$")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters ($N$) — geometric midpoint")
    ax.set_ylabel(r"Local Exponent $\alpha_{\mathrm{local}}$")
    ax.set_title("Local Scaling Exponent Decay")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(-0.05, 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Figure 2 saved: {save_path}")


def plot_fig3_error_overlap(cnn_jaccard, mob_jaccard, save_path,
                            cnn_summary=None, mob_summary=None):
    """Figure 3: Jaccard overlap vs parameters (reference = largest model).

    Revised: uses log-scale parameter count on x-axis with separate panels
    for ScaleCNN and MobileNetV2, as recommended by reviewer.
    """
    if not HAS_MPL:
        return
    _setup_style()

    # Build size_param -> num_params lookup from summaries
    param_lookup = {}
    if cnn_summary:
        for s in cnn_summary:
            param_lookup[("cnn", s["size_param"])] = s["num_params"]
    if mob_summary:
        for s in mob_summary:
            param_lookup[("mob", s["size_param"])] = s["num_params"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for jaccard_data, arch, ax in [(cnn_jaccard, "cnn", ax1), (mob_jaccard, "mob", ax2)]:
        if jaccard_data is None:
            continue
        overlaps, ref_size, ref_params = jaccard_data
        sizes = sorted(overlaps.keys())
        jaccards = [overlaps[s] for s in sizes]
        # Map size_param to actual parameter count
        params = [param_lookup.get((arch, s), s) for s in sizes]
        ax.plot(params, jaccards, f"{MARKERS[arch]}-", color=COLORS[arch],
                markersize=8, linewidth=1.8,
                label=f"{LABELS[arch]} (vs. largest)")
        ax.set_xscale("log")
        ax.set_xlabel("Parameters ($N$)")
        ax.set_title(f"({'a' if arch == 'cnn' else 'b'}) {LABELS[arch]}")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(0, 1.05)

    ax1.set_ylabel("Jaccard Overlap (Error Sets)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Figure 3 saved: {save_path}")


def plot_fig4_per_class_heatmap(cnn_summary, save_path):
    """Figure 4: Per-class accuracy heatmap for ScaleCNN, classes sorted by difficulty."""
    if not HAS_MPL:
        return
    _setup_style()

    per_class = np.array([s["per_class_acc_mean"] for s in cnn_summary])  # (n_sizes, 100)

    # Sort classes by difficulty (mean accuracy across all sizes, ascending)
    mean_class_acc = per_class.mean(axis=0)
    sort_idx = np.argsort(mean_class_acc)
    per_class_sorted = per_class[:, sort_idx]

    size_labels = [f"c={int(s['size_param'])}" for s in cnn_summary]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(per_class_sorted, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(size_labels)))
    ax.set_yticklabels(size_labels)
    ax.set_xlabel("CIFAR-100 Class (sorted by difficulty, hardest left)")
    ax.set_ylabel("ScaleCNN Configuration")
    ax.set_title("Per-Class Accuracy Across Model Sizes (ScaleCNN)")
    cbar = plt.colorbar(im, ax=ax, label="Per-Class Accuracy", shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Figure 4 saved: {save_path}")


def plot_fig5_class_triage(cnn_summary, mob_summary, save_path):
    """Figure 5: Class triage — Gini coefficient + bottom-K accuracy vs model size."""
    if not HAS_MPL:
        return
    _setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel (a): Gini coefficient ---
    for summary, arch in [(cnn_summary, "cnn"), (mob_summary, "mob")]:
        params = [s["num_params"] for s in summary]
        ginis = [compute_gini(s["per_class_acc_mean"]) for s in summary]
        ax1.plot(params, ginis, f"{MARKERS[arch]}-", color=COLORS[arch],
                 markersize=8, linewidth=1.8, label=LABELS[arch])

    ax1.set_xscale("log")
    ax1.set_xlabel("Parameters ($N$)")
    ax1.set_ylabel("Gini Coefficient of Per-Class Accuracy")
    ax1.set_title("(a) Class Inequality vs. Model Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")

    # --- Panel (b): Bottom-K accuracy ---
    for summary, arch in [(cnn_summary, "cnn"), (mob_summary, "mob")]:
        params = [s["num_params"] for s in summary]
        bottom5 = []
        bottom10 = []
        bottom20 = []
        for s in summary:
            pca = np.array(s["per_class_acc_mean"])
            sorted_acc = np.sort(pca)
            bottom5.append(sorted_acc[:5].mean())
            bottom10.append(sorted_acc[:10].mean())
            bottom20.append(sorted_acc[:20].mean())

        ax2.plot(params, bottom5, f"{MARKERS[arch]}-", color=COLORS[arch],
                 markersize=7, linewidth=1.8, label=f"{LABELS[arch]} bottom-5")
        ax2.plot(params, bottom10, f"{MARKERS[arch]}--", color=COLORS[arch],
                 markersize=5, linewidth=1.2, alpha=0.7, label=f"{LABELS[arch]} bottom-10")
        ax2.plot(params, bottom20, f"{MARKERS[arch]}:", color=COLORS[arch],
                 markersize=4, linewidth=1.0, alpha=0.5, label=f"{LABELS[arch]} bottom-20")

    ax2.set_xscale("log")
    ax2.set_xlabel("Parameters ($N$)")
    ax2.set_ylabel("Mean Accuracy")
    ax2.set_title("(b) Hardest-Class Accuracy vs. Model Size")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Figure 5 saved: {save_path}")


def plot_fig6_calibration(cnn_summary, mob_summary, save_path):
    """Figure 6: ECE vs parameters, both architectures."""
    if not HAS_MPL:
        return
    _setup_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    for summary, arch in [(cnn_summary, "cnn"), (mob_summary, "mob")]:
        params = [s["num_params"] for s in summary]
        ece_mean = [s["ece_mean"] for s in summary]
        ece_std = [s["ece_std"] for s in summary]

        ax.errorbar(params, ece_mean, yerr=ece_std, fmt=f"{MARKERS[arch]}-",
                    capsize=3, color=COLORS[arch], markersize=8, linewidth=1.8,
                    label=LABELS[arch])

    ax.set_xscale("log")
    ax.set_xlabel("Parameters ($N$)")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("Calibration vs. Model Size")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Annotate the inverted-U for CNN
    ax.annotate("CNN peak\n(inverted-U)", xy=(1.2e6, 0.11), fontsize=8,
                ha="center", color=COLORS["cnn"], alpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Figure 6 saved: {save_path}")


# =============================================================================
# Print Helpers
# =============================================================================

def print_summary_table(summary, arch_name):
    """Print formatted summary table for one architecture."""
    print(f"\n{'='*70}")
    print(f"  {arch_name}")
    print(f"{'='*70}")
    print(f"{'Size':>8} {'Params':>10} {'Top-1':>14} {'Top-5':>14} {'ECE':>12}")
    print("-" * 65)
    for s in summary:
        sp = s["size_param"]
        label = f"c={int(sp)}" if s["arch"] == "scalecnn" else f"w={sp:.2f}"
        print(
            f"{label:>8} "
            f"{s['num_params']:>10,} "
            f"{s['top1_acc_mean']:.2%} +/- {s['top1_acc_std']:.2%} "
            f"{s['top5_acc_mean']:.2%} +/- {s['top5_acc_std']:.2%} "
            f"{s['ece_mean']:.4f}"
        )


def print_fit_results(fit, arch_name):
    """Print power-law fit results."""
    print(f"\n  {arch_name} Power-Law Fit (error ~ N^-alpha):")
    print(f"    alpha = {fit['alpha']:.4f} +/- {fit['std_err']:.4f}")
    print(f"    R^2   = {fit['r_squared']:.4f}")


def print_local_exponents(exponents, arch_name):
    """Print local exponents table."""
    print(f"\n  {arch_name} Local Exponents:")
    print(f"  {'Transition':>20} {'Params':>18} {'alpha_local':>12}")
    print(f"  {'-'*55}")
    for e in exponents:
        print(f"  {e['label']:>20} {e['n1']:>8,} -> {e['n2']:>8,} {e['alpha_local']:>10.3f}")


# =============================================================================
# Main
# =============================================================================

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    results = load_results()
    n_cnn = len(results["cnn"])
    n_mob = len(results["mob"])
    print(f"Loaded {n_cnn} CNN runs + {n_mob} MobNet runs = {n_cnn + n_mob} total")

    if n_cnn == 0 and n_mob == 0:
        print(f"[ERROR] No results found in {RESULTS_DIR}")
        print("Run phase1_train.py first.")
        return

    # Aggregate
    cnn_summary = aggregate_by_size(results["cnn"]) if results["cnn"] else []
    mob_summary = aggregate_by_size(results["mob"]) if results["mob"] else []

    print(f"Aggregated: {len(cnn_summary)} CNN sizes, {len(mob_summary)} MobNet sizes")

    # --- Summary Tables ---
    if cnn_summary:
        print_summary_table(cnn_summary, "ScaleCNN")
    if mob_summary:
        print_summary_table(mob_summary, "MobileNetV2")

    # --- Power-Law Fits (on error rate) ---
    fits = {}
    for summary, name in [(cnn_summary, "cnn"), (mob_summary, "mob")]:
        if not summary:
            continue
        params = np.array([s["num_params"] for s in summary])
        error = 1.0 - np.array([s["top1_acc_mean"] for s in summary])
        fit = fit_power_law_error(params, error)
        fits[name] = fit
        arch_label = "ScaleCNN" if name == "cnn" else "MobileNetV2"
        print_fit_results(fit, arch_label)

    # --- Local Exponents ---
    cnn_exponents = compute_local_exponents(cnn_summary) if cnn_summary else []
    mob_exponents = compute_local_exponents(mob_summary) if mob_summary else []

    if cnn_exponents:
        print_local_exponents(cnn_exponents, "ScaleCNN")
    if mob_exponents:
        print_local_exponents(mob_exponents, "MobileNetV2")

    # --- Error Overlap (Jaccard) ---
    cnn_jaccard = compute_jaccard_vs_reference(results["cnn"]) if results["cnn"] else None
    mob_jaccard = compute_jaccard_vs_reference(results["mob"]) if results["mob"] else None

    if cnn_jaccard:
        overlaps, ref, ref_p = cnn_jaccard
        print(f"\n  ScaleCNN Jaccard (vs c={int(ref)}, {ref_p:,} params):")
        for s in sorted(overlaps.keys()):
            print(f"    c={int(s):>3}: J = {overlaps[s]:.3f}")

    if mob_jaccard:
        overlaps, ref, ref_p = mob_jaccard
        print(f"\n  MobileNetV2 Jaccard (vs w={ref:.2f}, {ref_p:,} params):")
        for s in sorted(overlaps.keys()):
            print(f"    w={s:.2f}: J = {overlaps[s]:.3f}")

    # --- Class Triage / Gini ---
    for summary, name in [(cnn_summary, "ScaleCNN"), (mob_summary, "MobileNetV2")]:
        if not summary:
            continue
        print(f"\n  {name} Class Inequality:")
        print(f"  {'Size':>8} {'Params':>10} {'Gini':>8} {'Bot-5':>8} {'Bot-10':>8} {'Bot-20':>8}")
        for s in summary:
            pca = np.array(s["per_class_acc_mean"])
            gini = compute_gini(pca)
            sorted_acc = np.sort(pca)
            sp = s["size_param"]
            label = f"c={int(sp)}" if s["arch"] == "scalecnn" else f"w={sp:.2f}"
            print(
                f"  {label:>8} {s['num_params']:>10,} "
                f"{gini:>8.3f} "
                f"{sorted_acc[:5].mean():>7.1%} "
                f"{sorted_acc[:10].mean():>7.1%} "
                f"{sorted_acc[:20].mean():>7.1%}"
            )

    # --- Generate all 6 figures ---
    print(f"\n{'='*70}")
    print("Generating publication figures...")

    if cnn_summary and mob_summary and "cnn" in fits and "mob" in fits:
        plot_fig1_scaling_law(cnn_summary, mob_summary, fits["cnn"], fits["mob"],
                             FIGURES_DIR / "fig1_scaling_law.pdf")

    if cnn_exponents or mob_exponents:
        plot_fig2_local_exponents(cnn_exponents, mob_exponents,
                                  FIGURES_DIR / "fig2_local_exponents.pdf")

    if cnn_jaccard or mob_jaccard:
        plot_fig3_error_overlap(cnn_jaccard, mob_jaccard,
                                FIGURES_DIR / "fig3_error_overlap.pdf",
                                cnn_summary=cnn_summary,
                                mob_summary=mob_summary)

    if cnn_summary:
        plot_fig4_per_class_heatmap(cnn_summary,
                                     FIGURES_DIR / "fig4_per_class_heatmap.pdf")

    if cnn_summary and mob_summary:
        plot_fig5_class_triage(cnn_summary, mob_summary,
                               FIGURES_DIR / "fig5_class_triage.pdf")

    if cnn_summary and mob_summary:
        plot_fig6_calibration(cnn_summary, mob_summary,
                              FIGURES_DIR / "fig6_calibration.pdf")

    # --- Save summary JSON ---
    summary_out = {
        "cnn_fit": {k: v for k, v in fits.get("cnn", {}).items() if k != "residuals"},
        "mob_fit": {k: v for k, v in fits.get("mob", {}).items() if k != "residuals"},
        "cnn_summary": cnn_summary,
        "mob_summary": mob_summary,
    }
    out_path = RESULTS_DIR / "phase1_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\n[SAVED] Summary: {out_path}")

    print(f"\n{'='*70}")
    print("Stage 1 analysis complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
