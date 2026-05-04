"""
Stage 3C: AIC/BIC model selection for scaling law functional form.
===================================================================
Fits 4 candidate models to error-vs-params data:
  1. Power law:          E(N) = C * N^(-alpha)
  2. Log-linear:         E(N) = a * log(N) + b
  3. BNSL (2-segment):   E(N) = C1*N^(-a1) if N < N_break else C2*N^(-a2)
  4. Stretched exponential: E(N) = C * exp(-b * N^c)

Reports AIC/BIC per dataset x architecture (12+ comparisons).

Usage:
    python model_selection.py                         # All datasets
    python model_selection.py --dataset cifar100      # Specific dataset
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import curve_fit, minimize
    from scipy.special import gammaln
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy required for model selection. Install: pip install scipy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR.parent / "paper" / "figures"

ARCH_COLORS = {"scalecnn": "#2563eb", "mobilenet": "#dc2626", "effnet": "#16a34a"}
ARCH_LABELS = {"scalecnn": "ScaleCNN", "mobilenet": "MobileNetV2", "effnet": "EfficientNet-Lite"}


# =============================================================================
# Candidate models
# =============================================================================

def power_law(N, C, alpha):
    """E(N) = C * N^(-alpha)"""
    return C * np.power(N, -alpha)


def log_linear(N, a, b):
    """E(N) = a * log10(N) + b"""
    return a * np.log10(N) + b


def stretched_exp(N, C, b, c):
    """E(N) = C * exp(-b * N^c)"""
    return C * np.exp(-b * np.power(N, c))


def bnsl_2segment(N, C1, a1, C2, a2, N_break):
    """
    Broken Neural Scaling Law (2-segment power law).
    E(N) = C1*N^(-a1) for N < N_break
    E(N) = C2*N^(-a2) for N >= N_break
    """
    result = np.empty_like(N, dtype=float)
    mask = N < N_break
    result[mask] = C1 * np.power(N[mask], -a1)
    result[~mask] = C2 * np.power(N[~mask], -a2)
    return result


# =============================================================================
# Fitting
# =============================================================================

def fit_power_law(params, errors):
    """Fit power law: E = C * N^(-alpha)."""
    log_N = np.log(params)
    log_E = np.log(errors)
    # Linear regression in log space
    slope, intercept = np.polyfit(log_N, log_E, 1)
    C = np.exp(intercept)
    alpha = -slope
    predicted = power_law(params, C, alpha)
    residuals = errors - predicted
    ss_res = np.sum(residuals ** 2)
    return {"name": "Power Law", "params": {"C": C, "alpha": alpha},
            "k": 2, "predicted": predicted, "ss_res": ss_res}


def fit_log_linear(params, errors):
    """Fit log-linear: E = a * log10(N) + b."""
    log_N = np.log10(params)
    slope, intercept = np.polyfit(log_N, errors, 1)
    predicted = log_linear(params, slope, intercept)
    residuals = errors - predicted
    ss_res = np.sum(residuals ** 2)
    return {"name": "Log-Linear", "params": {"a": slope, "b": intercept},
            "k": 2, "predicted": predicted, "ss_res": ss_res}


def fit_stretched_exp(params, errors):
    """Fit stretched exponential: E = C * exp(-b * N^c)."""
    if not HAS_SCIPY:
        return None
    try:
        popt, _ = curve_fit(stretched_exp, params, errors,
                           p0=[1.0, 1e-3, 0.3],
                           bounds=([0.01, 1e-10, 0.01], [10.0, 1.0, 1.0]),
                           maxfev=10000)
        C, b, c = popt
        predicted = stretched_exp(params, C, b, c)
        ss_res = np.sum((errors - predicted) ** 2)
        return {"name": "Stretched Exp.", "params": {"C": C, "b": b, "c": c},
                "k": 3, "predicted": predicted, "ss_res": ss_res}
    except (RuntimeError, ValueError):
        return None


def fit_bnsl(params, errors):
    """
    Fit broken neural scaling law (2-segment).
    Try multiple breakpoints and pick the best.
    """
    if not HAS_SCIPY or len(params) < 5:
        return None

    best_ss = float('inf')
    best_result = None

    # Try each internal point as a breakpoint
    for bp_idx in range(2, len(params) - 2):
        N_break = (params[bp_idx] + params[bp_idx + 1]) / 2

        try:
            # Fit segment 1
            mask1 = params < N_break
            mask2 = ~mask1
            if mask1.sum() < 2 or mask2.sum() < 2:
                continue

            log_N1 = np.log(params[mask1])
            log_E1 = np.log(errors[mask1])
            s1, i1 = np.polyfit(log_N1, log_E1, 1)
            C1, a1 = np.exp(i1), -s1

            log_N2 = np.log(params[mask2])
            log_E2 = np.log(errors[mask2])
            s2, i2 = np.polyfit(log_N2, log_E2, 1)
            C2, a2 = np.exp(i2), -s2

            predicted = bnsl_2segment(params, C1, a1, C2, a2, N_break)
            ss_res = np.sum((errors - predicted) ** 2)

            if ss_res < best_ss:
                best_ss = ss_res
                best_result = {
                    "name": "BNSL (2-seg)",
                    "params": {"C1": C1, "a1": a1, "C2": C2, "a2": a2, "N_break": N_break},
                    "k": 5,  # C1, a1, C2, a2, N_break
                    "predicted": predicted,
                    "ss_res": ss_res,
                }
        except (ValueError, np.linalg.LinAlgError):
            continue

    return best_result


# =============================================================================
# AIC / BIC
# =============================================================================

def compute_aic_bic(n, k, ss_res):
    """
    Compute AIC and BIC assuming Gaussian errors.
    n = number of data points
    k = number of model parameters
    ss_res = sum of squared residuals

    AIC = n * ln(ss_res/n) + 2k
    BIC = n * ln(ss_res/n) + k * ln(n)
    """
    if ss_res <= 0 or n <= k:
        return float('inf'), float('inf')

    log_likelihood_term = n * np.log(ss_res / n)
    aic = log_likelihood_term + 2 * k
    bic = log_likelihood_term + k * np.log(n)

    # AICc correction for small samples
    if n - k - 1 > 0:
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    else:
        aicc = float('inf')

    return aic, bic, aicc


# =============================================================================
# Load data
# =============================================================================

def load_scaling_data(dataset_name):
    """Load results, aggregate by arch/size, return (params, error_rate) arrays per arch."""
    results_dir = BASE_DIR / "results" / dataset_name
    if not results_dir.exists():
        return {}

    by_arch = defaultdict(lambda: defaultdict(list))

    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.stem.endswith("_summary") or json_file.stem.endswith("_results"):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            arch = data.get("arch")
            if arch:
                by_arch[arch][data["size_param"]].append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    result = {}
    for arch, sizes in by_arch.items():
        params_list = []
        errors_list = []
        for sp in sorted(sizes.keys()):
            runs = sizes[sp]
            params_list.append(runs[0]["num_params"])
            errors_list.append(np.mean([1.0 - r["top1_acc"] for r in runs]))
        result[arch] = (np.array(params_list, dtype=float), np.array(errors_list, dtype=float))

    return result


# =============================================================================
# Main analysis
# =============================================================================

def analyze_dataset(dataset_name):
    """Run model selection for one dataset."""
    print(f"\n{'='*70}")
    print(f"  Model Selection: {dataset_name}")
    print(f"{'='*70}")

    data = load_scaling_data(dataset_name)
    if not data:
        print(f"  No data found for {dataset_name}")
        return {}

    all_comparisons = {}

    for arch, (params, errors) in sorted(data.items()):
        label = ARCH_LABELS.get(arch, arch)
        n = len(params)

        print(f"\n  --- {label} ({n} sizes) ---")

        # Fit all candidates
        fits = []
        fit_pl = fit_power_law(params, errors)
        if fit_pl:
            fits.append(fit_pl)
        fit_ll = fit_log_linear(params, errors)
        if fit_ll:
            fits.append(fit_ll)
        fit_se = fit_stretched_exp(params, errors)
        if fit_se:
            fits.append(fit_se)
        fit_bn = fit_bnsl(params, errors)
        if fit_bn:
            fits.append(fit_bn)

        print(f"\n  {'Model':<20} {'k':>3} {'SS_res':>12} {'AIC':>10} {'BIC':>10} {'AICc':>10}")
        print(f"  {'-'*68}")

        comparison = []
        for fit in fits:
            aic, bic, aicc = compute_aic_bic(n, fit["k"], fit["ss_res"])
            entry = {
                "model": fit["name"],
                "k": fit["k"],
                "ss_res": float(fit["ss_res"]),
                "aic": float(aic),
                "bic": float(bic),
                "aicc": float(aicc),
                "params": {k: float(v) if isinstance(v, (np.floating, float)) else v
                          for k, v in fit["params"].items()},
            }
            comparison.append(entry)
            print(f"  {fit['name']:<20} {fit['k']:>3} {fit['ss_res']:>12.6f} "
                  f"{aic:>10.2f} {bic:>10.2f} {aicc:>10.2f}")

        # Identify winners
        if comparison:
            aic_winner = min(comparison, key=lambda x: x["aic"])
            bic_winner = min(comparison, key=lambda x: x["bic"])
            print(f"\n  AIC selects: {aic_winner['model']}")
            print(f"  BIC selects: {bic_winner['model']}")

            # Delta AIC relative to winner
            min_aic = aic_winner["aic"]
            print(f"\n  Delta AIC (vs best):")
            for c in comparison:
                delta = c["aic"] - min_aic
                strength = "strong" if delta > 10 else "some" if delta > 2 else "minimal"
                print(f"    {c['model']:<20} delta={delta:>8.2f}  ({strength} evidence against)")

        all_comparisons[arch] = comparison

    return all_comparisons


# =============================================================================
# Plotting
# =============================================================================

def plot_model_fits(dataset_name, all_comparisons, save_path):
    """Plot data with all candidate model fits overlaid."""
    if not HAS_MPL:
        return

    data = load_scaling_data(dataset_name)
    n_archs = len(data)
    if n_archs == 0:
        return

    fig, axes = plt.subplots(1, n_archs, figsize=(7 * n_archs, 5))
    if n_archs == 1:
        axes = [axes]

    line_styles = {"Power Law": "-", "Log-Linear": "--", "Stretched Exp.": "-.", "BNSL (2-seg)": ":"}
    fit_colors = {"Power Law": "#2563eb", "Log-Linear": "#dc2626",
                  "Stretched Exp.": "#16a34a", "BNSL (2-seg)": "#9333ea"}

    for ax, (arch, (params, errors)) in zip(axes, sorted(data.items())):
        label = ARCH_LABELS.get(arch, arch)
        ax.scatter(params, errors, color="black", s=50, zorder=5, label="Data")

        comparisons = all_comparisons.get(arch, [])
        x_fit = np.logspace(np.log10(params.min() * 0.8), np.log10(params.max() * 1.2), 200)

        for comp in comparisons:
            name = comp["model"]
            p = comp["params"]
            ls = line_styles.get(name, "-")
            color = fit_colors.get(name, "gray")

            try:
                if name == "Power Law":
                    y_fit = power_law(x_fit, p["C"], p["alpha"])
                elif name == "Log-Linear":
                    y_fit = log_linear(x_fit, p["a"], p["b"])
                elif name == "Stretched Exp.":
                    y_fit = stretched_exp(x_fit, p["C"], p["b"], p["c"])
                elif name == "BNSL (2-seg)":
                    y_fit = bnsl_2segment(x_fit, p["C1"], p["a1"], p["C2"], p["a2"], p["N_break"])
                else:
                    continue

                aic_str = f"AIC={comp['aic']:.1f}"
                ax.plot(x_fit, y_fit, ls=ls, color=color, linewidth=1.5, alpha=0.7,
                       label=f"{name} ({aic_str})")
            except Exception:
                continue

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parameters ($N$)")
        ax.set_ylabel("Error Rate")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    plt.suptitle(f"Model Selection — {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Model selection figure saved: {save_path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AIC/BIC model selection for scaling laws")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar100", "cifar10", "tinyimagenet", "speechcommands"])
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else ["cifar100", "cifar10", "tinyimagenet", "speechcommands"]

    all_dataset_results = {}

    for dataset_name in datasets:
        comparisons = analyze_dataset(dataset_name)
        if comparisons:
            all_dataset_results[dataset_name] = comparisons

            # Save
            output_dir = BASE_DIR / "results" / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "model_selection_results.json", "w") as f:
                json.dump(comparisons, f, indent=2)

            # Plot
            plot_model_fits(dataset_name, comparisons,
                          FIGURES_DIR / f"model_selection_{dataset_name}.pdf")

    # Summary table across all datasets
    if all_dataset_results:
        print(f"\n{'='*70}")
        print("  SUMMARY: AIC/BIC Winners Across Datasets")
        print(f"{'='*70}")
        print(f"  {'Dataset':<18} {'Architecture':<18} {'AIC Winner':<20} {'BIC Winner':<20}")
        print(f"  {'-'*76}")

        for ds, comparisons in all_dataset_results.items():
            for arch, comps in comparisons.items():
                label = ARCH_LABELS.get(arch, arch)
                if comps:
                    aic_winner = min(comps, key=lambda x: x["aic"])["model"]
                    bic_winner = min(comps, key=lambda x: x["bic"])["model"]
                else:
                    aic_winner = bic_winner = "N/A"
                print(f"  {ds:<18} {label:<18} {aic_winner:<20} {bic_winner:<20}")


if __name__ == "__main__":
    main()
