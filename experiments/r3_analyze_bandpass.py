"""
R3 NeurIPS — Analyze bandpass sweep results.

For each beta variant, fit alpha from the width-sweep. Then plot alpha(beta)
and compare to the theoretical slope gamma(beta-1).

Output:
  - results/bandpass_cifar100/alpha_vs_beta.json
  - paper/neurips2026/figures/fig_alpha_vs_beta.pdf
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASE = Path(__file__).parent
RESULTS = BASE / "results" / "bandpass_cifar100"
FIG_DIR = BASE.parent / "paper" / "neurips2026" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Measured gamma from CIFAR-10 stable rank
GAMMA_CNN_MEASURED = 0.114
# Original-CIFAR-100 alpha at beta=1.452
ALPHA_ORIG = 0.156


def load_runs():
    by_variant = defaultdict(list)
    for p in RESULTS.glob("bp_*.json"):
        with open(p) as f:
            d = json.load(f)
        by_variant[d["beta_tag"]].append(d)
    return by_variant


def fit_alpha(runs):
    """Fit error-rate power law across widths. Returns (alpha, r2)."""
    by_width = defaultdict(list)
    for r in runs:
        by_width[r["width"]].append(r)
    widths = sorted(by_width.keys())
    if len(widths) < 3:
        return None, None
    params = []
    errs = []
    for w in widths:
        rs = by_width[w]
        params.append(rs[0]["num_params"])
        mean_err = np.mean([1.0 - r["top1_acc"] for r in rs])
        errs.append(mean_err)
    lp = np.log(params)
    le = np.log(errs)
    c = np.polyfit(lp, le, 1)
    alpha = -c[0]
    pred = np.polyval(c, lp)
    r2 = 1 - np.sum((le - pred)**2) / np.sum((le - le.mean())**2)
    return float(alpha), float(r2)


def main():
    by_variant = load_runs()
    if not by_variant:
        print("No bandpass runs found yet.")
        return

    # Include original CIFAR-100 as a calibration point
    variants_data = []
    for tag, runs in by_variant.items():
        alpha, r2 = fit_alpha(runs)
        beta_actual = runs[0]["beta_actual"]
        n_runs = len(runs)
        n_widths = len(set(r["width"] for r in runs))
        print(f"  {tag}: beta={beta_actual:.3f}, alpha={alpha}, R^2={r2}, "
              f"{n_runs} runs, {n_widths} widths")
        if alpha is not None:
            variants_data.append({
                "tag": tag, "beta": beta_actual,
                "alpha": alpha, "r2": r2,
                "n_runs": n_runs, "n_widths": n_widths,
            })

    # Add the original CIFAR-100 point for reference
    variants_data.append({
        "tag": "original", "beta": 1.452,
        "alpha": ALPHA_ORIG, "r2": 0.965,
        "n_runs": 40, "n_widths": 8,
    })
    variants_data.sort(key=lambda x: x["beta"])

    # Save table
    out = {
        "variants": variants_data,
        "theory_slope_measured_gamma": GAMMA_CNN_MEASURED,
    }
    with open(RESULTS / "alpha_vs_beta.json", "w") as f:
        json.dump(out, f, indent=2)

    # Fit alpha = slope * (beta - 1) to measured points
    betas = np.array([v["beta"] for v in variants_data])
    alphas = np.array([v["alpha"] for v in variants_data])
    if len(betas) >= 2:
        # Fit through origin at beta=1: alpha = slope * (beta - 1)
        x = betas - 1
        slope_through = np.sum(alphas * x) / np.sum(x * x)

        # Also fit alpha = intercept + slope * (beta-1)
        coeffs = np.polyfit(x, alphas, 1)
        slope_free, intercept_free = coeffs[0], coeffs[1]

        # R^2 for fits
        pred_t = slope_through * x
        r2_t = 1 - np.sum((alphas - pred_t)**2) / np.sum((alphas - alphas.mean())**2)
        pred_f = slope_free * x + intercept_free
        r2_f = 1 - np.sum((alphas - pred_f)**2) / np.sum((alphas - alphas.mean())**2)

        print()
        print("=" * 60)
        print("alpha vs (beta - 1)  summary")
        print("=" * 60)
        print(f"  Through-origin fit:  slope = {slope_through:.3f}, R^2 = {r2_t:.3f}")
        print(f"  Free-intercept fit:  slope = {slope_free:.3f}, int = {intercept_free:.3f}, R^2 = {r2_f:.3f}")
        print(f"  Measured gamma (CIFAR-10 stable rank):  {GAMMA_CNN_MEASURED:.3f}")
        print(f"  Theory prediction slope = gamma:         {GAMMA_CNN_MEASURED:.3f}")
        print(f"  Observed / theory ratio (through-origin): {slope_through/GAMMA_CNN_MEASURED:.2f}x")

    # Plot
    if HAS_MPL and len(variants_data) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(betas, alphas, s=70, c="#1a5276", zorder=3,
                   label="Measured $\\alpha(\\beta)$")
        for v in variants_data:
            ax.annotate(v["tag"], (v["beta"], v["alpha"]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=9, color="#555")

        # Theory: alpha = gamma * (beta - 1)
        b_line = np.linspace(1.0, 2.2, 100)
        ax.plot(b_line, GAMMA_CNN_MEASURED * (b_line - 1),
                "--", color="#e87422", alpha=0.7,
                label=f"Theory: $\\gamma_{{\\mathrm{{measured}}}} (\\beta - 1)$ (slope {GAMMA_CNN_MEASURED:.2f})")

        # Empirical through-origin fit
        if len(betas) >= 2:
            ax.plot(b_line, slope_through * (b_line - 1),
                    "-", color="#196f3d", alpha=0.8,
                    label=f"Empirical: slope {slope_through:.2f}, $R^2={r2_t:.2f}$")

        ax.set_xlabel(r"Data spectral decay $\beta$")
        ax.set_ylabel(r"Scaling exponent $\alpha$")
        ax.set_title(r"ScaleCNN on CIFAR-100 bandpass variants: $\alpha(\beta)$")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_fig = FIG_DIR / "fig_alpha_vs_beta.pdf"
        plt.savefig(out_fig, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nSaved figure: {out_fig}")


if __name__ == "__main__":
    main()
