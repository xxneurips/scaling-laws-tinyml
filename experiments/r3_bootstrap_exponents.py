"""
R3 (NeurIPS prep) — Bootstrap 95% CIs on all fitted scaling exponents.

The prior reference reports point estimates only (alpha = 0.156, 0.106) without error bars
on the slope. we identify this as a statistical rigor gap.

Here: 10,000-resample bootstrap over the (size, seed) pairs. Produces 95% CIs
and bootstrap hypothesis test for alpha_CNN != alpha_Mob.

Output:
  - results/cifar100/bootstrap_exponents.json
  - printed table for inclusion in NeurIPS draft
"""
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

import sys
BASE = Path(__file__).parent
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cifar100"
RESULTS = BASE / "results" / DATASET
print(f"\n[DATASET={DATASET}]\n")

N_BOOT = 10000
rng = np.random.default_rng(2026)


def load_by_config():
    """Return dict: (arch, size_param) -> list of run dicts."""
    by_cfg = defaultdict(list)
    for fn in sorted(os.listdir(RESULTS)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(RESULTS / fn) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or d.get("arch") is None:
            continue
        by_cfg[(d["arch"], d["size_param"])].append(d)
    return by_cfg


def fit_power_law(params, metrics):
    """Fit log(metric) = -alpha * log(N) + c.  Returns alpha, intercept, R^2."""
    lp = np.log(np.asarray(params, dtype=float))
    lm = np.log(np.asarray(metrics, dtype=float))
    coeffs = np.polyfit(lp, lm, 1)
    alpha = -coeffs[0]
    pred = np.polyval(coeffs, lp)
    ss_res = np.sum((lm - pred) ** 2)
    ss_tot = np.sum((lm - lm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return alpha, coeffs[1], r2


def bootstrap_alpha(runs_by_size, metric_fn, n_boot=N_BOOT):
    """Stratified bootstrap: resample seeds WITHIN each size, refit.
       runs_by_size: dict size_param -> list of run dicts.
    """
    sizes = sorted(runs_by_size.keys())
    params = [runs_by_size[s][0]["num_params"] for s in sizes]

    # Point estimate from mean over seeds
    means = [np.mean([metric_fn(r) for r in runs_by_size[s]]) for s in sizes]
    alpha_hat, _, r2_hat = fit_power_law(params, means)

    # Bootstrap
    alphas_boot = np.empty(n_boot)
    for b in range(n_boot):
        boot_means = []
        for s in sizes:
            runs = runs_by_size[s]
            idx = rng.integers(0, len(runs), size=len(runs))
            vals = [metric_fn(runs[i]) for i in idx]
            boot_means.append(np.mean(vals))
        alphas_boot[b], _, _ = fit_power_law(params, boot_means)

    ci_lo, ci_hi = np.percentile(alphas_boot, [2.5, 97.5])
    return {
        "alpha_hat": float(alpha_hat),
        "alpha_boot_mean": float(alphas_boot.mean()),
        "alpha_boot_std": float(alphas_boot.std()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "r2": float(r2_hat),
        "boot_samples": alphas_boot,  # for comparison test
    }


def error_rate(r):
    # top1_acc stored as fraction in [0,1]; return error rate as fraction
    acc = r["top1_acc"]
    if acc > 1.0:
        acc = acc / 100.0
    return 1.0 - acc


def train_loss_final(r):
    return r["train_losses"][-1]


def main():
    by_cfg = load_by_config()

    # Group by arch
    by_arch = defaultdict(lambda: defaultdict(list))
    for (arch, sp), runs in by_cfg.items():
        by_arch[arch][sp] = runs

    print("=" * 90)
    print("BOOTSTRAP 95% CIs ON FITTED SCALING EXPONENTS (10,000 resamples)")
    print("=" * 90)

    result = {}
    for arch in ["scalecnn", "mobilenet"]:
        print(f"\n{arch.upper()}")
        print("-" * 60)

        r_err = bootstrap_alpha(by_arch[arch], error_rate)
        r_tr = bootstrap_alpha(by_arch[arch], train_loss_final)

        print(f"  Error-rate exponent:   alpha = {r_err['alpha_hat']:.4f}  "
              f"[95% CI: {r_err['ci_lo']:.4f}, {r_err['ci_hi']:.4f}]  R^2 = {r_err['r2']:.4f}")
        print(f"  Train-loss exponent:   alpha = {r_tr['alpha_hat']:.4f}  "
              f"[95% CI: {r_tr['ci_lo']:.4f}, {r_tr['ci_hi']:.4f}]  R^2 = {r_tr['r2']:.4f}")

        # strip boot samples for JSON serialization
        result[arch] = {
            "error_rate": {k: v for k, v in r_err.items() if k != "boot_samples"},
            "train_loss": {k: v for k, v in r_tr.items() if k != "boot_samples"},
        }
        # but keep in a separate var for comparison test
        by_arch[arch]["_err_samples"] = r_err["boot_samples"]
        by_arch[arch]["_tr_samples"] = r_tr["boot_samples"]

    # Hypothesis test: alpha_CNN != alpha_Mob for error rate
    print("\n" + "=" * 90)
    print("HYPOTHESIS TEST: alpha_scaleCNN != alpha_mobilenet (error rate)")
    print("=" * 90)
    diff = by_arch["scalecnn"]["_err_samples"] - by_arch["mobilenet"]["_err_samples"]
    diff_mean = diff.mean()
    diff_ci = np.percentile(diff, [2.5, 97.5])
    p_under_null = (diff <= 0).mean() if diff_mean > 0 else (diff >= 0).mean()
    print(f"  alpha_CNN - alpha_Mob = {diff_mean:+.4f}  [95% CI: {diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]")
    print(f"  Bootstrap p-value (two-sided): {2*p_under_null:.6f}")
    print(f"  Fraction of resamples with sign reversal: {p_under_null:.6f}")
    if diff_ci[0] > 0 or diff_ci[1] < 0:
        print(f"  -> alpha gap is significant at 95% level (CI excludes 0)")
    else:
        print(f"  -> alpha gap NOT significant (CI includes 0)")

    result["comparison_err"] = {
        "diff_mean": float(diff_mean),
        "diff_ci_lo": float(diff_ci[0]),
        "diff_ci_hi": float(diff_ci[1]),
        "p_two_sided": float(2 * p_under_null),
    }

    out_path = RESULTS / f"bootstrap_exponents_{DATASET}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    # LaTeX-ready snippet
    print("\n" + "=" * 90)
    print("LATEX SNIPPET (drop into NeurIPS draft)")
    print("=" * 90)
    cnn_err = result["scalecnn"]["error_rate"]
    mob_err = result["mobilenet"]["error_rate"]
    print(r"\begin{equation}")
    print(r"    \alpha_{\mathrm{CNN}} = " +
          f"{cnn_err['alpha_hat']:.3f}\\ " +
          r"[" + f"{cnn_err['ci_lo']:.3f}" + r",\ " + f"{cnn_err['ci_hi']:.3f}" + r"],\quad "
          r"\alpha_{\mathrm{MobNet}} = " +
          f"{mob_err['alpha_hat']:.3f}\\ " +
          r"[" + f"{mob_err['ci_lo']:.3f}" + r",\ " + f"{mob_err['ci_hi']:.3f}" + r"]")
    print(r"\end{equation}")
    cmp = result["comparison_err"]
    sig = "p < 0.001" if cmp["p_two_sided"] < 0.001 else f"p = {cmp['p_two_sided']:.3f}"
    print(f"Gap: {cmp['diff_mean']:+.3f} [95% CI: {cmp['diff_ci_lo']:+.3f}, {cmp['diff_ci_hi']:+.3f}], {sig}")


if __name__ == "__main__":
    main()
