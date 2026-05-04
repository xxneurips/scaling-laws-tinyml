"""
R5 NeurIPS — Synthesize all γ-driver ablation results into a decomposition table.

Pulls:
  - results/class_count/summary.json (γ vs class count)
  - results/sample_count/summary.json (γ vs sample count)
  - results/bandpass_gamma/summary.json (γ vs β manipulation = spectral)
  - results/deit/summary.json (γ for non-CNN architecture)
  - results/{cifar10,cifar100,tinyimagenet}/act_spectrum_stability.json

Produces:
  - paper-ready table fragments
  - regression of γ vs (log classes, log samples, β)
"""
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
RES = BASE / "results"


def load(p):
    return json.load(open(p)) if p.exists() else None


def fit_line(xs, ys):
    if len(xs) < 2:
        return None
    coef = np.polyfit(xs, ys, 1)
    pred = np.polyval(coef, xs)
    r2 = 1 - np.sum((np.array(ys) - pred) ** 2) / max(1e-12, np.sum((np.array(ys) - np.mean(ys)) ** 2))
    return float(coef[0]), float(coef[1]), float(r2)


def main():
    print("=" * 78)
    print("R5 SYNTHESIS — γ-drivers decomposition")
    print("=" * 78)

    # 1) γ vs class count (CIFAR-100 subsets)
    cc = load(RES / "class_count" / "summary.json")
    if cc and cc.get("fits"):
        print("\n[CLASS COUNT] CIFAR-100 → {10, 25, 50, 100} classes:")
        print(f"  {'n_classes':>10} {'gamma':>8} {'R2':>6}")
        ncs, gammas = [], []
        for f in cc["fits"]:
            print(f"  {f['n_classes']:>10} {f['gamma']:>8.4f} {f['r2']:>6.3f}")
            ncs.append(math.log(f["n_classes"]))
            gammas.append(f["gamma"])
        fit = fit_line(ncs, gammas)
        if fit:
            slope, intercept, r2 = fit
            print(f"\n  γ ~ {slope:.4f} · log(n_classes) + {intercept:.4f}   (R²={r2:.3f})")
            print(f"  Δγ over 10×→100× class change: {slope * math.log(10):.4f}")

    # 2) γ vs sample count
    sc = load(RES / "sample_count" / "summary.json")
    if sc and sc.get("fits"):
        print("\n[SAMPLE COUNT] CIFAR-100 train subsets:")
        print(f"  {'n_samples':>10} {'gamma':>8} {'R2':>6}")
        ns, gammas = [], []
        for f in sc["fits"]:
            print(f"  {f['n_samples']:>10} {f['gamma']:>8.4f} {f['r2']:>6.3f}")
            ns.append(math.log(f["n_samples"]))
            gammas.append(f["gamma"])
        fit = fit_line(ns, gammas)
        if fit:
            slope, intercept, r2 = fit
            print(f"\n  γ ~ {slope:.4f} · log(n_samples) + {intercept:.4f}   (R²={r2:.3f})")

    # 3) γ vs β (bandpass = controlled spectral manipulation)
    bp = load(RES / "bandpass_gamma" / "summary.json")
    if bp and bp.get("fits"):
        print("\n[β MANIPULATION] CIFAR-100 bandpass variants:")
        print(f"  {'β':>6} {'gamma':>8} {'R2':>6}")
        bs, gammas = [], []
        for f in bp["fits"]:
            print(f"  {f['beta_actual']:>6.3f} {f['gamma_variant']:>8.4f} {f['r2']:>6.3f}")
            bs.append(f["beta_actual"])
            gammas.append(f["gamma_variant"])
        fit = fit_line(bs, gammas)
        if fit:
            slope, intercept, r2 = fit
            print(f"\n  γ ~ {slope:.4f} · β + {intercept:.4f}   (R²={r2:.3f})")

    # 4) DeiT γ (non-CNN comparator)
    deit = load(RES / "deit" / "summary.json")
    if deit and deit.get("fits_gamma"):
        print("\n[DEIT-TINY γ] cross-dataset comparator:")
        print(f"  {'dataset':<14} {'gamma_vit':>9} {'R2':>6}")
        for f in deit["fits_gamma"]:
            print(f"  {f['dataset']:<14} {f['gamma']:>9.4f} {f['r2']:>6.3f}")
    if deit and deit.get("fits_alpha"):
        print("\n[DEIT-TINY α] cross-dataset comparator:")
        print(f"  {'dataset':<14} {'alpha_vit':>9} {'R2':>6}")
        for f in deit["fits_alpha"]:
            print(f"  {f['dataset']:<14} {f['alpha']:>9.4f} {f['r2']:>6.3f}")

    # 5) Decomposition summary
    print("\n" + "=" * 78)
    print("DECOMPOSITION VERDICT")
    print("=" * 78)
    if cc and sc and bp:
        cc_swing = max(f["gamma"] for f in cc["fits"]) - min(f["gamma"] for f in cc["fits"])
        sc_swing = max(f["gamma"] for f in sc["fits"]) - min(f["gamma"] for f in sc["fits"])
        bp_swing = max(f["gamma_variant"] for f in bp["fits"]) - min(f["gamma_variant"] for f in bp["fits"])
        cross_dataset_swing = 0.191 - 0.114  # CIFAR-10 → Tiny ImageNet from main paper
        print(f"  γ swing across class-count   : {cc_swing:.4f}")
        print(f"  γ swing across sample-count  : {sc_swing:.4f}")
        print(f"  γ swing across β (spectral)  : {bp_swing:.4f}")
        print(f"  γ swing across datasets      : {cross_dataset_swing:.4f}  (reference)")
        attributable = cc_swing + sc_swing
        spectral_frac = bp_swing / cross_dataset_swing if cross_dataset_swing > 0 else 0
        struct_frac = attributable / cross_dataset_swing if cross_dataset_swing > 0 else 0
        print(f"\n  Spectral fraction of γ variation:    {spectral_frac:.1%}")
        print(f"  Structural (cls + smp) fraction:     {struct_frac:.1%}")

    print()


if __name__ == "__main__":
    main()
