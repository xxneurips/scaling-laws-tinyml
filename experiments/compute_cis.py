"""
Compute bootstrap CIs on scaling exponents and cross-seed Jaccard
from existing phase1 results (90 JSON files).
"""
import json
import glob
import numpy as np
from scipy import stats
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "phase1")

# --- ScaleCNN configs ---
CNN_CHANNELS = [4, 8, 12, 16, 24, 32, 48, 64]
CNN_PARAMS = {4: 21936, 8: 80348, 12: 175336, 16: 306900,
              24: 679756, 32: 1198916, 48: 2676148, 64: 4738596}

# --- MobileNetV2 configs ---
MOB_WIDTHS = [0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
MOB_PARAMS = {0.10: 214180, 0.15: 262596, 0.25: 366212, 0.35: 524228,
              0.50: 815780, 0.75: 1483524, 1.00: 2351972, 1.50: 5129252,
              2.00: 8953188, 3.00: 19803748}

SEEDS = [0, 1, 2, 3, 4]


def load_result(arch, config, seed):
    if arch == "cnn":
        fname = f"cnn_c{config:03d}_s{seed}.json"
    else:
        fname = f"mob_w{config:.2f}_s{seed}.json"
    path = os.path.join(RESULTS_DIR, fname)
    with open(path) as f:
        return json.load(f)


def fit_power_law(params_list, error_rates):
    """Fit log(error) = -alpha * log(N) + log(C), return alpha, R2."""
    log_n = np.log(params_list)
    log_e = np.log(error_rates)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_e)
    return -slope, r_value**2, std_err


def compute_jaccard(mask_a, mask_b):
    """Jaccard index between two binary error masks."""
    a = np.array(mask_a, dtype=bool)
    b = np.array(mask_b, dtype=bool)
    # Error masks: True = incorrect
    err_a = ~a
    err_b = ~b
    intersection = np.sum(err_a & err_b)
    union = np.sum(err_a | err_b)
    if union == 0:
        return 1.0
    return intersection / union


# ====================================================================
# 1. Per-seed power law fits
# ====================================================================
print("=" * 70)
print("1. PER-SEED POWER LAW FITS")
print("=" * 70)

# ScaleCNN
cnn_alphas = []
for seed in SEEDS:
    params_list = []
    errors = []
    for c in CNN_CHANNELS:
        r = load_result("cnn", c, seed)
        params_list.append(CNN_PARAMS[c])
        errors.append(1.0 - r["top1_acc"])
    alpha, r2, se = fit_power_law(np.array(params_list), np.array(errors))
    cnn_alphas.append(alpha)
    print(f"  ScaleCNN seed={seed}: alpha={alpha:.4f}, R2={r2:.4f}")

print(f"\n  ScaleCNN: alpha = {np.mean(cnn_alphas):.4f} +/- {np.std(cnn_alphas, ddof=1):.4f}")
print(f"  95% CI: [{np.mean(cnn_alphas) - 1.96*np.std(cnn_alphas, ddof=1)/np.sqrt(5):.4f}, "
      f"{np.mean(cnn_alphas) + 1.96*np.std(cnn_alphas, ddof=1)/np.sqrt(5):.4f}]")

# MobileNetV2
mob_alphas = []
for seed in SEEDS:
    params_list = []
    errors = []
    for w in MOB_WIDTHS:
        r = load_result("mob", w, seed)
        params_list.append(MOB_PARAMS[w])
        errors.append(1.0 - r["top1_acc"])
    alpha, r2, se = fit_power_law(np.array(params_list), np.array(errors))
    mob_alphas.append(alpha)
    print(f"  MobNetV2 seed={seed}: alpha={alpha:.4f}, R2={r2:.4f}")

print(f"\n  MobNetV2: alpha = {np.mean(mob_alphas):.4f} +/- {np.std(mob_alphas, ddof=1):.4f}")
print(f"  95% CI: [{np.mean(mob_alphas) - 1.96*np.std(mob_alphas, ddof=1)/np.sqrt(5):.4f}, "
      f"{np.mean(mob_alphas) + 1.96*np.std(mob_alphas, ddof=1)/np.sqrt(5):.4f}]")

# Is the difference significant?
t_stat, p_val = stats.ttest_ind(cnn_alphas, mob_alphas)
print(f"\n  Difference significance: t={t_stat:.3f}, p={p_val:.6f}")
print(f"  Delta alpha = {np.mean(cnn_alphas) - np.mean(mob_alphas):.4f}")

# ====================================================================
# 2. Cross-seed Jaccard (same config, different seeds) — stability check
# ====================================================================
print("\n" + "=" * 70)
print("2. CROSS-SEED JACCARD (same config, different seeds)")
print("=" * 70)

for arch, config, label in [("cnn", 4, "ScaleCNN c=4"),
                             ("cnn", 64, "ScaleCNN c=64"),
                             ("mob", 0.10, "MobNet m=0.10"),
                             ("mob", 3.00, "MobNet m=3.00")]:
    jaccards = []
    for s1 in SEEDS:
        for s2 in SEEDS:
            if s1 >= s2:
                continue
            r1 = load_result(arch, config, s1)
            r2 = load_result(arch, config, s2)
            j = compute_jaccard(r1["correct_mask"], r2["correct_mask"])
            jaccards.append(j)
    print(f"  {label}: Jaccard = {np.mean(jaccards):.3f} +/- {np.std(jaccards):.3f} "
          f"(range: {np.min(jaccards):.3f}-{np.max(jaccards):.3f}, n={len(jaccards)} pairs)")

# ====================================================================
# 3. Multi-seed Jaccard: smallest vs largest (all seed combos)
# ====================================================================
print("\n" + "=" * 70)
print("3. MULTI-SEED JACCARD: smallest vs largest")
print("=" * 70)

# ScaleCNN c=4 vs c=64: all 25 seed combinations
cnn_jaccards = []
for s1 in SEEDS:
    for s2 in SEEDS:
        r_small = load_result("cnn", 4, s1)
        r_large = load_result("cnn", 64, s2)
        j = compute_jaccard(r_small["correct_mask"], r_large["correct_mask"])
        cnn_jaccards.append(j)

print(f"  ScaleCNN c=4 vs c=64 (25 pairs):")
print(f"    Jaccard = {np.mean(cnn_jaccards):.3f} +/- {np.std(cnn_jaccards):.3f}")
print(f"    Range: [{np.min(cnn_jaccards):.3f}, {np.max(cnn_jaccards):.3f}]")
print(f"    Seed-matched (diagonal): ", end="")
diag = [compute_jaccard(load_result("cnn", 4, s)["correct_mask"],
                        load_result("cnn", 64, s)["correct_mask"]) for s in SEEDS]
print(f"{np.mean(diag):.3f} +/- {np.std(diag):.3f}")

# MobNet m=0.10 vs m=3.00
mob_jaccards = []
for s1 in SEEDS:
    for s2 in SEEDS:
        r_small = load_result("mob", 0.10, s1)
        r_large = load_result("mob", 3.00, s2)
        j = compute_jaccard(r_small["correct_mask"], r_large["correct_mask"])
        mob_jaccards.append(j)

print(f"\n  MobNetV2 m=0.10 vs m=3.00 (25 pairs):")
print(f"    Jaccard = {np.mean(mob_jaccards):.3f} +/- {np.std(mob_jaccards):.3f}")
print(f"    Range: [{np.min(mob_jaccards):.3f}, {np.max(mob_jaccards):.3f}]")

# ====================================================================
# 4. Mean confidence for smallest models (calibration mechanism check)
# ====================================================================
print("\n" + "=" * 70)
print("4. MEAN CONFIDENCE (calibration mechanism)")
print("=" * 70)

for arch, config, label in [("cnn", 4, "ScaleCNN c=4"),
                             ("cnn", 32, "ScaleCNN c=32"),
                             ("cnn", 64, "ScaleCNN c=64"),
                             ("mob", 0.10, "MobNet m=0.10"),
                             ("mob", 3.00, "MobNet m=3.00")]:
    confs = []
    for seed in SEEDS:
        r = load_result(arch, config, seed)
        if "mean_confidence" in r:
            confs.append(r["mean_confidence"])
        elif "confidences" in r:
            confs.append(np.mean(r["confidences"]))
        elif "predictions" in r:
            # predictions are typically softmax outputs or predicted class probs
            preds = np.array(r["predictions"])
            if preds.ndim == 2:
                confs.append(np.mean(np.max(preds, axis=1)))
            else:
                confs.append(None)
        else:
            confs.append(None)
    valid = [c for c in confs if c is not None]
    if valid:
        print(f"  {label}: mean confidence = {np.mean(valid):.4f} +/- {np.std(valid):.4f}")
    else:
        print(f"  {label}: no confidence data found")
