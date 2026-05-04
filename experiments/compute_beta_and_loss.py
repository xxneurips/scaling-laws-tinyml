"""
Compute:
1. Beta (spectral decay exponent) from CIFAR-100 eigenspectrum
2. Training loss power law fits for comparison with error rate fits
"""
import json
import numpy as np
from scipy import stats
import os

# ====================================================================
# 1. CIFAR-100 EIGENSPECTRUM -> beta
# ====================================================================
print("=" * 70)
print("1. CIFAR-100 EIGENSPECTRUM")
print("=" * 70)

try:
    import torchvision
    import torchvision.transforms as transforms

    # Load CIFAR-100 training set (no augmentation, just normalize to [0,1])
    dataset = torchvision.datasets.CIFAR100(
        root=os.path.join(os.path.dirname(__file__), "data"),
        train=True, download=True,
        transform=transforms.ToTensor()
    )

    # Flatten all images: 50000 x 3072
    print("  Loading and flattening 50K images...")
    X = np.zeros((len(dataset), 3072), dtype=np.float32)
    for i in range(len(dataset)):
        img, _ = dataset[i]
        X[i] = img.numpy().reshape(-1)

    # Center the data
    X_centered = X - X.mean(axis=0)

    # Compute covariance eigenvalues
    print("  Computing covariance matrix (3072 x 3072)...")
    cov = np.cov(X_centered, rowvar=False)
    print("  Computing eigenvalues...")
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Remove near-zero eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    print(f"  {len(eigenvalues)} non-zero eigenvalues")
    print(f"  Top 10: {eigenvalues[:10]}")
    print(f"  Explained variance (top 100): {eigenvalues[:100].sum() / eigenvalues.sum():.3f}")

    # Fit power law: log(lambda_k) = -beta * log(k) + const
    # Use eigenvalues 10 to 1000 (skip the very top which may not follow power law)
    k_range = np.arange(10, min(1000, len(eigenvalues)))
    log_k = np.log(k_range)
    log_lambda = np.log(eigenvalues[k_range])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_lambda)
    beta = -slope

    print(f"\n  Power law fit (k=10 to {min(1000, len(eigenvalues))}):")
    print(f"    beta = {beta:.3f} +/- {std_err:.3f}")
    print(f"    R2 = {r_value**2:.4f}")

    # Also fit a narrower range (k=10 to 500)
    k_range2 = np.arange(10, min(500, len(eigenvalues)))
    slope2, _, r2, _, se2 = stats.linregress(np.log(k_range2), np.log(eigenvalues[k_range2]))
    print(f"\n  Power law fit (k=10 to {min(500, len(eigenvalues))}):")
    print(f"    beta = {-slope2:.3f} +/- {se2:.3f}")
    print(f"    R2 = {r2**2:.4f}")

    # Wider range
    k_range3 = np.arange(5, min(2000, len(eigenvalues)))
    slope3, _, r3, _, se3 = stats.linregress(np.log(k_range3), np.log(eigenvalues[k_range3]))
    print(f"\n  Power law fit (k=5 to {min(2000, len(eigenvalues))}):")
    print(f"    beta = {-slope3:.3f} +/- {se3:.3f}")
    print(f"    R2 = {r3**2:.4f}")

    # Predicted alpha from theory
    gamma = 0.5
    alpha_predicted = gamma * (beta - 1)
    print(f"\n  Predicted scaling exponent:")
    print(f"    alpha = gamma*(beta-1) = 0.5 * ({beta:.3f} - 1) = {alpha_predicted:.4f}")
    print(f"    Measured: alpha_CNN = 0.156, alpha_MobNet = 0.106")

except ImportError:
    print("  torchvision not available, skipping eigenspectrum computation")
except Exception as e:
    print(f"  Error: {e}")

# ====================================================================
# 2. TRAINING LOSS POWER LAW FITS
# ====================================================================
print("\n" + "=" * 70)
print("2. TRAINING LOSS POWER LAW FITS")
print("=" * 70)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "phase1")
SEEDS = [0, 1, 2, 3, 4]

CNN_CHANNELS = [4, 8, 12, 16, 24, 32, 48, 64]
CNN_PARAMS = {4: 21936, 8: 80348, 12: 175336, 16: 306900,
              24: 679756, 32: 1198916, 48: 2676148, 64: 4738596}

MOB_WIDTHS = [0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
MOB_PARAMS = {0.10: 214180, 0.15: 262596, 0.25: 366212, 0.35: 524228,
              0.50: 815780, 0.75: 1483524, 1.00: 2351972, 1.50: 5129252,
              2.00: 8953188, 3.00: 19803748}


def load_result(arch, config, seed):
    if arch == "cnn":
        fname = f"cnn_c{config:03d}_s{seed}.json"
    else:
        fname = f"mob_w{config:.2f}_s{seed}.json"
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        return json.load(f)


# ScaleCNN: training loss
print("\n  ScaleCNN training loss by config:")
for c in CNN_CHANNELS:
    losses = [load_result("cnn", c, s)["final_train_loss"] for s in SEEDS]
    errors = [1 - load_result("cnn", c, s)["top1_acc"] for s in SEEDS]
    print(f"    c={c:3d}: params={CNN_PARAMS[c]:>10,d}  "
          f"train_loss={np.mean(losses):.4f}+/-{np.std(losses):.4f}  "
          f"test_error={np.mean(errors):.4f}+/-{np.std(errors):.4f}")

# Per-seed training loss power law fits
print("\n  Per-seed training loss power law fits:")
cnn_loss_alphas = []
for seed in SEEDS:
    params_list = np.array([CNN_PARAMS[c] for c in CNN_CHANNELS])
    losses = np.array([load_result("cnn", c, seed)["final_train_loss"] for c in CNN_CHANNELS])
    log_n = np.log(params_list)
    log_l = np.log(losses)
    slope, intercept, r_value, _, se = stats.linregress(log_n, log_l)
    alpha = -slope
    cnn_loss_alphas.append(alpha)
    print(f"    ScaleCNN seed={seed}: alpha_loss={alpha:.4f}, R2={r_value**2:.4f}")

print(f"\n  ScaleCNN training loss: alpha = {np.mean(cnn_loss_alphas):.4f} +/- {np.std(cnn_loss_alphas, ddof=1):.4f}")

mob_loss_alphas = []
for seed in SEEDS:
    params_list = np.array([MOB_PARAMS[w] for w in MOB_WIDTHS])
    losses = np.array([load_result("mob", w, seed)["final_train_loss"] for w in MOB_WIDTHS])
    log_n = np.log(params_list)
    log_l = np.log(losses)
    slope, intercept, r_value, _, se = stats.linregress(log_n, log_l)
    alpha = -slope
    mob_loss_alphas.append(alpha)
    print(f"    MobNetV2 seed={seed}: alpha_loss={alpha:.4f}, R2={r_value**2:.4f}")

print(f"\n  MobNetV2 training loss: alpha = {np.mean(mob_loss_alphas):.4f} +/- {np.std(mob_loss_alphas, ddof=1):.4f}")

print(f"\n  Comparison:")
print(f"    ScaleCNN:  alpha_error = 0.156 +/- 0.002,  alpha_train_loss = {np.mean(cnn_loss_alphas):.3f} +/- {np.std(cnn_loss_alphas, ddof=1):.3f}")
print(f"    MobNetV2:  alpha_error = 0.106 +/- 0.001,  alpha_train_loss = {np.mean(mob_loss_alphas):.3f} +/- {np.std(mob_loss_alphas, ddof=1):.3f}")
