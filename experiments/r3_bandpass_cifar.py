"""
R3 NeurIPS — Bandpass-filtered CIFAR variants for predictive alpha(beta) test.

Goal: shift the spectral decay exponent beta of the data by applying a radial
frequency filter, then verify that the resulting alpha tracks theory.

This is the predictive test (alpha_predicted = gamma * (beta - 1)) that the
we identified as the NeurIPS-grade single figure.

Filter design: radial 2D FFT -> multiply magnitude spectrum by k^a, where a is
chosen so the post-filter eigenvalue decay has a target beta.

After saving, re-measures beta on each filtered variant and stores the actual
(post-filter) beta in the saved .pt file for downstream consumption.

Usage:
    python r3_bandpass_cifar.py --target-betas 1.0 1.7 2.0 --dataset cifar100
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

BASE = Path(__file__).parent
DATA_ROOT = BASE / "data"


def load_dataset(name):
    tf = transforms.ToTensor()
    if name == "cifar10":
        train = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=tf)
        test = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=tf)
    elif name == "cifar100":
        train = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=tf)
        test = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=tf)
    else:
        raise ValueError(name)
    Xtr = np.stack([x.numpy() for x, _ in train])  # (N, 3, 32, 32)
    ytr = np.array([y for _, y in train])
    Xte = np.stack([x.numpy() for x, _ in test])
    yte = np.array([y for _, y in test])
    return Xtr, ytr, Xte, yte


def radial_freq_grid(H, W):
    """Return (H, W) array of radial frequency magnitude k = sqrt(u^2 + v^2)."""
    u = np.fft.fftfreq(H) * H
    v = np.fft.fftfreq(W) * W
    U, V = np.meshgrid(u, v, indexing="ij")
    return np.sqrt(U**2 + V**2)


def apply_radial_filter(X, exponent_shift):
    """
    Multiply FFT magnitude of each image by k^exponent_shift.

    exponent_shift > 0 : steepen decay (more high-freq suppression)
    exponent_shift < 0 : flatten decay (more high-freq boost)
    exponent_shift == 0: identity.

    Returns filtered images in same shape, same dtype.
    """
    N, C, H, W = X.shape
    k = radial_freq_grid(H, W)
    k[0, 0] = 1.0  # avoid divide-by-zero at DC; zero DC gets multiplied by 1
    mask = k ** exponent_shift
    mask[0, 0] = 1.0  # preserve DC

    X_filt = np.empty_like(X)
    for c in range(C):
        F = np.fft.fft2(X[:, c, :, :], axes=(-2, -1))
        F_f = F * mask[None, :, :]
        X_filt[:, c, :, :] = np.real(np.fft.ifft2(F_f, axes=(-2, -1)))

    return X_filt.astype(X.dtype)


def measure_beta(X, fit_range=(5, 500)):
    """Measure spectral decay exponent beta by SVD of flattened centered data."""
    Xf = X.reshape(X.shape[0], -1).astype(np.float32)
    Xc = Xf - Xf.mean(axis=0)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    eig = np.sort((S**2) / Xc.shape[0])[::-1]
    k = np.arange(1, len(eig) + 1)
    lo, hi = fit_range
    lk = np.log(k[lo:hi])
    le = np.log(eig[lo:hi])
    c = np.polyfit(lk, le, 1)
    pred = np.polyval(c, lk)
    r2 = 1 - np.sum((le - pred) ** 2) / np.sum((le - le.mean()) ** 2)
    return float(-c[0]), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    ap.add_argument("--target-betas", nargs="+", type=float, default=[1.0, 1.7])
    ap.add_argument("--out-dir", default=str(BASE / "data" / "bandpass"))
    args = ap.parse_args()

    Xtr, ytr, Xte, yte = load_dataset(args.dataset)
    beta_orig, r2_orig = measure_beta(Xtr)
    print(f"[{args.dataset}] original beta = {beta_orig:.3f} (R^2 = {r2_orig:.3f})")

    out = Path(args.out_dir) / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"dataset": args.dataset, "beta_original": beta_orig, "variants": []}

    # For each target beta, binary-search the exponent_shift that produces it
    for beta_target in args.target_betas:
        # The shift needed: if original beta is b0 and we want b1, we multiply
        # eigenvalues by k^{-(b1-b0)} i.e. amplitude by k^{-(b1-b0)/2}.
        # But spectral eigendecay and radial image spectrum are related but not
        # identical. Use a simple binary search on a scalar shift a in [-2, 2].

        # beta is monotonically DECREASING in shift (more +shift boosts high freq -> flatter decay).
        # Search: if current b < target, need steeper decay -> MORE NEGATIVE shift -> bring hi down.
        lo, hi = -3.0, 3.0
        shift = 0.0
        beta_got = beta_orig
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            Xf = apply_radial_filter(Xtr[:5000], mid)
            b, _ = measure_beta(Xf)
            if b < beta_target:
                hi = mid
            else:
                lo = mid
            shift = mid
            beta_got = b
        print(f"  target beta = {beta_target:.2f}  -> shift = {shift:+.3f}  (subset beta = {beta_got:.3f})")

        # Apply on full train + test
        print(f"    applying filter to full dataset...")
        Xtr_f = apply_radial_filter(Xtr, shift)
        Xte_f = apply_radial_filter(Xte, shift)
        b_full, r2_full = measure_beta(Xtr_f)
        print(f"    full-dataset beta = {b_full:.3f} (R^2 = {r2_full:.3f})")

        # Save as npz
        tag = f"beta{beta_target:.2f}".replace(".", "p")
        npz_path = out / f"{args.dataset}_bandpass_{tag}.npz"
        np.savez(npz_path,
                 X_train=Xtr_f, y_train=ytr,
                 X_test=Xte_f, y_test=yte,
                 beta_target=beta_target, beta_actual=b_full,
                 shift=shift)
        print(f"    saved: {npz_path}")

        manifest["variants"].append({
            "beta_target": beta_target,
            "beta_actual": b_full,
            "r2": r2_full,
            "shift": shift,
            "path": str(npz_path),
        })

    mpath = out / "manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {mpath}")


if __name__ == "__main__":
    main()
