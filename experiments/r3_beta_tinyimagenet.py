"""Measure beta on Tiny ImageNet (64x64)."""
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent


def main():
    import torchvision.transforms as transforms
    # Tiny ImageNet isn't in torchvision directly. Try the standard path.
    import torchvision.datasets as ds
    # Load from the expected location used by phase1_train's get_dataset
    from datasets import get_dataset
    train, _, _, _ = get_dataset("tinyimagenet", root=BASE / "data")
    print(f"Tiny ImageNet train size: {len(train)}")

    # Sample up to 50000 images (or the full set if smaller)
    N = min(50000, len(train))
    Xs = []
    for i in range(N):
        x, _ = train[i]
        Xs.append(x.numpy().flatten())
    X = np.stack(Xs).astype(np.float32)
    print(f"X shape: {X.shape}")

    Xc = X - X.mean(axis=0)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    eig = np.sort((S**2) / Xc.shape[0])[::-1]
    k = np.arange(1, len(eig) + 1)

    print("\nTinyImageNet EIGENSPECTRUM")
    print("=" * 50)
    for lo, hi in [(2, 200), (5, 300), (5, 500), (10, 1000), (5, 1500)]:
        if hi > len(eig):
            hi = len(eig)
        lk = np.log(k[lo:hi])
        le = np.log(eig[lo:hi])
        c = np.polyfit(lk, le, 1)
        pred = np.polyval(c, lk)
        r2 = 1 - np.sum((le - pred)**2) / np.sum((le - le.mean())**2)
        print(f"  k={lo}-{hi}: beta = {-c[0]:.4f}, R^2 = {r2:.4f}")

    lo, hi = 5, 500
    lk = np.log(k[lo:hi])
    le = np.log(eig[lo:hi])
    c = np.polyfit(lk, le, 1)
    out = {"dataset": "tinyimagenet", "beta": float(-c[0]), "fit_range": [lo, hi]}
    out_path = BASE / "results" / "tinyimagenet" / "beta_tinyimagenet.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
