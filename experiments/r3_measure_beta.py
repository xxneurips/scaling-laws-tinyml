"""R3 NeurIPS — Measure beta (spectral decay) on CIFAR-10 and CIFAR-100."""
import sys
import numpy as np
from pathlib import Path

DATASET = sys.argv[1] if len(sys.argv) > 1 else "cifar10"


def main():
    import torchvision
    import torchvision.transforms as transforms

    if DATASET == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
    elif DATASET == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")

    X = np.array([img.numpy().flatten() for img, _ in ds], dtype=np.float32)
    X_c = X - X.mean(axis=0)
    print(f"[{DATASET}] Data shape: {X.shape}")

    _, S, _ = np.linalg.svd(X_c, full_matrices=False)
    eig = (S ** 2) / X_c.shape[0]
    eig = np.sort(eig)[::-1]
    k = np.arange(1, len(eig) + 1)

    print(f"\n{DATASET.upper()} EIGENSPECTRUM")
    print("=" * 50)
    for lo, hi in [(2, 200), (5, 300), (5, 500), (10, 1000), (5, 1500)]:
        lk = np.log(k[lo:hi])
        le = np.log(eig[lo:hi])
        c = np.polyfit(lk, le, 1)
        pred = np.polyval(c, lk)
        r2 = 1 - np.sum((le - pred)**2) / np.sum((le - le.mean())**2)
        print(f"  k={lo}-{hi}: beta = {-c[0]:.4f}, R^2 = {r2:.4f}")

    # Save for paper
    lk = np.log(k[5:500])
    le = np.log(eig[5:500])
    c = np.polyfit(lk, le, 1)
    out = {
        "dataset": DATASET,
        "beta": float(-c[0]),
        "fit_range": [5, 500],
        "r2": float(1 - np.sum((le - np.polyval(c, lk))**2) / np.sum((le - le.mean())**2)),
    }
    import json
    out_path = Path(__file__).parent / "results" / DATASET / f"beta_{DATASET}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
