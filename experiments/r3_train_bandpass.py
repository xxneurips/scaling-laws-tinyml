"""
R3 NeurIPS — Train ScaleCNN on bandpass-filtered CIFAR-100 variants.

Produces alpha(beta) data for the predictive test of spectral capacity theory.

Usage:
    python r3_train_bandpass.py  # runs full sweep
    python r3_train_bandpass.py --widths 8 16 32 64 --betas 1.20 1.70 2.00 --seeds 0 1 --epochs 100
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from phase1_train import ScaleCNN, evaluate

BASE = Path(__file__).parent
BANDPASS_DIR = BASE / "data" / "bandpass" / "cifar100"
RESULTS_DIR = BASE / "results" / "bandpass_cifar100"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class BandpassDataset(Dataset):
    """CIFAR-100 images after radial frequency filtering. Expects float32 arrays."""
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            # Random crop with padding 4
            x = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
            h = torch.randint(0, 9, (1,)).item()
            w = torch.randint(0, 9, (1,)).item()
            x = x[:, h:h + 32, w:w + 32]
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-1])
        return x, self.y[idx]


def load_bandpass(beta_tag):
    """Load an npz variant into datasets."""
    p = BANDPASS_DIR / f"cifar100_bandpass_{beta_tag}.npz"
    data = np.load(p, allow_pickle=False)
    Xtr, ytr = data["X_train"], data["y_train"]
    Xte, yte = data["X_test"], data["y_test"]
    beta_actual = float(data["beta_actual"])
    return Xtr, ytr, Xte, yte, beta_actual


def train_one_config(width, beta_tag, seed, epochs=100, device="cuda"):
    run_name = f"bp_{beta_tag}_c{width:03d}_s{seed}"
    out_path = RESULTS_DIR / f"{run_name}.json"
    if out_path.exists():
        print(f"[SKIP] {run_name} exists")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Xtr, ytr, Xte, yte, beta_actual = load_bandpass(beta_tag)
    print(f"\n[RUN] {run_name}  beta_actual={beta_actual:.3f}  Xtr={Xtr.shape}")

    train_loader = DataLoader(BandpassDataset(Xtr, ytr, augment=True),
                              batch_size=128, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(BandpassDataset(Xte, yte, augment=False),
                             batch_size=128, shuffle=False, num_workers=2,
                             pin_memory=True)

    model = ScaleCNN(base_channels=width, num_classes=100, in_channels=3, n_blocks=4).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)
        train_losses.append(epoch_loss / n)
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch}/{epochs}: loss={train_losses[-1]:.4f}  ({(time.time()-t0)/60:.1f} min)")

    # Final eval
    metrics = evaluate(model, test_loader, device, num_classes=100)
    metrics.pop("_all_probs", None)  # drop npz blob

    result = {
        "run_name": run_name,
        "arch": "scalecnn",
        "width": width,
        "beta_tag": beta_tag,
        "beta_actual": beta_actual,
        "seed": seed,
        "num_params": num_params,
        "epochs": epochs,
        "train_losses": train_losses,
        "final_train_loss": train_losses[-1],
        "top1_acc": metrics["top1_acc"],
        "top5_acc": metrics["top5_acc"],
        "ece": metrics["ece"],
        "correct_mask": metrics["correct_mask"],
        "total_time_s": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"[DONE] {run_name}  acc={metrics['top1_acc']:.4f}  ({(time.time()-t0)/60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--betas", nargs="+", default=["beta1p20", "beta1p70", "beta2p00"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'n/a'}")

    total = len(args.widths) * len(args.betas) * len(args.seeds)
    i = 0
    for beta_tag in args.betas:
        for width in args.widths:
            for seed in args.seeds:
                i += 1
                print(f"\n--- Run {i}/{total} ---")
                train_one_config(width, beta_tag, seed, epochs=args.epochs, device=device)


if __name__ == "__main__":
    main()
