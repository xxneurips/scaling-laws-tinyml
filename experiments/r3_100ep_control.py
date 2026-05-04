"""
R3 NeurIPS — 100-epoch matched-protocol control on original CIFAR-100.

Methodological note: the bandpass sweep uses 100 epochs, but the baseline in Section 6
is the 200-epoch main CIFAR-100 sweep. Without a matched-epoch control, the
flatness claim is confounded by training duration. This runs ScaleCNN at 4
widths × 2 seeds on the ORIGINAL CIFAR-100 at 100 epochs to match the bandpass
protocol.

Output:
  results/bandpass_cifar100/bp_original_c{width}_s{seed}.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from phase1_train import ScaleCNN, evaluate
from datasets import get_dataset

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results" / "bandpass_cifar100"


def train_control(width, seed, device):
    tag = f"bp_original_c{width:03d}_s{seed}"
    out_path = RESULTS_DIR / f"{tag}.json"
    if out_path.exists():
        print(f"[SKIP] {tag}")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set, test_set, num_classes, input_shape = get_dataset("cifar100", root=BASE / "data")
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0,
                             pin_memory=True)

    model = ScaleCNN(base_channels=width, num_classes=100, in_channels=3, n_blocks=4).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n[RUN] {tag}  params={num_params:,}")

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    losses = []
    t0 = time.time()
    for epoch in range(100):
        model.train()
        el = 0.0; n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            el += loss.item() * x.size(0); n += x.size(0)
        losses.append(el / n)
        scheduler.step()
        if epoch % 20 == 0 or epoch == 99:
            print(f"  ep {epoch}: loss={losses[-1]:.4f}  ({(time.time()-t0)/60:.1f} min)")

    metrics = evaluate(model, test_loader, device, num_classes=100)
    metrics.pop("_all_probs", None)

    result = {
        "run_name": tag, "arch": "scalecnn",
        "width": width, "beta_tag": "original_100ep",
        "beta_actual": 1.452, "seed": seed,
        "num_params": num_params, "epochs": 100,
        "train_losses": losses, "final_train_loss": losses[-1],
        "top1_acc": metrics["top1_acc"], "ece": metrics["ece"],
        "correct_mask": metrics["correct_mask"],
        "total_time_s": time.time() - t0,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"[DONE] {tag}  acc={metrics['top1_acc']:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0) if device=='cuda' else '-'}")
    widths = [8, 16, 32, 64]
    seeds = [0, 1]
    for width in widths:
        for seed in seeds:
            train_control(width, seed, device)


if __name__ == "__main__":
    main()
