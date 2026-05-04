"""
R5 NeurIPS — Sample-count ablation on CIFAR-100 (full 100-class label space).

Test: does gamma vary with training set size at fixed
class count and fixed data spectrum?

For each sample_count in {10K, 25K, 50K (= full)}:
  Subsample CIFAR-100 train set to N samples (deterministic stratified
  by class), keep test set full, train ScaleCNN at 4 widths × 3 seeds × 100
  epochs. Measure stable rank gamma_sample.

Goal (combined with r5_class_count): decompose dataset-level gamma variation
      into structural drivers (samples, classes) vs spectral drivers (beta).

Usage:
    python r5_sample_count.py --samples 10000 25000 50000 --widths 8 16 32 64 --seeds 0 1 2 --epochs 100
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from phase1_train import ScaleCNN, evaluate
from measure_stable_rank import compute_stable_rank, get_penultimate_hook
from datasets import get_dataset

BASE = Path(__file__).parent
CKPT_DIR = BASE / "checkpoints" / "sample_count"
RESULTS_DIR = BASE / "results" / "sample_count"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def stratified_subset(dataset, n_samples, seed=0):
    """Stratified subsample: keep n_samples/n_classes from each class."""
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    per_class = n_samples // len(classes)
    rng = np.random.RandomState(seed)
    indices = []
    for c in classes:
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        indices.extend(idx_c[:per_class].tolist())
    return Subset(dataset, indices)


def make_loaders(n_samples, batch_size=128):
    train_set, test_set, _, _ = get_dataset("cifar100", root=BASE / "data")
    train_subset = stratified_subset(train_set, n_samples, seed=0)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def measure_sr(model, loader, device):
    model.eval()
    activations, handle = get_penultimate_hook(model, "scalecnn")
    blocks = []
    for x, _ in loader:
        x = x.to(device)
        _ = model(x)
        blocks.append(activations["penultimate"].clone())
    handle.remove()
    A = torch.cat(blocks, dim=0)
    if A.dim() > 2:
        A = A.view(A.size(0), -1)
    A = A - A.mean(dim=0, keepdim=True)
    return float(compute_stable_rank(A))


def train_one(width, n_samples, seed, epochs, device):
    run = f"smp{n_samples:05d}_c{width:03d}_s{seed}"
    out_path = RESULTS_DIR / f"{run}.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, test_loader = make_loaders(n_samples)
    model = ScaleCNN(base_channels=width, num_classes=100,
                     in_channels=3, n_blocks=4).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    losses = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        losses.append(loss_sum / n)
        scheduler.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  {run} ep {epoch}/{epochs}: loss={losses[-1]:.4f}  ({(time.time()-t0)/60:.1f} min)")

    metrics = evaluate(model, test_loader, device, num_classes=100)
    metrics.pop("_all_probs", None)
    sr = measure_sr(model, test_loader, device)

    torch.save({
        "model_state": model.state_dict(),
        "arch": "scalecnn", "size_param": width,
        "num_classes": 100, "input_shape": [3, 32, 32],
    }, CKPT_DIR / f"{run}.pt")

    result = {
        "run": run, "arch": "scalecnn", "width": width,
        "n_samples": n_samples, "seed": seed,
        "num_params": num_params, "epochs": epochs,
        "final_train_loss": losses[-1],
        "top1_acc": metrics["top1_acc"],
        "stable_rank": sr,
        "total_time_s": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[DONE] {run}  acc={metrics['top1_acc']:.4f}  sr={sr:.2f}")
    return result


def fit_gamma_per_samplecount(records, n_samples):
    runs = [r for r in records if r["n_samples"] == n_samples]
    if len(runs) < 2:
        return None
    by_w = {}
    for r in runs:
        by_w.setdefault(r["width"], []).append(r["stable_rank"])
    widths_sorted = sorted(by_w.keys())
    Ns, srs = [], []
    for w in widths_sorted:
        rep = next(r for r in runs if r["width"] == w)
        Ns.append(rep["num_params"])
        srs.append(float(np.mean(by_w[w])))
    Ns, srs = np.array(Ns, float), np.array(srs, float)
    log_n = np.log10(Ns)
    log_sr = np.log10(srs)
    slope, intercept = np.polyfit(log_n, log_sr, 1)
    pred = slope * log_n + intercept
    r2 = 1 - np.sum((log_sr - pred) ** 2) / np.sum((log_sr - log_sr.mean()) ** 2)
    return {
        "n_samples": n_samples,
        "gamma": float(slope), "r2": float(r2),
        "widths": widths_sorted, "srs_mean": srs.tolist(), "params": Ns.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="+", default=[10000, 25000, 50000])
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'n/a'}")
    n_runs = len(args.samples) * len(args.widths) * len(args.seeds)
    print(f"Configs: {len(args.samples)} sample sizes × {len(args.widths)} widths × "
          f"{len(args.seeds)} seeds = {n_runs} runs")

    records = []
    for s in args.samples:
        for w in args.widths:
            for sd in args.seeds:
                r = train_one(w, s, sd, epochs=args.epochs, device=device)
                if r is not None:
                    records.append(r)

    fits = []
    for s in args.samples:
        f = fit_gamma_per_samplecount(records, s)
        if f:
            fits.append(f)

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({"records": records, "fits": fits}, f, indent=2)

    print("\n=== Gamma vs sample count ===")
    print(f"{'n_samples':>10} {'gamma':>9} {'R2':>6}")
    for f in fits:
        print(f"{f['n_samples']:>10} {f['gamma']:>9.4f} {f['r2']:>6.3f}")


if __name__ == "__main__":
    main()
