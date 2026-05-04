"""
R4 NeurIPS — Re-train bandpass-CIFAR-100 ScaleCNN sweeps with checkpoint
save, then measure gamma per variant via stable rank.

Closes the §6 internal-consistency gap: previously alpha_pred was computed
with the unfiltered-CIFAR-100 gamma=0.177 for *all* variants, but the
paper's headline (Result 3) is that gamma is data-dependent. Filtered
variants are different data, so should have their own gamma.

Output: results/bandpass_gamma/{beta_tag}_gamma.json with gamma_variant,
        plus a summary alpha_pred(beta, gamma_variant) vs alpha_meas table.

Usage (RTX 3090, ~6 GPU-hours expected):
    python r4_bandpass_gamma.py --epochs 100 --widths 8 16 32 64 --seeds 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from phase1_train import ScaleCNN, evaluate
from r3_train_bandpass import BandpassDataset, load_bandpass
from measure_stable_rank import compute_stable_rank, get_penultimate_hook

BASE = Path(__file__).parent
CKPT_DIR = BASE / "checkpoints" / "bandpass_cifar100"
RESULTS_DIR = BASE / "results" / "bandpass_gamma"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def measure_gamma(model, test_loader, device, arch="scalecnn"):
    model.eval()
    activations, handle = get_penultimate_hook(model, arch)
    blocks = []
    for x, _ in test_loader:
        x = x.to(device)
        _ = model(x)
        blocks.append(activations["penultimate"].clone())
    handle.remove()
    A = torch.cat(blocks, dim=0)
    if A.dim() > 2:
        A = A.view(A.size(0), -1)
    A = A - A.mean(dim=0, keepdim=True)
    return float(compute_stable_rank(A))


def train_one(width, beta_tag, seed, epochs, device):
    run_name = f"bp_{beta_tag}_c{width:03d}_s{seed}"
    ckpt_path = CKPT_DIR / f"{run_name}.pt"
    out_path = RESULTS_DIR / f"{run_name}.json"
    if ckpt_path.exists() and out_path.exists():
        print(f"[SKIP] {run_name} exists")
        with open(out_path) as f:
            return json.load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Xtr, ytr, Xte, yte, beta_actual = load_bandpass(beta_tag)
    train_loader = DataLoader(
        BandpassDataset(Xtr, ytr, augment=True),
        batch_size=128, shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        BandpassDataset(Xte, yte, augment=False),
        batch_size=128, shuffle=False, num_workers=2, pin_memory=True,
    )

    model = ScaleCNN(base_channels=width, num_classes=100, in_channels=3, n_blocks=4).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
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
        train_losses.append(loss_sum / n)
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  {run_name} ep {epoch}/{epochs}: loss={train_losses[-1]:.4f}  "
                  f"({(time.time()-t0)/60:.1f} min)")

    metrics = evaluate(model, test_loader, device, num_classes=100)
    metrics.pop("_all_probs", None)

    sr = measure_gamma(model, test_loader, device, arch="scalecnn")

    torch.save({
        "model_state": model.state_dict(),
        "arch": "scalecnn",
        "size_param": width,
        "num_classes": 100,
        "input_shape": [3, 32, 32],
    }, ckpt_path)

    result = {
        "run_name": run_name,
        "arch": "scalecnn",
        "width": width,
        "beta_tag": beta_tag,
        "beta_actual": beta_actual,
        "seed": seed,
        "num_params": num_params,
        "epochs": epochs,
        "final_train_loss": train_losses[-1],
        "top1_acc": metrics["top1_acc"],
        "stable_rank": sr,
        "total_time_s": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[DONE] {run_name}  acc={metrics['top1_acc']:.4f}  sr={sr:.2f}")
    return result


def fit_gamma_per_variant(records, beta_tag):
    runs = [r for r in records if r["beta_tag"] == beta_tag]
    if len(runs) < 2:
        return None
    by_w = {}
    for r in runs:
        by_w.setdefault(r["width"], []).append(r["stable_rank"])
    widths_sorted = sorted(by_w.keys())
    Ns, srs = [], []
    for w in widths_sorted:
        run_at_w = next(r for r in runs if r["width"] == w)
        Ns.append(run_at_w["num_params"])
        srs.append(float(np.mean(by_w[w])))
    Ns, srs = np.array(Ns, float), np.array(srs, float)
    log_n = np.log10(Ns)
    log_sr = np.log10(srs)
    slope, intercept = np.polyfit(log_n, log_sr, 1)
    pred = slope * log_n + intercept
    r2 = 1 - np.sum((log_sr - pred) ** 2) / np.sum((log_sr - log_sr.mean()) ** 2)
    return {
        "beta_tag": beta_tag,
        "beta_actual": float(np.mean([r["beta_actual"] for r in runs])),
        "gamma_variant": float(slope),
        "r2": float(r2),
        "widths": widths_sorted,
        "stable_ranks_mean": srs.tolist(),
        "params": Ns.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--betas", nargs="+", default=["beta1p20", "beta1p70", "beta2p00"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'n/a'}")

    records = []
    for beta_tag in args.betas:
        for w in args.widths:
            for seed in args.seeds:
                r = train_one(w, beta_tag, seed, epochs=args.epochs, device=device)
                if r is not None:
                    records.append(r)

    fits = []
    for beta_tag in args.betas:
        f = fit_gamma_per_variant(records, beta_tag)
        if f:
            fits.append(f)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"records": records, "fits": fits}, f, indent=2)

    print("\n=== Per-variant gamma ===")
    print(f"{'beta_tag':<12} {'beta_actual':>11} {'gamma':>8} {'R2':>6}")
    for f in fits:
        print(f"{f['beta_tag']:<12} {f['beta_actual']:>11.3f} {f['gamma_variant']:>8.4f} {f['r2']:>6.3f}")

    # Internally consistent prediction: alpha_pred = gamma_variant * (beta_variant - 1)
    # Compare to alpha_meas already collected by r3_train_bandpass.
    alpha_meas_lookup = {  # from FINDINGS.md (matched-protocol 100-ep)
        "beta1p20": 0.126,
        "original_100ep": 0.141,  # beta=1.452
        "beta1p70": 0.123,
        "beta2p00": 0.127,
    }
    print("\n=== Internally consistent prediction vs. measurement ===")
    print(f"{'variant':<14} {'beta':>7} {'gamma':>8} {'alpha_pred':>11} {'alpha_meas':>11} {'ratio':>7}")
    for f in fits:
        beta = f["beta_actual"]
        gamma = f["gamma_variant"]
        alpha_pred = gamma * (beta - 1.0)
        alpha_meas = alpha_meas_lookup.get(f["beta_tag"])
        ratio = (alpha_meas / alpha_pred) if (alpha_meas and alpha_pred and abs(alpha_pred) > 1e-6) else None
        meas_str = f"{alpha_meas:.3f}" if alpha_meas else "n/a"
        ratio_str = f"{ratio:.2f}x" if ratio else "n/a"
        print(f"{f['beta_tag']:<14} {beta:>7.3f} {gamma:>8.4f} {alpha_pred:>11.4f} {meas_str:>11} {ratio_str:>7}")


if __name__ == "__main__":
    main()
