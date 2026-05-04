"""
R5 NeurIPS — Width-scaled DeiT-Tiny on CIFAR-10/100.

Addresses the critique: "two convolutional architectures = not
architecture-dependent in any meaningful sense." We add a non-convolutional
(transformer) architecture and re-test the architecture-dependent alpha
claim.

Width-scaled vanilla ViT:
  patch_size = 4 (for 32x32 -> 64 tokens)
  depth = 12
  num_heads = 3
  embed_dim in {96, 144, 192, 288}  (DeiT-Tiny default = 192)

Usage:
    python r5_deit_tiny.py --datasets cifar10 cifar100 --embed-dims 96 144 192 288 --seeds 0 1 2 --epochs 100
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import get_dataset

BASE = Path(__file__).parent
CKPT_DIR = BASE / "checkpoints" / "deit"
RESULTS_DIR = BASE / "results" / "deit"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Width-scaled ViT for small images
# =============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MHSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WidthScaledViT(nn.Module):
    """ViT scalable by embed_dim. Depth + heads + patch fixed.
    Penultimate representation = pre-classifier features (after final LayerNorm)."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 num_classes=100, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_feat = x[:, 0]  # (B, D) — penultimate representation
        return self.head(cls_feat)


# =============================================================================
# Hook for stable rank
# =============================================================================

def get_penultimate_hook_vit(model):
    """Hook the head's input = post-norm CLS feature."""
    activations = {}
    def hook_fn(module, input, output):
        activations["penultimate"] = (input[0] if isinstance(input, tuple) else input).detach().cpu()
    handle = model.head.register_forward_hook(hook_fn)
    return activations, handle


@torch.no_grad()
def measure_sr_vit(model, loader, device):
    model.eval()
    activations, handle = get_penultimate_hook_vit(model)
    blocks = []
    for x, _ in loader:
        x = x.to(device)
        _ = model(x)
        blocks.append(activations["penultimate"].clone())
    handle.remove()
    A = torch.cat(blocks, dim=0)
    A = A - A.mean(dim=0, keepdim=True)
    A = A.float()
    fro_sq = (A ** 2).sum().item()
    S = torch.linalg.svdvals(A)
    spec_sq = (S[0] ** 2).item()
    return float(fro_sq / spec_sq) if spec_sq > 0 else 0.0


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# =============================================================================
# Training loop
# =============================================================================

def train_one(dataset, embed_dim, seed, epochs, device):
    run = f"vit_{dataset}_d{embed_dim:03d}_s{seed}"
    out_path = RESULTS_DIR / f"{run}.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set, test_set, num_classes, input_shape = get_dataset(dataset, root=BASE / "data")
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                             num_workers=0, pin_memory=True)

    img_size = input_shape[-1]
    patch_size = 4 if img_size == 32 else 8
    model = WidthScaledViT(img_size=img_size, patch_size=patch_size,
                           in_chans=input_shape[0], num_classes=num_classes,
                           embed_dim=embed_dim, depth=12, num_heads=3).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    # AdamW + warmup is more stable than SGD for ViTs on small data
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    warmup = 5
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        e = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * e))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    losses = []
    t0 = time.time()
    print(f"\n[START] {run}  params={num_params:,}")
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
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  {run} ep {epoch}/{epochs}: loss={losses[-1]:.4f}  ({(time.time()-t0)/60:.1f} min)")

    acc = evaluate(model, test_loader, device)
    sr = measure_sr_vit(model, test_loader, device)

    torch.save({
        "model_state": model.state_dict(),
        "arch": "vit", "embed_dim": embed_dim,
        "num_classes": num_classes, "input_shape": list(input_shape),
        "patch_size": patch_size,
    }, CKPT_DIR / f"{run}.pt")

    result = {
        "run": run, "arch": "vit", "dataset": dataset,
        "embed_dim": embed_dim, "seed": seed,
        "num_params": num_params, "epochs": epochs,
        "final_train_loss": losses[-1],
        "top1_acc": acc, "stable_rank": sr,
        "total_time_s": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[DONE] {run}  acc={acc:.4f}  sr={sr:.2f}  ({(time.time()-t0)/60:.1f} min)")
    return result


def fit_alpha(records, dataset):
    runs = [r for r in records if r["dataset"] == dataset]
    if len(runs) < 2:
        return None
    by_d = {}
    for r in runs:
        by_d.setdefault(r["embed_dim"], []).append((r["num_params"], 1.0 - r["top1_acc"]))
    Ns, errs = [], []
    for d in sorted(by_d.keys()):
        pairs = by_d[d]
        Ns.append(pairs[0][0])
        errs.append(float(np.mean([p[1] for p in pairs])))
    Ns, errs = np.array(Ns, float), np.array(errs, float)
    log_n = np.log10(Ns)
    log_e = np.log10(errs)
    slope, intercept = np.polyfit(log_n, log_e, 1)
    pred = slope * log_n + intercept
    r2 = 1 - np.sum((log_e - pred) ** 2) / np.sum((log_e - log_e.mean()) ** 2)
    return {"dataset": dataset, "alpha": float(-slope), "r2": float(r2),
            "params": Ns.tolist(), "errs": errs.tolist()}


def fit_gamma(records, dataset):
    runs = [r for r in records if r["dataset"] == dataset]
    if len(runs) < 2:
        return None
    by_d = {}
    for r in runs:
        by_d.setdefault(r["embed_dim"], []).append((r["num_params"], r["stable_rank"]))
    Ns, srs = [], []
    for d in sorted(by_d.keys()):
        pairs = by_d[d]
        Ns.append(pairs[0][0])
        srs.append(float(np.mean([p[1] for p in pairs])))
    Ns, srs = np.array(Ns, float), np.array(srs, float)
    log_n = np.log10(Ns)
    log_sr = np.log10(srs)
    slope, intercept = np.polyfit(log_n, log_sr, 1)
    pred = slope * log_n + intercept
    r2 = 1 - np.sum((log_sr - pred) ** 2) / np.sum((log_sr - log_sr.mean()) ** 2)
    return {"dataset": dataset, "gamma": float(slope), "r2": float(r2),
            "params": Ns.tolist(), "srs": srs.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    ap.add_argument("--embed-dims", type=int, nargs="+", default=[96, 144, 192, 288])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'n/a'}")
    n = len(args.datasets) * len(args.embed_dims) * len(args.seeds)
    print(f"Configs: {len(args.datasets)} datasets × {len(args.embed_dims)} widths × "
          f"{len(args.seeds)} seeds = {n} runs")

    records = []
    for ds in args.datasets:
        for d in args.embed_dims:
            for s in args.seeds:
                r = train_one(ds, d, s, epochs=args.epochs, device=device)
                if r is not None:
                    records.append(r)

    fits_alpha = [fit_alpha(records, ds) for ds in args.datasets]
    fits_gamma = [fit_gamma(records, ds) for ds in args.datasets]
    fits_alpha = [f for f in fits_alpha if f]
    fits_gamma = [f for f in fits_gamma if f]

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({"records": records, "fits_alpha": fits_alpha, "fits_gamma": fits_gamma}, f, indent=2)

    print("\n=== ViT alpha (error vs params) per dataset ===")
    print(f"{'dataset':<14} {'alpha':>8} {'R2':>6}")
    for f in fits_alpha:
        print(f"{f['dataset']:<14} {f['alpha']:>8.4f} {f['r2']:>6.3f}")
    print("\n=== ViT gamma (stable_rank vs params) per dataset ===")
    print(f"{'dataset':<14} {'gamma':>8} {'R2':>6}")
    for f in fits_gamma:
        print(f"{f['dataset']:<14} {f['gamma']:>8.4f} {f['r2']:>6.3f}")


if __name__ == "__main__":
    main()
