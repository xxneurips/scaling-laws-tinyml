"""
Stage 1: scaling-law training sweep
=====================================
Train model families at multiple sizes, 5 seeds each.
Three architecture families to span 10K–20M parameters (3+ orders of magnitude):
  - ScaleCNN: simple plain ConvNet (10K–500K params) for the tiny regime
  - MobileNetV2: inverted residuals (164K–20M params) for the standard regime
  - EfficientNet-Lite: compound-scaled CNN (~50K–20M params)

Supports datasets: cifar100, cifar10, tinyimagenet, speechcommands

Usage:
    python phase1_train.py                           # Run all configs (CIFAR-100)
    python phase1_train.py --dataset cifar10         # Run on CIFAR-10
    python phase1_train.py --arch mobilenet --width 0.25 --seed 0
    python phase1_train.py --arch scalecnn --channels 16 --seed 0
    python phase1_train.py --arch effnet --effnet-width 0.50 --seed 0
    python phase1_train.py --resume                  # Resume from checkpoints
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2

from datasets import get_dataset


# =============================================================================
# Configuration
# =============================================================================

# --- MobileNetV2 width multipliers (164K–20M params) ---
MOBILENET_WIDTHS = [0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]

# --- ScaleCNN base channel widths (10K–500K params) ---
SCALECNN_CHANNELS = [4, 8, 12, 16, 24, 32, 48, 64]

# --- EfficientNet-Lite width multipliers (~50K–20M params) ---
EFFNET_WIDTHS = [0.25, 0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

SEEDS = [0, 1, 2, 3, 4]
NUM_CLASSES = 100  # Default; overridden by --dataset
EPOCHS = 200
BATCH_SIZE = 128
LR_INIT = 0.1
LR_MIN = 1e-5
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
CUTOUT_SIZE = 8  # Cutout patch size for 32x32 images
NUM_WORKERS = 2

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "phase1"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "phase1"
DATA_DIR = BASE_DIR / "data"
PROBS_DIR = BASE_DIR / "results" / "phase1" / "probs"  # Softmax probability matrices


# =============================================================================
# Data
# =============================================================================

def get_data_loaders(dataset_name="cifar100"):
    """
    Get train/test DataLoaders for the specified dataset.
    Returns (train_loader, test_loader, num_classes, input_shape).
    """
    train_set, test_set, num_classes, input_shape = get_dataset(dataset_name, root=DATA_DIR)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    return train_loader, test_loader, num_classes, input_shape


# =============================================================================
# Model — ScaleCNN (plain ConvNet for tiny regime)
# =============================================================================

class ScaleCNN(nn.Module):
    """
    Simple plain ConvNet whose capacity scales cleanly with base channel count.
    Architecture: 4 or 5 conv blocks (conv-BN-ReLU-conv-BN-ReLU-pool) + linear head.
    Channels: [c, 2c, 4c, 8c] (or [c, 2c, 4c, 8c, 16c] for 5 blocks).

    With c=4: ~10K params. With c=64: ~500K params. Clean quadratic scaling.
    Set n_blocks=5 for higher resolution inputs (e.g. 64x64 Tiny ImageNet).
    """
    def __init__(self, base_channels, num_classes=100, in_channels=3, n_blocks=4):
        super().__init__()
        c = base_channels
        layers = []
        ch_in = in_channels

        for i in range(n_blocks):
            ch_out = c * (2 ** min(i, n_blocks - 1))
            layers.extend([
                nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ])
            if i < n_blocks - 1:
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.AdaptiveAvgPool2d(1))
            ch_in = ch_out

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(ch_out, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =============================================================================
# Model builders
# =============================================================================

def build_mobilenet(width_mult, num_classes=100, input_shape=(3, 32, 32)):
    """MobileNetV2 with given width multiplier, adapted for small inputs."""
    model = mobilenet_v2(weights=None, width_mult=width_mult, num_classes=num_classes)

    in_channels = input_shape[0]
    spatial = input_shape[1]

    # Adapt first conv for small input: stride 2->1 to preserve spatial resolution
    first_conv = model.features[0][0]
    stride = 1 if spatial <= 64 else 2
    model.features[0][0] = nn.Conv2d(
        in_channels, first_conv.out_channels,
        kernel_size=3, stride=stride, padding=1, bias=False
    )

    num_params = sum(p.numel() for p in model.parameters())
    return model, num_params


def build_scalecnn(base_channels, num_classes=100, input_shape=(3, 32, 32)):
    """ScaleCNN with given base channel width. Uses 5 blocks for 64x64 input."""
    spatial = input_shape[1]
    in_channels = input_shape[0]
    n_blocks = 5 if spatial >= 64 else 4
    model = ScaleCNN(base_channels, num_classes=num_classes,
                     in_channels=in_channels, n_blocks=n_blocks)
    num_params = sum(p.numel() for p in model.parameters())
    return model, num_params


def build_effnet(width_mult, num_classes=100, input_shape=(3, 32, 32)):
    """
    EfficientNet-Lite0 with custom width multiplier, adapted for small inputs.
    Requires timm library.
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for EfficientNet-Lite. Install: pip install timm")

    # Create EfficientNet-Lite0 with custom width
    # timm doesn't directly expose width_mult for efficientnet_lite, so we
    # use the base model and scale the classifier
    model = timm.create_model(
        'tf_efficientnet_lite0',
        pretrained=False,
        num_classes=num_classes,
    )

    in_channels = input_shape[0]
    spatial = input_shape[1]

    # Scale width: replace all conv layers' channels by multiplying with width_mult
    # For simplicity, we use timm's built-in width scaling via model variant names
    # or manually adjust. Here we use a simpler approach: scale the stem and blocks.
    if width_mult != 1.0:
        # Rebuild with scaled width using timm's channel_multiplier
        model = timm.create_model(
            'tf_efficientnet_lite0',
            pretrained=False,
            num_classes=num_classes,
            width_multiplier=width_mult,
        )

    # Adapt first conv for small inputs (stride 2->1 for 32x32)
    if spatial <= 64:
        first_conv = model.conv_stem
        model.conv_stem = nn.Conv2d(
            in_channels, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=1,  # reduce stride for small spatial dims
            padding=first_conv.padding,
            bias=False,
        )

    num_params = sum(p.numel() for p in model.parameters())
    return model, num_params


def build_model(arch, size_param, num_classes=100, input_shape=(3, 32, 32)):
    """Unified model builder. Returns (model, num_params)."""
    if arch == "mobilenet":
        return build_mobilenet(size_param, num_classes, input_shape)
    elif arch == "scalecnn":
        return build_scalecnn(int(size_param), num_classes, input_shape)
    elif arch == "effnet":
        return build_effnet(size_param, num_classes, input_shape)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples


# =============================================================================
# Evaluation — Tier 1 + Tier 2 metrics
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device, num_classes=100):
    """
    Comprehensive evaluation returning Tier 1, Tier 2, and extended metrics.

    Returns dict with:
        - top1_acc, top5_acc: Overall accuracy
        - per_class_acc/correct/total: Per-class breakdown
        - ece: Expected Calibration Error (15 bins)
        - per_bin_calibration: Per-bin calibration data (centers, accs, confs, counts)
        - test_ce_loss: Mean test cross-entropy loss
        - nll: Negative log-likelihood (same as CE loss for classification)
        - brier_score: Mean multi-class Brier score
        - avg_confidence_correct/incorrect: Confidence stats
        - mean_error_severity: Mean class distance of errors
        - predictions, targets, correct_mask: Raw data for overlap analysis
        - all_probs: Full softmax probability matrix (returned separately for .npz saving)
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    all_correct = []
    all_top5_correct = []
    all_ce_losses = []

    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)

    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)

        # Per-sample CE loss
        sample_losses = ce_loss_fn(outputs, targets)
        all_ce_losses.append(sample_losses.cpu())

        # Top-1
        _, preds = outputs.max(1)
        correct = preds.eq(targets)

        # Top-5 (guard for datasets with fewer than 5 classes)
        k = min(5, num_classes)
        _, topk_preds = outputs.topk(k, dim=1)
        top5_correct = topk_preds.eq(targets.unsqueeze(1)).any(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())
        all_correct.append(correct.cpu())
        all_top5_correct.append(top5_correct.cpu())

        # Per-class accounting
        for c in range(num_classes):
            mask = targets == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += (correct & mask).sum().item()

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)         # (N, C) full softmax matrix
    all_correct = torch.cat(all_correct)
    all_top5_correct = torch.cat(all_top5_correct)
    all_ce_losses = torch.cat(all_ce_losses)

    top1_acc = all_correct.float().mean().item()
    top5_acc = all_top5_correct.float().mean().item()

    per_class_acc = (per_class_correct / per_class_total.clamp(min=1)).numpy()

    # --- Test CE loss / NLL ---
    test_ce_loss = all_ce_losses.mean().item()
    nll = test_ce_loss  # CE loss = NLL for classification with softmax

    # --- Brier score (multi-class) ---
    # Brier = (1/N) * sum_i sum_c (p_ic - y_ic)^2
    one_hot = torch.zeros_like(all_probs)
    one_hot.scatter_(1, all_targets.unsqueeze(1), 1.0)
    brier_score = ((all_probs - one_hot) ** 2).sum(dim=1).mean().item()

    # --- ECE with per-bin data ---
    confidences = all_probs.max(dim=1).values
    ece, per_bin_calibration = compute_ece(
        confidences.numpy(), all_correct.numpy(), n_bins=15, return_per_bin=True
    )

    # Confidence statistics
    correct_mask = all_correct.bool()
    avg_conf_correct = confidences[correct_mask].mean().item() if correct_mask.any() else 0.0
    avg_conf_incorrect = confidences[~correct_mask].mean().item() if (~correct_mask).any() else 0.0

    # --- Error severity: mean rank distance of misclassifications ---
    incorrect_mask = ~correct_mask
    if incorrect_mask.any():
        wrong_preds = all_preds[incorrect_mask]
        true_labels = all_targets[incorrect_mask]
        error_distances = (wrong_preds - true_labels).abs().float()
        mean_error_severity = error_distances.mean().item()
    else:
        mean_error_severity = 0.0

    return {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "per_class_acc": per_class_acc.tolist(),
        "per_class_correct": per_class_correct.numpy().tolist(),
        "per_class_total": per_class_total.numpy().tolist(),
        "ece": ece,
        "per_bin_calibration": per_bin_calibration,
        "test_ce_loss": test_ce_loss,
        "nll": nll,
        "brier_score": brier_score,
        "avg_confidence_correct": avg_conf_correct,
        "avg_confidence_incorrect": avg_conf_incorrect,
        "mean_error_severity": mean_error_severity,
        # Store raw predictions for error overlap analysis
        "predictions": all_preds.numpy().tolist(),
        "targets": all_targets.numpy().tolist(),
        "correct_mask": all_correct.numpy().tolist(),
        # Full probability matrix (not serialized to JSON — saved as .npz separately)
        "_all_probs": all_probs.numpy(),
    }


def compute_ece(confidences, accuracies, n_bins=15, return_per_bin=False):
    """
    Expected Calibration Error with optional per-bin breakdown.

    Returns:
        If return_per_bin=False: scalar ECE (float)
        If return_per_bin=True: (ece, per_bin_dict) where per_bin_dict has:
            - bin_centers: list of bin center values
            - bin_accuracies: list of per-bin accuracy
            - bin_confidences: list of per-bin mean confidence
            - bin_counts: list of sample counts per bin
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = int(mask.sum())
        bin_centers.append(float((lo + hi) / 2))
        bin_counts.append(count)

        if count == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            continue

        bin_acc = float(accuracies[mask].mean())
        bin_conf = float(confidences[mask].mean())
        bin_weight = count / total
        ece += bin_weight * abs(bin_acc - bin_conf)

        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)

    if return_per_bin:
        per_bin = {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accs,
            "bin_confidences": bin_confs,
            "bin_counts": bin_counts,
        }
        return float(ece), per_bin

    return float(ece)


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt.get("best_acc", 0.0)


# =============================================================================
# Single run
# =============================================================================

def run_single(arch, size_param, seed, device, dataset_name="cifar100",
               num_classes=100, input_shape=(3, 32, 32), resume=False):
    """Train one model variant and save all metrics."""
    # Construct run name with dataset prefix for non-default datasets
    ds_prefix = "" if dataset_name == "cifar100" else f"{dataset_name}_"

    if arch == "mobilenet":
        run_name = f"{ds_prefix}mob_w{size_param:.2f}_s{seed}"
    elif arch == "effnet":
        run_name = f"{ds_prefix}eff_w{size_param:.2f}_s{seed}"
    else:
        run_name = f"{ds_prefix}cnn_c{int(size_param):03d}_s{seed}"

    # Use dataset-specific results directory
    results_dir = BASE_DIR / "results" / dataset_name
    probs_dir = results_dir / "probs"
    checkpoint_dir = BASE_DIR / "checkpoints" / dataset_name

    result_path = results_dir / f"{run_name}.json"
    probs_path = probs_dir / f"{run_name}_probs.npz"
    ckpt_path = checkpoint_dir / f"{run_name}.pt"

    # Skip if already completed
    if result_path.exists() and not resume:
        print(f"[SKIP] {run_name} — results exist")
        return

    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data
    train_loader, test_loader, _, _ = get_data_loaders(dataset_name)

    # Model
    model, num_params = build_model(arch, size_param, num_classes, input_shape)
    model = model.to(device)

    # Choose optimizer based on dataset (Adam for speech, SGD otherwise)
    use_adam = dataset_name == "speechcommands"
    epochs = 100 if dataset_name == "speechcommands" else EPOCHS

    print(f"\n{'='*70}")
    print(f"[RUN] {run_name} | arch={arch} | size={size_param} | params={num_params:,} | seed={seed}")
    print(f"      dataset={dataset_name} | classes={num_classes} | input={input_shape} | epochs={epochs}")
    print(f"{'='*70}")

    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    if use_adam:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=LR_INIT, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LR_MIN
    )

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    if resume and ckpt_path.exists():
        start_epoch, best_acc = load_checkpoint(ckpt_path, model, optimizer, scheduler)
        print(f"[RESUME] from epoch {start_epoch}, best_acc={best_acc:.4f}")

    # Training loop
    train_losses = []
    epoch_times = []

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        elapsed = time.time() - t0

        train_losses.append(train_loss)
        epoch_times.append(elapsed)

        # Periodic eval (every 10 epochs + last 20 epochs for convergence check)
        if epoch % 10 == 0 or epoch >= epochs - 20:
            metrics = evaluate(model, test_loader, device, num_classes)
            acc = metrics["top1_acc"]
            if acc > best_acc:
                best_acc = acc

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"loss={train_loss:.4f} | "
                f"acc={acc:.4f} | "
                f"best={best_acc:.4f} | "
                f"lr={lr:.6f} | "
                f"{elapsed:.1f}s"
            )

        # Checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_acc": best_acc,
                "arch": arch,
                "size_param": size_param,
                "num_classes": num_classes,
                "input_shape": input_shape,
            }, ckpt_path)

    # --- Final comprehensive evaluation ---
    print(f"\n[EVAL] Final evaluation for {run_name}...")
    final_metrics = evaluate(model, test_loader, device, num_classes)

    # Save full softmax probability matrix to .npz (N_test x C, ~4MB each)
    probs_matrix = final_metrics.pop("_all_probs")  # Remove from JSON result
    probs_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(probs_path, probs=probs_matrix)
    print(f"  Saved softmax probs: {probs_path} ({probs_matrix.shape})")

    # Save final checkpoint (keep it for stable rank measurement, TFLite conversion, etc.)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint({
        "epoch": epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_acc": best_acc,
        "arch": arch,
        "size_param": size_param,
        "num_classes": num_classes,
        "input_shape": input_shape,
    }, ckpt_path)

    # Convergence check: loss change in last 20 epochs
    if len(train_losses) >= 20:
        last20_losses = train_losses[-20:]
        loss_change_pct = abs(last20_losses[-1] - last20_losses[0]) / (abs(last20_losses[0]) + 1e-8) * 100
    else:
        loss_change_pct = -1.0

    # Assemble full result
    result = {
        "run_name": run_name,
        "arch": arch,
        "dataset": dataset_name,
        "size_param": size_param,
        "seed": seed,
        "num_params": num_params,
        "num_classes": num_classes,
        "input_shape": list(input_shape),
        "epochs": epochs,
        "converged": loss_change_pct < 1.0,
        "loss_change_last20_pct": loss_change_pct,
        "final_train_loss": train_losses[-1],
        "train_losses": train_losses,
        "mean_epoch_time_s": np.mean(epoch_times),
        "total_time_s": sum(epoch_times),
        **final_metrics,
    }

    # Save JSON
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] {run_name} | acc={final_metrics['top1_acc']:.4f} | ECE={final_metrics['ece']:.4f}")
    print(f"  CE loss={final_metrics['test_ce_loss']:.4f} | Brier={final_metrics['brier_score']:.4f}")
    print(f"  Saved to {result_path}")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1: scaling-law training sweep")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "cifar10", "tinyimagenet", "speechcommands"],
                        help="Dataset to train on (default: cifar100)")
    parser.add_argument("--arch", type=str, default=None,
                        choices=["mobilenet", "scalecnn", "effnet"],
                        help="Architecture to run (default: all)")
    parser.add_argument("--width", type=float, default=None,
                        help="MobileNetV2 width multiplier (single run)")
    parser.add_argument("--channels", type=int, default=None,
                        help="ScaleCNN base channels (single run)")
    parser.add_argument("--effnet-width", type=float, default=None,
                        help="EfficientNet-Lite width multiplier (single run)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to run")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset info
    dataset_name = args.dataset
    _, _, num_classes, input_shape = get_data_loaders(dataset_name)

    # Create directories
    results_dir = BASE_DIR / "results" / dataset_name
    checkpoint_dir = BASE_DIR / "checkpoints" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else SEEDS

    # Determine which runs to do: list of (arch, size_param, seed)
    configs = []

    if args.width is not None:
        for s in seeds:
            configs.append(("mobilenet", args.width, s))
    elif args.channels is not None:
        for s in seeds:
            configs.append(("scalecnn", args.channels, s))
    elif args.effnet_width is not None:
        for s in seeds:
            configs.append(("effnet", args.effnet_width, s))
    elif args.arch == "mobilenet":
        for w in MOBILENET_WIDTHS:
            for s in seeds:
                configs.append(("mobilenet", w, s))
    elif args.arch == "scalecnn":
        for c in SCALECNN_CHANNELS:
            for s in seeds:
                configs.append(("scalecnn", c, s))
    elif args.arch == "effnet":
        for w in EFFNET_WIDTHS:
            for s in seeds:
                configs.append(("effnet", w, s))
    else:
        # All configs: ScaleCNN + MobileNet + EfficientNet-Lite
        for c in SCALECNN_CHANNELS:
            for s in seeds:
                configs.append(("scalecnn", c, s))
        for w in MOBILENET_WIDTHS:
            for s in seeds:
                configs.append(("mobilenet", w, s))
        for w in EFFNET_WIDTHS:
            for s in seeds:
                configs.append(("effnet", w, s))

    total = len(configs)
    n_archs = len(set(a for a, _, _ in configs))
    n_sizes = len(set((a, p) for a, p, _ in configs))
    n_seeds = len(set(s for _, _, s in configs))
    print(f"\nScaling Laws: {total} runs ({n_sizes} model sizes x {n_seeds} seeds, {n_archs} architectures)")
    print(f"Dataset: {dataset_name} ({num_classes} classes, input {input_shape})")
    print(f"Results: {results_dir}")
    print(f"Checkpoints: {checkpoint_dir}\n")

    # Run
    completed = 0
    for i, (arch, size_param, seed) in enumerate(configs):
        print(f"\n--- Run {i+1}/{total} ---")
        run_single(
            arch, size_param, seed, device,
            dataset_name=dataset_name,
            num_classes=num_classes,
            input_shape=input_shape,
            resume=args.resume,
        )
        completed += 1

    print(f"\n{'='*70}")
    print(f"Training complete. {completed}/{total} runs finished.")
    print(f"Dataset: {dataset_name} | Results: {results_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
