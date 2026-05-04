"""
Stage 3A: Direct gamma (γ) measurement via stable rank of penultimate activations.
==================================================================================
For each trained checkpoint:
  1. Load model weights
  2. Forward pass 10K test images
  3. Hook penultimate layer activations → N_test × D matrix
  4. Compute stable rank: ||A||_F² / ||A||_2²
  5. Plot stable_rank vs N (params) across all architectures
  6. Fit γ directly: stable_rank ~ N^γ
  7. Compare measured γ to implied γ from Eq. 8: α = γ(β-1)

Usage:
    python measure_stable_rank.py                          # All datasets, all archs
    python measure_stable_rank.py --dataset cifar100       # Specific dataset
    python measure_stable_rank.py --dataset cifar100 --arch scalecnn
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Import model builders and data loaders from training script
from phase1_train import build_model, get_data_loaders, SCALECNN_CHANNELS, MOBILENET_WIDTHS, EFFNET_WIDTHS, SEEDS

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR.parent / "paper" / "figures"


# =============================================================================
# Stable rank computation
# =============================================================================

def get_penultimate_hook(model, arch):
    """
    Register a forward hook on the penultimate layer.
    Returns a dict that will hold the activations after forward pass.
    """
    activations = {}

    def hook_fn(module, input, output):
        # Store the input to the final classifier (= penultimate activations)
        if isinstance(input, tuple):
            activations["penultimate"] = input[0].detach().cpu()
        else:
            activations["penultimate"] = input.detach().cpu()

    # Hook the final linear layer — its input is the penultimate representation
    if arch == "scalecnn":
        handle = model.classifier.register_forward_hook(hook_fn)
    elif arch == "mobilenet":
        handle = model.classifier[-1].register_forward_hook(hook_fn)
    elif arch == "effnet":
        handle = model.classifier.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    return activations, handle


def compute_stable_rank(activation_matrix):
    """
    Compute stable rank of an activation matrix A (N × D).
    stable_rank = ||A||_F² / ||A||_2²
    where ||A||_2 is the largest singular value.
    """
    A = activation_matrix
    frobenius_sq = (A ** 2).sum().item()
    # Compute largest singular value via SVD (only need the first)
    # Use torch for efficiency
    A_tensor = torch.from_numpy(A).float() if isinstance(A, np.ndarray) else A.float()
    S = torch.linalg.svdvals(A_tensor)
    spectral_sq = (S[0] ** 2).item()

    if spectral_sq == 0:
        return 0.0
    return frobenius_sq / spectral_sq


@torch.no_grad()
def measure_model_stable_rank(model, loader, device, arch, num_classes):
    """Forward pass all test data, collect penultimate activations, compute stable rank."""
    model.eval()
    activations, handle = get_penultimate_hook(model, arch)

    all_acts = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        _ = model(inputs)
        all_acts.append(activations["penultimate"])

    handle.remove()

    # Concatenate: (N_test, D)
    act_matrix = torch.cat(all_acts, dim=0)
    if act_matrix.dim() > 2:
        act_matrix = act_matrix.view(act_matrix.size(0), -1)

    # Center the activations (zero-mean columns)
    act_matrix = act_matrix - act_matrix.mean(dim=0, keepdim=True)

    sr = compute_stable_rank(act_matrix)
    return sr, act_matrix.shape


# =============================================================================
# Main measurement loop
# =============================================================================

def measure_all(dataset_name, arch_filter, device):
    """Measure stable rank for all checkpoints of a given dataset."""
    checkpoint_dir = BASE_DIR / "checkpoints" / dataset_name
    results = []

    if not checkpoint_dir.exists():
        print(f"[WARN] No checkpoints found at {checkpoint_dir}")
        return results

    # Load dataset info
    _, test_loader, num_classes, input_shape = get_data_loaders(dataset_name)

    # Enumerate checkpoints
    for ckpt_file in sorted(checkpoint_dir.glob("*.pt")):
        # Parse run name to get arch and size_param
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        arch = ckpt.get("arch")
        size_param = ckpt.get("size_param")
        ckpt_num_classes = ckpt.get("num_classes", num_classes)
        ckpt_input_shape = tuple(ckpt.get("input_shape", input_shape))

        if arch is None or size_param is None:
            # Try to parse from filename
            name = ckpt_file.stem
            if name.startswith("cnn_"):
                arch = "scalecnn"
                size_param = int(name.split("_c")[1].split("_s")[0])
            elif name.startswith("mob_"):
                arch = "mobilenet"
                size_param = float(name.split("_w")[1].split("_s")[0])
            elif name.startswith("eff_"):
                arch = "effnet"
                size_param = float(name.split("_w")[1].split("_s")[0])
            else:
                print(f"[SKIP] Cannot parse: {ckpt_file.name}")
                continue

        if arch_filter and arch != arch_filter:
            continue

        # Build model and load weights
        model, num_params = build_model(arch, size_param, ckpt_num_classes, ckpt_input_shape)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)

        # Measure stable rank
        sr, shape = measure_model_stable_rank(model, test_loader, device, arch, ckpt_num_classes)

        result = {
            "run_name": ckpt_file.stem,
            "arch": arch,
            "size_param": size_param,
            "num_params": num_params,
            "stable_rank": sr,
            "activation_shape": list(shape),
            "dataset": dataset_name,
        }
        results.append(result)
        print(f"  {ckpt_file.stem}: params={num_params:>10,} | stable_rank={sr:.2f} | shape={shape}")

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    return results


def fit_gamma(results):
    """
    Fit stable_rank ~ N^γ per architecture.
    Returns dict of {arch: {gamma, r_squared, ...}}.
    """
    by_arch = defaultdict(list)
    for r in results:
        by_arch[r["arch"]].append(r)

    fits = {}
    for arch, runs in by_arch.items():
        # Aggregate by size (average over seeds)
        by_size = defaultdict(list)
        for r in runs:
            by_size[r["size_param"]].append(r["stable_rank"])

        params = []
        srs = []
        for sp in sorted(by_size.keys()):
            # Get num_params from first run at this size
            run = next(r for r in runs if r["size_param"] == sp)
            params.append(run["num_params"])
            srs.append(np.mean(by_size[sp]))

        params = np.array(params, dtype=float)
        srs = np.array(srs, dtype=float)

        if len(params) < 2:
            continue

        # Fit log(sr) = gamma * log(N) + const
        log_n = np.log10(params)
        log_sr = np.log10(srs)

        if HAS_SCIPY:
            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(log_n, log_sr)
        else:
            # Simple numpy fallback
            A = np.vstack([log_n, np.ones(len(log_n))]).T
            slope, intercept = np.linalg.lstsq(A, log_sr, rcond=None)[0]
            r_value = np.corrcoef(log_n, log_sr)[0, 1]
            p_value = None
            std_err = 0.0

        fits[arch] = {
            "gamma": slope,
            "r_squared": r_value ** 2,
            "std_err": std_err,
            "params": params.tolist(),
            "stable_ranks": srs.tolist(),
        }

    return fits


# =============================================================================
# Plotting
# =============================================================================

ARCH_COLORS = {"scalecnn": "#2563eb", "mobilenet": "#dc2626", "effnet": "#16a34a"}
ARCH_MARKERS = {"scalecnn": "o", "mobilenet": "s", "effnet": "D"}
ARCH_LABELS = {"scalecnn": "ScaleCNN", "mobilenet": "MobileNetV2", "effnet": "EfficientNet-Lite"}


def plot_stable_rank(fits, dataset_name, save_path):
    """Plot stable rank vs params with gamma fit lines."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for arch, fit in fits.items():
        params = np.array(fit["params"])
        srs = np.array(fit["stable_ranks"])
        color = ARCH_COLORS.get(arch, "gray")
        marker = ARCH_MARKERS.get(arch, "o")
        label = ARCH_LABELS.get(arch, arch)

        ax.scatter(params, srs, color=color, marker=marker, s=60, zorder=3,
                   label=f"{label} (data)")

        # Fit line
        x_fit = np.logspace(np.log10(params.min() * 0.5), np.log10(params.max() * 2), 50)
        log_srs = np.log10(srs)
        log_params = np.log10(params)
        slope = fit["gamma"]
        intercept = np.mean(log_srs) - slope * np.mean(log_params)
        y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
        ax.plot(x_fit, y_fit, "--", color=color, alpha=0.6,
                label=rf"{label}: $\gamma$={fit['gamma']:.3f}, $R^2$={fit['r_squared']:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters ($N$)")
    ax.set_ylabel("Stable Rank")
    ax.set_title(f"Stable Rank vs. Model Size ({dataset_name})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Stable rank figure saved: {save_path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Measure stable rank of penultimate activations")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "cifar10", "tinyimagenet", "speechcommands"])
    parser.add_argument("--arch", type=str, default=None,
                        choices=["scalecnn", "mobilenet", "effnet"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="Only measure checkpoints from this seed (default: 0)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Measure stable ranks
    print(f"\nMeasuring stable ranks for {args.dataset}...")
    all_results = measure_all(args.dataset, args.arch, device)

    if not all_results:
        print("[ERROR] No results. Ensure checkpoints exist (retrain with updated phase1_train.py).")
        return

    # Fit gamma per architecture
    fits = fit_gamma(all_results)

    print(f"\n{'='*70}")
    print("Gamma fits:")
    for arch, fit in fits.items():
        label = ARCH_LABELS.get(arch, arch)
        print(f"  {label}: gamma = {fit['gamma']:.4f} +/- {fit['std_err']:.4f}, R^2 = {fit['r_squared']:.4f}")

    # Theory comparison: alpha = gamma * (beta - 1)
    # beta ≈ 1.30 from compute_beta_and_loss.py for CIFAR-100
    beta = 1.30
    print(f"\n  Implied alpha from theory (beta={beta:.2f}):")
    for arch, fit in fits.items():
        label = ARCH_LABELS.get(arch, arch)
        alpha_implied = fit["gamma"] * (beta - 1)
        print(f"    {label}: alpha_implied = {fit['gamma']:.4f} * ({beta:.2f} - 1) = {alpha_implied:.4f}")

    # Save results
    output_dir = BASE_DIR / "results" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stable_rank_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "results": all_results,
            "fits": {k: {kk: vv for kk, vv in v.items()} for k, v in fits.items()},
            "dataset": args.dataset,
        }, f, indent=2)
    print(f"\n[SAVED] {output_path}")

    # Plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_stable_rank(fits, args.dataset, FIGURES_DIR / f"stable_rank_{args.dataset}.pdf")


if __name__ == "__main__":
    main()
