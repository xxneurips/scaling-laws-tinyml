"""
R4 NeurIPS — Measure activation-spectrum exponent beta_act.

Tests the Appendix A proxy assumption: stable_rank ~ K(N) holds iff
penultimate-activation singular values follow sigma_k^2 ~ k^{-beta_act}
with beta_act approximately matching beta_data.

For each dataset (CIFAR-10, CIFAR-100, Tiny ImageNet), for the largest
ScaleCNN width we have, we hook the penultimate activations on the full
test set, SVD them, and fit log(sigma_k^2) ~ -beta_act log(k).

Usage:
    python r4_measure_act_spectrum.py --dataset cifar100 --arch scalecnn
    python r4_measure_act_spectrum.py            # all datasets, scalecnn at largest width

Output: results/{dataset}/act_spectrum.json  with beta_act, beta_data, ratio.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phase1_train import build_model, get_data_loaders
from measure_stable_rank import get_penultimate_hook

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"


@torch.no_grad()
def collect_penultimate(model, loader, device, arch):
    """Forward pass test set, collect penultimate-layer activations as N x D matrix."""
    model.eval()
    activations, handle = get_penultimate_hook(model, arch)
    blocks = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        _ = model(inputs)
        blocks.append(activations["penultimate"].clone())
    handle.remove()
    A = torch.cat(blocks, dim=0)
    if A.dim() > 2:
        A = A.view(A.size(0), -1)
    A = A - A.mean(dim=0, keepdim=True)
    return A


def fit_beta_powerlaw(eigvals, lo=5, hi=None):
    """
    Fit log(eig_k) = -beta * log(k) + c on k in [lo, hi).
    Returns (beta, r2, fit_range).
    """
    eig = np.sort(eigvals)[::-1]
    eig = eig[eig > 0]
    if hi is None:
        hi = min(500, len(eig))
    lk = np.log(np.arange(lo, hi) + 1.0)  # +1 so lo=5 -> k=6 etc; near-equivalent
    le = np.log(eig[lo:hi])
    coef = np.polyfit(lk, le, 1)
    beta_act = -coef[0]
    pred = np.polyval(coef, lk)
    r2 = 1 - np.sum((le - pred) ** 2) / np.sum((le - le.mean()) ** 2)
    return float(beta_act), float(r2), [lo, hi]


def measure_for_dataset(dataset, arch, device):
    """Pick largest checkpoint of `arch` on `dataset`, compute beta_act."""
    ckpt_dir = BASE / "checkpoints" / dataset
    arch_tag = "cnn" if arch == "scalecnn" else "mob"
    # Match both `{arch_tag}_*_s0.pt` and `{dataset}_{arch_tag}_*_s0.pt` naming
    cks = sorted(ckpt_dir.glob(f"*{arch_tag}_*_s0.pt"))
    if not cks:
        print(f"[WARN] no {arch} seed-0 checkpoints at {ckpt_dir}")
        return None
    ckpt_path = cks[-1]  # largest size_param at seed 0
    print(f"[{dataset}] loading {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", arch)
    size_param = ckpt.get("size_param")
    num_classes = ckpt.get("num_classes")
    input_shape = tuple(ckpt.get("input_shape"))

    _, test_loader, _, _ = get_data_loaders(dataset)
    model, _ = build_model(arch, size_param, num_classes, input_shape)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    A = collect_penultimate(model, test_loader, device, arch)
    print(f"  activation matrix: {tuple(A.shape)}")

    S = torch.linalg.svdvals(A.float())
    eig = (S.cpu().numpy()) ** 2  # singular values squared = eigenvalues of A^T A
    beta_act, r2_act, fit_range = fit_beta_powerlaw(eig, lo=5, hi=min(500, len(eig)))

    # Load beta_data for the same dataset
    beta_path = BASE / "results" / dataset / f"beta_{dataset}.json"
    if beta_path.exists():
        with open(beta_path) as f:
            beta_data = json.load(f)["beta"]
    else:
        beta_data = None

    out = {
        "dataset": dataset,
        "arch": arch,
        "checkpoint": ckpt_path.name,
        "size_param": size_param,
        "n_test": int(A.shape[0]),
        "d_act": int(A.shape[1]),
        "beta_act": beta_act,
        "r2_act": r2_act,
        "fit_range": fit_range,
        "beta_data": beta_data,
        "ratio_act_over_data": (beta_act / beta_data) if beta_data else None,
    }
    print(
        f"  beta_act = {beta_act:.4f}  (R^2={r2_act:.3f})    "
        f"beta_data = {beta_data}    ratio = {out['ratio_act_over_data']}"
    )

    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "act_spectrum.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None,
                    choices=["cifar10", "cifar100", "tinyimagenet"])
    ap.add_argument("--arch", type=str, default="scalecnn",
                    choices=["scalecnn", "mobilenet"])
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = [args.dataset] if args.dataset else ["cifar10", "cifar100", "tinyimagenet"]
    summary = []
    for ds in datasets:
        try:
            r = measure_for_dataset(ds, args.arch, device)
            if r is not None:
                summary.append(r)
        except Exception as e:
            print(f"[ERR] {ds}: {e}")

    if summary:
        print("\n=== Activation-spectrum summary ===")
        print(f"{'dataset':<14} {'arch':<10} {'beta_act':>9} {'beta_data':>10} {'ratio':>7}")
        for r in summary:
            ratio = f"{r['ratio_act_over_data']:.3f}" if r['ratio_act_over_data'] else "n/a"
            bd = f"{r['beta_data']:.3f}" if r['beta_data'] else "n/a"
            print(f"{r['dataset']:<14} {r['arch']:<10} {r['beta_act']:>9.4f} {bd:>10} {ratio:>7}")


if __name__ == "__main__":
    main()
