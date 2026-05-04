"""
R5 NeurIPS — beta_act fit-window stability check.

Addresses the question: "Have you verified the fit by varying the upper k cutoff?"

Re-fits log sigma_k^2 = -beta_act log k + c on penultimate-layer activations of
the largest seed-0 ScaleCNN checkpoint per dataset, varying the fit window
across [5, 50], [5, 100], [5, 200], [5, 500].

A stable beta_act estimate requires the exponent to be ~constant across
windows; large drift = the power-law fit is fitting numerical noise.
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


def fit_beta(eigvals, lo, hi):
    eig = np.sort(eigvals)[::-1]
    eig = eig[eig > 0]
    hi = min(hi, len(eig))
    if hi <= lo + 1:
        return None
    lk = np.log(np.arange(lo, hi) + 1.0)
    le = np.log(eig[lo:hi])
    coef = np.polyfit(lk, le, 1)
    pred = np.polyval(coef, lk)
    r2 = 1 - np.sum((le - pred) ** 2) / np.sum((le - le.mean()) ** 2)
    return float(-coef[0]), float(r2), int(hi - lo)


def measure_for_dataset(dataset, arch, device):
    ckpt_dir = BASE / "checkpoints" / dataset
    arch_tag = "cnn" if arch == "scalecnn" else "mob"
    cks = sorted(ckpt_dir.glob(f"*{arch_tag}_*_s0.pt"))
    if not cks:
        return None
    ckpt_path = cks[-1]
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
    S = torch.linalg.svdvals(A.float())
    eig = (S.cpu().numpy()) ** 2

    print(f"\n[{dataset}] {ckpt_path.name}, activation matrix {tuple(A.shape)}")
    fits = {}
    for window in [(5, 50), (5, 100), (5, 200), (5, 500)]:
        result = fit_beta(eig, window[0], window[1])
        if result is None:
            continue
        beta, r2, n_pts = result
        key = f"k={window[0]}-{window[1]}"
        fits[key] = {"beta_act": beta, "r2": r2, "n_pts": n_pts}
        print(f"  {key:14s}: beta_act = {beta:.4f}  R2 = {r2:.3f}  ({n_pts} pts)")

    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "act_spectrum_stability.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset, "ckpt": ckpt_path.name, "fits": fits}, f, indent=2)
    return fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    choices=["cifar10", "cifar100", "tinyimagenet"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = [args.dataset] if args.dataset else ["cifar10", "cifar100", "tinyimagenet"]
    for ds in datasets:
        try:
            measure_for_dataset(ds, "scalecnn", device)
        except Exception as e:
            print(f"[ERR] {ds}: {e}")


if __name__ == "__main__":
    main()
