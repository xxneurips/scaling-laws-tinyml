"""
R2 Comment 3: Plot training loss scaling across 5 seeds to visualize stability.
Shows final training loss vs. parameters (log-log) for all seeds,
plus the power-law fit for alpha_train.
"""
import json
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "cifar100"


def load_runs():
    runs = []
    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        if json_file.stem.endswith("_summary") or json_file.stem.endswith("_results"):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            if "arch" in data and "train_losses" in data:
                runs.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return runs


def main():
    runs = load_runs()

    # Group by (arch, size_param)
    by_config = {}
    for r in runs:
        key = (r["arch"], r["size_param"])
        by_config.setdefault(key, []).append(r)

    print("=" * 80)
    print("Training Loss Scaling (all seeds)")
    print("=" * 80)

    for arch in ["scalecnn", "mobilenet"]:
        label = "ScaleCNN" if arch == "scalecnn" else "MobileNetV2"
        print(f"\n{label}:")
        print(f"  {'Config':>10} {'Params':>10} {'Mean Loss':>12} {'Std Loss':>10} {'Seeds':>6}")
        print(f"  {'-'*55}")

        configs = sorted([k for k in by_config if k[0] == arch], key=lambda k: by_config[k][0]["num_params"])

        params_list = []
        mean_losses = []
        std_losses = []
        all_seed_losses = []

        for key in configs:
            seed_runs = by_config[key]
            final_losses = [r["train_losses"][-1] for r in seed_runs]
            params = seed_runs[0]["num_params"]
            mean_l = np.mean(final_losses)
            std_l = np.std(final_losses)

            size_label = f"c={key[1]}" if arch == "scalecnn" else f"m={key[1]:.2f}"
            print(f"  {size_label:>10} {params:>10,} {mean_l:>12.4f} {std_l:>10.4f} {len(final_losses):>6}")

            params_list.append(params)
            mean_losses.append(mean_l)
            std_losses.append(std_l)
            all_seed_losses.append(final_losses)

        # Fit power law: L ~ N^{-alpha_train}
        log_params = np.log(params_list)
        log_losses = np.log(mean_losses)
        coeffs = np.polyfit(log_params, log_losses, 1)
        alpha_train = -coeffs[0]

        # R²
        pred = coeffs[0] * log_params + coeffs[1]
        ss_res = np.sum((log_losses - pred) ** 2)
        ss_tot = np.sum((log_losses - np.mean(log_losses)) ** 2)
        r2 = 1 - ss_res / ss_tot

        print(f"\n  Power-law fit: alpha_train = {alpha_train:.4f}, R^2 = {r2:.4f}")

    # Generate figure
    if not HAS_MPL:
        print("\nMatplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, arch in enumerate(["scalecnn", "mobilenet"]):
        ax = axes[ax_idx]
        label = "ScaleCNN" if arch == "scalecnn" else "MobileNetV2"

        configs = sorted([k for k in by_config if k[0] == arch], key=lambda k: by_config[k][0]["num_params"])

        params_arr = []
        for key in configs:
            seed_runs = by_config[key]
            params = seed_runs[0]["num_params"]
            params_arr.append(params)

            # Plot each seed as a faint dot
            for r in seed_runs:
                final_loss = r["train_losses"][-1]
                ax.scatter(params, final_loss, color='C0', alpha=0.3, s=20, zorder=2)

        # Plot means with error bars
        mean_losses = []
        std_losses = []
        for key in configs:
            seed_runs = by_config[key]
            final_losses = [r["train_losses"][-1] for r in seed_runs]
            mean_losses.append(np.mean(final_losses))
            std_losses.append(np.std(final_losses))

        ax.errorbar(params_arr, mean_losses, yerr=std_losses,
                     fmt='o-', color='C0', markersize=6, capsize=3,
                     linewidth=1.5, zorder=3, label='Mean ± std (5 seeds)')

        # Power-law fit
        log_p = np.log(params_arr)
        log_l = np.log(mean_losses)
        coeffs = np.polyfit(log_p, log_l, 1)
        alpha_train = -coeffs[0]
        r2 = 1 - np.sum((log_l - (coeffs[0]*log_p + coeffs[1]))**2) / np.sum((log_l - np.mean(log_l))**2)

        p_fit = np.logspace(np.log10(min(params_arr)*0.8), np.log10(max(params_arr)*1.2), 100)
        l_fit = np.exp(coeffs[1]) * p_fit ** coeffs[0]
        ax.plot(p_fit, l_fit, 'r--', linewidth=1.5, alpha=0.7,
                label=f'$L \\sim N^{{-{alpha_train:.2f}}}$ ($R^2={r2:.3f}$)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Parameters $N$', fontsize=11)
        ax.set_ylabel('Final Training Loss', fontsize=11)
        ax.set_title(f'({chr(97+ax_idx)}) {label}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

    plt.suptitle('Training Loss Scaling Law (5 seeds per configuration)', fontsize=13, y=1.02)
    plt.tight_layout()

    # Save to both locations
    for fig_dir in [
        BASE_DIR.parent / "paper" / "fixed_after_review" / "figures",
        ]:
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_path = fig_dir / "fig_training_loss_scaling.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")

    plt.close()


if __name__ == "__main__":
    main()
