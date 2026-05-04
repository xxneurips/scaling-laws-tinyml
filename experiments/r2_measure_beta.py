"""
R2 Comment 2: Measure beta from the eigenvalue spectrum of CIFAR-100 covariance matrix.
Compare with the assumed beta ≈ 1.3 used in the paper.
"""
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main():
    # Load CIFAR-100 training data
    print("Loading CIFAR-100 training data...")
    import torchvision
    import torchvision.transforms as transforms

    dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True,
        transform=transforms.ToTensor()
    )

    # Flatten all images to vectors (50000 x 3072)
    print("Flattening images...")
    X = np.array([img.numpy().flatten() for img, _ in dataset], dtype=np.float32)
    print(f"  Data shape: {X.shape}")

    # Center the data
    X_centered = X - X.mean(axis=0)

    # Compute covariance matrix eigenvalues
    print("Computing covariance matrix eigenvalues...")
    # Use SVD for numerical stability (cov = X^T X / n)
    # Eigenvalues of cov = singular values^2 / n
    _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = (S ** 2) / X_centered.shape[0]
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    print(f"  Number of eigenvalues: {len(eigenvalues)}")
    print(f"  Top 5 eigenvalues: {eigenvalues[:5]}")
    print(f"  Eigenvalue range: {eigenvalues[-1]:.6f} to {eigenvalues[0]:.4f}")

    # Fit power-law: lambda_k ~ k^{-beta}
    # Use log-log linear regression on the top eigenvalues
    # Skip the very first few (often anomalous) and the tail (noise floor)
    k = np.arange(1, len(eigenvalues) + 1)

    # Fit on range k=5 to k=500 (bulk of the spectrum, avoiding edges)
    fit_start, fit_end = 5, 500
    log_k = np.log(k[fit_start:fit_end])
    log_eig = np.log(eigenvalues[fit_start:fit_end])

    # Linear regression: log(lambda) = -beta * log(k) + const
    coeffs = np.polyfit(log_k, log_eig, 1)
    beta_measured = -coeffs[0]
    intercept = coeffs[1]

    # R² for the fit
    log_eig_pred = coeffs[0] * log_k + coeffs[1]
    ss_res = np.sum((log_eig - log_eig_pred) ** 2)
    ss_tot = np.sum((log_eig - np.mean(log_eig)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print(f"\n{'='*60}")
    print(f"  CIFAR-100 Eigenspectrum Analysis")
    print(f"{'='*60}")
    print(f"  Fit range: k = {fit_start} to {fit_end}")
    print(f"  Measured beta = {beta_measured:.4f}")
    print(f"  R^2 = {r_squared:.4f}")
    print(f"  Paper assumed beta ~ 1.3")
    print(f"  Literature range (natural images): beta ~ 1.0-1.2")
    print(f"{'='*60}")

    # Also try different fit ranges to check robustness
    print("\n  Robustness check (varying fit range):")
    for s, e in [(2, 200), (5, 300), (5, 500), (10, 1000), (5, 1500)]:
        if e > len(eigenvalues):
            e = len(eigenvalues)
        lk = np.log(k[s:e])
        le = np.log(eigenvalues[s:e])
        c = np.polyfit(lk, le, 1)
        beta_r = -c[0]
        le_pred = c[0] * lk + c[1]
        r2 = 1 - np.sum((le - le_pred)**2) / np.sum((le - np.mean(le))**2)
        print(f"    k={s}-{e}: beta = {beta_r:.4f}, R^2 = {r2:.4f}")

    # Generate figure
    if HAS_MPL:
        from pathlib import Path
        fig_dir = Path(__file__).parent.parent / "paper" / "fixed_after_review" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.loglog(k, eigenvalues, 'b-', alpha=0.7, linewidth=0.8, label='CIFAR-100 eigenvalues')

        # Plot the power-law fit
        k_fit = k[fit_start:fit_end]
        eig_fit = np.exp(intercept) * k_fit ** (-beta_measured)
        ax.loglog(k_fit, eig_fit, 'r--', linewidth=2,
                  label=f'Power-law fit: $\\beta = {beta_measured:.2f}$ ($R^2 = {r_squared:.3f}$)')

        ax.set_xlabel('Eigenvalue index $k$', fontsize=11)
        ax.set_ylabel('Eigenvalue $\\lambda_k$', fontsize=11)
        ax.set_title('CIFAR-100 Covariance Eigenspectrum', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        save_path = fig_dir / "fig_eigenspectrum.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[PLOT] Saved: {save_path}")


if __name__ == "__main__":
    main()
