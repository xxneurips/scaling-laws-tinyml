# Decomposing the Tiny-Regime Scaling Exponent — Code & Measurements

Anonymous code release accompanying the NeurIPS 2026 submission *"Decomposing the
Tiny-Regime Scaling Exponent: Class Count, Not Spectrum, Drives the Effective
Rank"*. This repository contains all training, measurement, and analysis scripts
needed to reproduce the paper's main and appendix results, plus the per-run JSON
measurement files used to make every table and figure.

## Directory layout

```
experiments/
  r2_*.py            # Stage 2: cross-arch Jaccard, bootstrap CIs, β estimation
  r3_*.py            # Stage 3: bandpass intervention, controlled Jaccard
  r4_*.py            # Stage 4: activation-spectrum measurement, multi-seed γ
  r5_*.py            # Stage 5: γ-decomposition (class count, sample count, ViT)
  datasets.py        # Dataset loaders (CIFAR-10/100, Tiny ImageNet)
  measure_stable_rank.py  # Stable-rank computation on penultimate activations
  compute_beta_and_loss.py
  compute_cis.py     # Bootstrap confidence intervals for α / γ
  model_selection.py # ScaleCNN + MobileNetV2 width-scaled architectures
  phase1_train.py    # Baseline width sweep
  phase1_analyze.py
  results/           # All per-run JSONs (used to make every paper figure/table)
    class_count/     # γ vs class count on CIFAR-100 subsets (§6, Table 2)
    sample_count/    # γ vs training-set size (§6)
    bandpass_gamma/  # multi-seed Appendix E rerun
    deit/            # DeiT-Tiny ViT comparator (§6 last paragraph)
    cifar10/, cifar100/, tinyimagenet/  # Stage-1/2 width sweeps
```

## Reproducing the headline numbers

```bash
pip install -r requirements.txt

# γ-decomposition table (paper §6, Table 2)
python experiments/r5_synthesize.py     # reads results/{class_count,sample_count,bandpass_gamma}/
python experiments/r5_partial_fit.py    # log-linear fits

# Re-run individual ablations (each ~2-4 hr on one RTX 3090, 100 epochs):
python experiments/r5_class_count.py    --classes 10 25 50 100 --widths 8 16 32 64 --seeds 0 1 2
python experiments/r5_sample_count.py   --samples 10000 25000 50000 --widths 8 16 32 64 --seeds 0 1 2
python experiments/r5_deit_tiny.py      --datasets cifar10 cifar100 --embed-dims 96 144 192 288 --seeds 0 1 2
python experiments/r4_bandpass_gamma.py --betas beta1p20 beta1p70 beta2p00 --widths 8 16 32 64 --seeds 0 1 2

# Activation-spectrum stability (Appendix A.1)
python experiments/r5_beta_act_stability.py
```

Total full-sweep compute: ≈8 GPU-days on a single RTX 3090.

## Datasets

CIFAR-10, CIFAR-100, and Tiny ImageNet are loaded via `torchvision` /
`torchvision.datasets.ImageFolder`. The bandpass-filtered CIFAR-100 variants
used in the controlled-β experiment are constructed on the fly by
`r3_bandpass_cifar.py` (no external download required).

## License

Code: MIT. Measurement JSONs: CC-BY 4.0.
