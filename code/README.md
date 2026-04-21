# Cloned Repositories

## Repo 1: AdamW-and-SGDW
- URL: https://github.com/loshchil/AdamW-and-SGDW
- Purpose: Reference implementation accompanying the decoupled weight decay paper.
- Location: `code/adamw-and-sgdw/`
- Key files: `main.lua`, `train.lua`, `opts.lua`, `UPDATETORCHFILES/`
- Notes:
  - Built for Torch7/Lua rather than modern PyTorch.
  - README focuses on CIFAR-10/ImageNet32x32 Shake-Shake experiments.
  - Useful mainly as a historical reference for optimizer behavior, not as a drop-in baseline for this workspace.

## Repo 2: why-weight-decay
- URL: https://github.com/tml-epfl/why-weight-decay
- Purpose: Official code for the NeurIPS 2024 paper on weight decay dynamics.
- Location: `code/why-weight-decay/`
- Key files: `overparameterized_nets/`, `large_language_models/`, top-level `README.md`
- Notes:
  - Contains reproducibility scripts for both vision and LLM experiments.
  - Requires a separate heavy environment and GPU-oriented setup.
  - Most useful here as a methodological reference for optimizer settings, ablation structure, and interpretation of weight decay effects.

## Validation Summary

- Both repositories cloned successfully.
- Neither repository was executed in this workspace because they target heavier training stacks than needed for the planned small tabular experiments.
- The experiment runner should treat them as reference implementations, not immediate dependencies.
