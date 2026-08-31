# Rebuilt model outputs

This directory is intentionally weight-free in the release package.

- `scripts/train_four_rf_predictors.py` creates `models/reproduced_oracles/`.
- `scripts/train_shared_polygon_vae.ps1` installs the selected shared VAE as `models/polygon_vae_best_valid_novel_stable_020.pt`.

Model hashes and source-data hashes are written beside the rebuilt artifacts.
