# Model outputs

The release package contains only two fixed weights:

- `models/oracles/target_EGFR_model.pkl`
- `models/oracles/target_VEGFR2_model.pkl`

Their exact historical row-level training data is unavailable, so these files
are required for formal output-level reproduction. See `models/oracles/README.md`.
All other model outputs are rebuilt.

- `scripts/train_four_rf_predictors.py` creates the formal PARP1/BRD4 models in
  `models/reproduced_oracles/`; the opt-in recovered-data mode also creates a
  separate EGFR/VEGFR2 alternative there.
- `scripts/train_shared_polygon_vae.ps1` installs the selected shared VAE as `models/polygon_vae_best_valid_novel_stable_020.pt`.

Model hashes and source-data hashes are written beside the rebuilt artifacts.
