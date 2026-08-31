# Local compatibility patches

- The official archive requested `torch==2.12.0` from the CUDA 12.6 wheel
  index. This machine has an RTX 5090 (Blackwell, compute capability 12.0),
  so the environment is pinned to `torch==2.11.0` from the CUDA 12.8 index.
- No REINVENT model architecture, objective, loss, optimizer, or training
  algorithm is changed by this patch.
- `reinvent.utils.hw_report` imported the Unix-only Python `resource` module
  unconditionally. The import and Unix memory report are now guarded on
  Windows; CUDA reporting is unchanged.
- Added SciPy to the environment manifest because the bundled plotting module
  imports `scipy.stats.gaussian_kde` during CLI startup.
- Transfer learning reads the locked shared source file with
  `standardize_smiles = false`. Canonicalization/desalting is performed once
  by the shared oracle at scoring time; this avoids a REINVENT-only rewrite of
  the training set and a redundant single-process pass over 1.58M rows.
