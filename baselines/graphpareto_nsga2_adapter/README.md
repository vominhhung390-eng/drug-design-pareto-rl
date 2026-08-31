# GraphPareto–NSGA-II dual-target adapter

Official source: <https://github.com/Jonas-Verhellen/MolecularGraphPareto>

Pinned source commit: `826e533b1b3995a8944e7c5cefe087806ff8c03f`.

`upstream_official/` is a pristine shallow clone. The adapter directly imports
the official graph mutation, crossover, structural arbiter, and molecule
classes. Non-dominated sorting and crowding selection are source-faithful ports
of the official `nsga-ii/nsga2.py` implementation. The scientific adaptations
are limited to the common source SMILES file, the two frozen EGFR/VEGFR2
predictors, exact terminal-proposal budget accounting, deterministic seeds,
checkpointing, and common output files.

The upstream-only `pygmo` hypervolume and `MultipleComparisons` internal-
similarity calls are per-generation reporting statistics. They do not affect
parent generation or NSGA-II survivor selection and are intentionally replaced
by the project's common evaluator for Windows portability.

Default method parameters remain the official NSGA-II defaults:

- initial population: 100
- survivor population: 100
- mutation parent batch: 20
- crossover parent-pair batch: 20
- structural rule set: Glaxo plus the upstream built-in property filters

Example smoke run:

```powershell
.\.venv\Scripts\python.exe .\adapter_optimize_dual_oracle.py `
  --output-dir ..\..\results\baselines\graphpareto_nsga2\smoke_128_seed42 `
  --budget 128 --seed 42 --resume
```
