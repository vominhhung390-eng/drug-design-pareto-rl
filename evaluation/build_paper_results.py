#!/usr/bin/env python
"""Build a unified, seed-level paper table from standardized evaluations.

The script deliberately discovers only ``evaluation_summary.csv`` files, so
partial training directories cannot enter the report.  Optional quality,
novelty, and RF-dispersion audits are merged when present.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEED_RE = re.compile(r"^(?P<method>.+)_seed(?P<seed>\d+)$")


def one_row(path: Path, experiment: str) -> dict:
    run_dir = path.parent.parent
    match = SEED_RE.match(run_dir.name)
    method = match.group("method") if match else run_dir.name
    seed = int(match.group("seed")) if match else np.nan
    row = pd.read_csv(path).iloc[0].to_dict()
    row.update(experiment=experiment, method=method, seed=seed, run_dir=str(run_dir))

    optional = [
        (path.parent / "quality_constrained" / "quality_constrained_summary.csv", "quality_"),
        (path.parent / "novelty_audit" / "novelty_summary.csv", "novelty_"),
        (run_dir / "predictor_uncertainty_calibrated" / "rf_tree_dispersion_summary.csv", "rf_"),
    ]
    for audit, prefix in optional:
        if audit.exists():
            values = pd.read_csv(audit).iloc[0].to_dict()
            row.update({prefix + key: value for key, value in values.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots", nargs="+", help="Inputs as LABEL=PATH; PATH is scanned recursively"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for spec in args.roots:
        if "=" not in spec:
            raise ValueError(f"Root must be LABEL=PATH, got: {spec}")
        label, raw_root = spec.split("=", 1)
        root = Path(raw_root)
        for path in sorted(root.rglob("evaluation_summary.csv")):
            rows.append(one_row(path, label))
    if not rows:
        raise FileNotFoundError("No completed standardized evaluations found")

    args.output.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows).sort_values(["experiment", "method", "seed"])
    runs.to_csv(args.output / "all_seed_metrics.csv", index=False, encoding="utf-8-sig")

    preferred = [
        "hypervolume", "pareto_size", "best_min_activity", "validity",
        "uniqueness_valid", "qed_mean", "structural_alert_rate",
        "scaffold_diversity", "scaffold_entropy",
        "quality_quality_pass_rate", "quality_quality_constrained_hypervolume",
        "novelty_exact_smiles_novelty", "rf_both_below_cutoff_rate",
        "rf_below_cutoff_hypervolume", "rf_tree_q05_hypervolume",
    ]
    metrics = [name for name in preferred if name in runs.columns]
    grouped = runs.groupby(["experiment", "method"], dropna=False)
    records = []
    for keys, frame in grouped:
        record = {"experiment": keys[0], "method": keys[1], "n_seeds": len(frame)}
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="coerce")
            record[f"{metric}_n"] = int(values.notna().sum())
            record[f"{metric}_mean"] = values.mean()
            record[f"{metric}_std"] = values.std(ddof=1)
        records.append(record)
    pd.DataFrame(records).sort_values(["experiment", "method"]).to_csv(
        args.output / "aggregate_metrics.csv", index=False, encoding="utf-8-sig"
    )

    completeness = []
    for (experiment, method), frame in grouped:
        completeness.append({
            "experiment": experiment,
            "method": method,
            "completed_seeds": len(frame),
            "quality_audited": int(frame.get("quality_quality_pass_rate", pd.Series(dtype=float)).notna().sum()),
            "novelty_audited": int(frame.get("novelty_exact_smiles_novelty", pd.Series(dtype=float)).notna().sum()),
            "rf_audited": int(frame.get("rf_both_below_cutoff_rate", pd.Series(dtype=float)).notna().sum()),
        })
    pd.DataFrame(completeness).to_csv(
        args.output / "completeness.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
