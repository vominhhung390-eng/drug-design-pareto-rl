#!/usr/bin/env python
"""Strict ChEMBL sensitivity analysis for the locked EGFR/VEGFR2 RFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from validate_target_predictors import (
    aggregate_compounds,
    clean_activities,
    conformal_audit,
    evaluate_frame,
    fingerprints,
    make_plots,
    max_similarity,
    predict_locked,
    regression_metrics,
    safe_correlation,
    scaffold_split,
    sha256,
)


MUTATION_PATTERN = re.compile(
    r"\b(mutant|mutation|L858R|T790M|C797S|del19|exon\s*19|L861Q|G719[ACDSX]|S768I)\b",
    re.IGNORECASE,
)


def load_pages(directory: Path, target: str) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    metadata: list[dict] = []
    for path in sorted(directory.glob(f"{target.lower()}_offset*_limit1000.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        activities = payload.get("activities", [])
        rows.extend(activities)
        metadata.append({"path": str(path.resolve()), "sha256": sha256(path), "rows": len(activities)})
    if not rows:
        raise ValueError(f"No ChEMBL pages found for {target} in {directory}")
    return rows, metadata


def fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.3f}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "config" / "predictor_validation.json")
    parser.add_argument("--pages", type=Path, default=root / "data" / "external" / "chembl" / "predictor_validation")
    parser.add_argument("--output", type=Path, default=root / "results" / "predictor_validation_20260729" / "chembl_strict")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    locked_regression: list[dict] = []
    locked_classification: list[dict] = []
    reference_regression: list[dict] = []
    reference_classification: list[dict] = []
    audit_rows: list[dict] = []
    uncertainty_rows: list[dict] = []
    similarity_rows: list[dict] = []
    conformal_rows: dict[str, dict] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    source_metadata: dict[str, object] = {
        "source": "ChEMBL REST API",
        "query_scope": "human single-protein biochemical binding assays; exact IC50/Kd; explicit variants and mutation-labelled assay descriptions excluded",
        "retrieved_date": "2026-07-29",
        "targets": {},
    }

    for target, target_config in config["targets"].items():
        rows, pages = load_pages(args.pages, target)
        description_mutants = sum(
            bool(MUTATION_PATTERN.search(str(row.get("assay_description") or "")))
            for row in rows
        )
        rows = [
            row for row in rows
            if not MUTATION_PATTERN.search(str(row.get("assay_description") or ""))
        ]
        cleaned, audit = clean_activities(rows, config)
        compounds = aggregate_compounds(cleaned)
        historical_mask = compounds["first_document_year"].le(config["reference_end_year"]).to_numpy(bool)
        post_mask = compounds["first_document_year"].ge(config["post_publication_start_year"]).to_numpy(bool)
        strict_mask = compounds["first_document_year"].ge(config["strict_recent_start_year"]).to_numpy(bool)
        audit_rows.append(
            {
                "target": target,
                "downloaded_rows": sum(page["rows"] for page in pages),
                "excluded_mutation_description": description_mutants,
                **audit,
                "unique_compounds": len(compounds),
                "historical_compounds": int(historical_mask.sum()),
                "post_publication_novel_compounds": int(post_mask.sum()),
                "strict_recent_novel_compounds": int(strict_mask.sum()),
            }
        )
        source_metadata["targets"][target] = {
            "chembl_id": target_config["chembl_id"],
            "pages": pages,
        }
        cleaned.to_csv(args.output / f"{target.lower()}_eligible_measurements.csv", index=False, encoding="utf-8-sig")
        compounds.to_csv(args.output / f"{target.lower()}_aggregated_compounds.csv", index=False, encoding="utf-8-sig")

        x_operational, _ = fingerprints(compounds["smiles"], use_chirality=True)
        x_faithful, bitvectors = fingerprints(compounds["smiles"], use_chirality=False)
        model_path = root / target_config["model"]
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        mean, std, q05, q95 = predict_locked(model, x_operational)
        faithful_mean, _, _, _ = predict_locked(model, x_faithful)
        compounds["locked_prediction"] = mean
        compounds["locked_prediction_training_faithful"] = faithful_mean
        compounds["locked_tree_std"] = std
        compounds["locked_tree_q05"] = q05
        compounds["locked_tree_q95"] = q95
        compounds["operational_vs_training_faithful_abs_diff"] = np.abs(mean - faithful_mean)
        compounds["cohort_historical"] = historical_mask
        compounds["cohort_post_publication"] = post_mask
        compounds["cohort_strict_recent"] = strict_mask
        compounds["max_similarity_historical"] = np.nan
        compounds.loc[post_mask, "max_similarity_historical"] = max_similarity(
            [bitvectors[i] for i in np.flatnonzero(post_mask)],
            [bitvectors[i] for i in np.flatnonzero(historical_mask)],
        )

        cohorts = {
            "post_publication_novel": compounds.loc[post_mask].copy(),
            "strict_recent_novel": compounds.loc[strict_mask].copy(),
            "all_eligible_diagnostic": compounds.copy(),
        }
        for model_label, prediction_column in {
            "locked_operational_chiral": "locked_prediction",
            "locked_training_faithful_achiral": "locked_prediction_training_faithful",
        }.items():
            for cohort_name, cohort in cohorts.items():
                reg, cls = evaluate_frame(cohort, prediction_column, cohort_name, config["activity_thresholds"])
                locked_regression.append({"target": target, "model": model_label, **reg})
                locked_classification.extend({"target": target, "model": model_label, **row} for row in cls)

        post = cohorts["post_publication_novel"]
        abs_error = np.abs(post["locked_prediction"] - post["pactivity"])
        uncertainty_rows.append(
            {
                "target": target,
                "n": len(post),
                "tree_std_abs_error_spearman": safe_correlation(spearmanr, post["locked_tree_std"].to_numpy(float), abs_error.to_numpy(float)),
                "tree_q05_q95_empirical_coverage": float(np.mean((post["pactivity"] >= post["locked_tree_q05"]) & (post["pactivity"] <= post["locked_tree_q95"]))),
                "mean_tree_q05_q95_width": float(np.mean(post["locked_tree_q95"] - post["locked_tree_q05"])),
                "mean_max_similarity_historical": float(post["max_similarity_historical"].mean()),
                "rate_similarity_below_0_60": float((post["max_similarity_historical"] < 0.60).mean()),
                "mean_operational_vs_training_faithful_abs_diff": float(post["operational_vs_training_faithful_abs_diff"].mean()),
                "max_operational_vs_training_faithful_abs_diff": float(post["operational_vs_training_faithful_abs_diff"].max()),
            }
        )
        for label, lower, upper in (("lt_0.4", -np.inf, 0.4), ("0.4_to_0.6", 0.4, 0.6), ("0.6_to_0.8", 0.6, 0.8), ("ge_0.8", 0.8, np.inf)):
            subset = post[(post["max_similarity_historical"] >= lower) & (post["max_similarity_historical"] < upper)]
            if len(subset):
                similarity_rows.append({"target": target, "similarity_bin": label, **regression_metrics(subset["pactivity"].to_numpy(float), subset["locked_prediction"].to_numpy(float))})
        conformal_rows[target] = conformal_audit(post, config["conformal_coverage"], config["random_seed"])

        historical = compounds.loc[historical_mask].reset_index(drop=True)
        temporal = compounds.loc[post_mask].reset_index(drop=True)
        x_historical, _ = fingerprints(historical["smiles"], use_chirality=False)
        train_idx, test_idx = scaffold_split(historical, config["scaffold_test_fraction"], config["random_seed"])
        scaffold_model = RandomForestRegressor(n_estimators=config["reference_rf_estimators"], random_state=config["random_seed"], n_jobs=-1)
        scaffold_model.fit(x_historical[train_idx], historical.iloc[train_idx]["pactivity"].to_numpy(float))
        scaffold_test = historical.iloc[test_idx].copy()
        scaffold_test["reference_prediction"] = scaffold_model.predict(x_historical[test_idx])
        reg, cls = evaluate_frame(scaffold_test, "reference_prediction", "reconstructed_scaffold_split", config["activity_thresholds"])
        reference_regression.append({"target": target, "model": "reconstructed_ecfp4_rf", **reg})
        reference_classification.extend({"target": target, "model": "reconstructed_ecfp4_rf", **row} for row in cls)

        temporal_model = RandomForestRegressor(n_estimators=config["reference_rf_estimators"], random_state=config["random_seed"], n_jobs=-1)
        temporal_model.fit(x_historical, historical["pactivity"].to_numpy(float))
        temporal["reference_prediction"] = temporal_model.predict(x_faithful[np.flatnonzero(post_mask)])
        reg, cls = evaluate_frame(temporal, "reference_prediction", "reconstructed_temporal_split", config["activity_thresholds"])
        reference_regression.append({"target": target, "model": "reconstructed_ecfp4_rf", **reg})
        reference_classification.extend({"target": target, "model": "reconstructed_ecfp4_rf", **row} for row in cls)

        compounds.to_csv(args.output / f"{target.lower()}_locked_predictions.csv", index=False, encoding="utf-8-sig")
        prediction_frames[target] = compounds

    pd.DataFrame(audit_rows).to_csv(args.output / "data_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(locked_regression).to_csv(args.output / "locked_model_regression_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(locked_classification).to_csv(args.output / "locked_model_classification_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_regression).to_csv(args.output / "reference_model_regression_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_classification).to_csv(args.output / "reference_model_classification_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(uncertainty_rows).to_csv(args.output / "uncertainty_and_domain_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(similarity_rows).to_csv(args.output / "historical_similarity_bin_metrics.csv", index=False, encoding="utf-8-sig")
    (args.output / "conformal_summary.json").write_text(json.dumps(conformal_rows, indent=2), encoding="utf-8")
    (args.output / "source_metadata.json").write_text(json.dumps(source_metadata, indent=2), encoding="utf-8")
    make_plots(prediction_frames, args.output)

    post_rows = [row for row in locked_regression if row["cohort"] == "post_publication_novel"]
    lines = [
        "# Strict ChEMBL predictor validation",
        "",
        "Primary sensitivity analysis: human single-protein biochemical binding assays with exact IC50/Kd, standard validity flags, no explicit assay variant, and no mutation-labelled assay description. Compounds first reported for the target in 2024 or later form the post-publication proxy cohort; this is not guaranteed to be disjoint from the undisclosed BindingDB training compounds.",
        "",
        "| Target | Fingerprint mode | n | R2 | RMSE | MAE | Pearson r | Spearman rho | Bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in post_rows:
        lines.append(
            f"| {row['target']} | {row['model']} | {row['n']} | {fmt(row['r2'])} | {fmt(row['rmse'])} | {fmt(row['mae'])} | {fmt(row['pearson_r'])} | {fmt(row['spearman_rho'])} | {fmt(row['mean_bias_pred_minus_obs'])} |"
        )
    lines += [
        "",
        "The operational chiral fingerprint matches the completed generation experiments. The achiral mode matches the original POLYGON RF training script. Historical-similarity and RF-dispersion outputs are reliability proxies, not a true training-set applicability domain or calibrated uncertainty interval.",
    ]
    (args.output / "chembl_strict_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Strict ChEMBL validation complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
