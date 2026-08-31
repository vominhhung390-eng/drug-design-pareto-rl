#!/usr/bin/env python
"""Validate the locked EGFR/VEGFR2 RF oracles without overstating evidence.

The original POLYGON RF training compounds are not distributed with the model.
Accordingly, this script reports three deliberately separated analyses:

1. Locked-model concordance on post-publication ChEMBL compounds that have no
   earlier target record in the downloaded ChEMBL activity history.
2. A reconstructed ECFP4-RF pipeline evaluated with scaffold and temporal
   splits.  This validates the modelling recipe, not the locked weights.
3. RF tree dispersion and similarity to historical ChEMBL chemistry as
   reliability proxies.  Historical similarity is not labelled as the true
   predictor applicability domain because the exact RF training set is absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

RDLogger.DisableLog("rdApp.error")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url: str, timeout: int, retries: int) -> dict[str, Any]:
    headers = {"User-Agent": "predictor-validation/1.0 (academic reproducibility)"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Failed after {retries} attempts: {url}") from last_error


def api_status(base: str, timeout: int, retries: int) -> dict[str, Any]:
    for endpoint in ("status.json", "chembl_release.json?limit=100"):
        try:
            return fetch_json(f"{base}/{endpoint}", timeout, retries)
        except Exception:
            continue
    return {"status": "unavailable"}


def download_activities(
    base: str,
    target_id: str,
    cache_path: Path,
    page_size: int,
    timeout: int,
    retries: int,
    refresh: bool,
) -> list[dict[str, Any]]:
    if cache_path.exists() and not refresh:
        return [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line]

    params = {
        "target_chembl_id": target_id,
        "standard_type__in": "IC50,Kd",
        "standard_relation": "=",
        "pchembl_value__isnull": "false",
        # Restrict the response to fields consumed by the strict cleaning and
        # provenance pipeline.  ChEMBL otherwise returns a much wider activity
        # schema, making large target downloads unnecessarily slow.
        "only": ",".join(
            [
                "activity_id",
                "assay_chembl_id",
                "assay_description",
                "assay_type",
                "assay_variant_mutation",
                "bao_format",
                "bao_label",
                "canonical_smiles",
                "data_validity_comment",
                "document_chembl_id",
                "document_year",
                "molecule_chembl_id",
                "pchembl_value",
                "potential_duplicate",
                "standard_flag",
                "standard_relation",
                "standard_type",
                "standard_units",
                "target_organism",
            ]
        ),
        "limit": str(page_size),
        "offset": "0",
    }
    url = f"{base}/activity.json?{urllib.parse.urlencode(params)}"
    rows: list[dict[str, Any]] = []
    while url:
        payload = fetch_json(url, timeout, retries)
        rows.extend(payload.get("activities", []))
        next_url = payload.get("page_meta", {}).get("next")
        url = urllib.parse.urljoin(base + "/", next_url) if next_url else ""
        print(f"{target_id}: downloaded {len(rows)} activities", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_bindingdb_snapshot(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    try:
        return payload["getLindsByUniprotsResponse"]["affinities"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unexpected BindingDB response schema: {path}") from exc


def fetch_pubmed_years(
    pmids: Iterable[str],
    cache_path: Path,
    endpoint: str,
    timeout: int,
    retries: int,
    refresh: bool,
) -> dict[str, int]:
    if cache_path.exists() and not refresh:
        return {str(k): int(v) for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
    unique_pmids = sorted({str(pmid) for pmid in pmids if str(pmid).isdigit()}, key=int)
    years: dict[str, int] = {}
    for start in range(0, len(unique_pmids), 200):
        batch = unique_pmids[start : start + 200]
        params = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(batch), "retmode": "json"})
        payload = fetch_json(f"{endpoint}?{params}", min(timeout, 60), retries)
        result = payload.get("result", {})
        for pmid in batch:
            record = result.get(pmid, {})
            date_text = " ".join(str(record.get(key, "")) for key in ("pubdate", "epubdate", "sortpubdate"))
            match = re.search(r"(?:19|20)\d{2}", date_text)
            if match:
                years[pmid] = int(match.group(0))
        print(f"PubMed: resolved {min(start + 200, len(unique_pmids))}/{len(unique_pmids)} IDs", flush=True)
        time.sleep(0.35)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(years, indent=2, sort_keys=True), encoding="utf-8")
    return years


def clean_bindingdb_activities(
    rows: list[dict[str, Any]],
    pubmed_years: dict[str, int],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    audit: defaultdict[str, int] = defaultdict(int)
    cleaned: list[dict[str, Any]] = []
    allowed_types = set(config["allowed_activity_types"])
    exact_number = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
    for row in rows:
        audit["api_rows"] += 1
        if row.get("affinity_type") not in allowed_types:
            audit["excluded_activity_type"] += 1
            continue
        affinity_text = str(row.get("affinity", "")).strip()
        if not exact_number.fullmatch(affinity_text):
            audit["excluded_censored_or_non_numeric"] += 1
            continue
        affinity_nm = float(affinity_text)
        if not np.isfinite(affinity_nm) or affinity_nm <= 0:
            audit["excluded_nonpositive_affinity"] += 1
            continue
        pactivity = 9.0 - math.log10(affinity_nm)
        if not 2.0 <= pactivity <= 12.0:
            audit["excluded_out_of_range"] += 1
            continue
        canonical, mol = canonicalize(row.get("smile", ""))
        if canonical is None or mol is None:
            audit["excluded_invalid_smiles"] += 1
            continue
        scaffold = safe_scaffold(canonical, mol)
        pmid = str(row.get("pmid") or "").strip()
        year = pubmed_years.get(pmid)
        cleaned.append(
            {
                "smiles": canonical,
                "pactivity": pactivity,
                "affinity_nm": affinity_nm,
                "standard_type": row.get("affinity_type"),
                "pmid": pmid or None,
                "doi": row.get("doi"),
                "document_year": year,
                "year_known": year is not None,
                "scaffold": scaffold,
            }
        )
        audit["eligible_measurements"] += 1
        audit["eligible_measurements_with_pubmed_year"] += int(year is not None)
    frame = pd.DataFrame(cleaned)
    if frame.empty:
        return frame, dict(audit)
    return frame.sort_values(["smiles", "document_year", "pmid"], na_position="last").reset_index(drop=True), dict(audit)


def canonicalize(smiles: str) -> tuple[str | None, Chem.Mol | None]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if fragments:
        mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), mol


def safe_scaffold(canonical: str, mol: Chem.Mol) -> str:
    """Return an achiral Murcko scaffold without letting malformed stereo abort the audit."""
    try:
        scaffold_mol = Chem.Mol(mol)
        Chem.RemoveStereochemistry(scaffold_mol)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=scaffold_mol, includeChirality=False
        )
        return scaffold or f"ACYCLIC:{canonical}"
    except Exception:
        return f"SCAFFOLD_FALLBACK:{canonical}"


def clean_activities(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    audit: defaultdict[str, int] = defaultdict(int)
    cleaned: list[dict[str, Any]] = []
    allowed_types = set(config["allowed_activity_types"])
    allowed_assays = set(config["allowed_assay_types"])
    for row in rows:
        audit["api_rows"] += 1
        if row.get("standard_type") not in allowed_types:
            audit["excluded_activity_type"] += 1
            continue
        if row.get("assay_type") not in allowed_assays:
            audit["excluded_assay_type"] += 1
            continue
        if row.get("target_organism") != "Homo sapiens":
            audit["excluded_organism"] += 1
            continue
        if row.get("standard_flag") != 1:
            audit["excluded_nonstandard"] += 1
            continue
        if row.get("standard_relation") != "=":
            audit["excluded_relation"] += 1
            continue
        if row.get("standard_units") != "nM":
            audit["excluded_units"] += 1
            continue
        if row.get("data_validity_comment") not in (None, ""):
            audit["excluded_validity_flag"] += 1
            continue
        if int(row.get("potential_duplicate") or 0) != 0:
            audit["excluded_potential_duplicate"] += 1
            continue
        if row.get("assay_variant_mutation") not in (None, ""):
            audit["excluded_explicit_variant"] += 1
            continue
        try:
            y = float(row["pchembl_value"])
            year = int(row["document_year"])
        except (TypeError, ValueError, KeyError):
            audit["excluded_missing_value_or_year"] += 1
            continue
        if not 2.0 <= y <= 12.0:
            audit["excluded_out_of_range"] += 1
            continue
        canonical, mol = canonicalize(row.get("canonical_smiles", ""))
        if canonical is None or mol is None:
            audit["excluded_invalid_smiles"] += 1
            continue
        scaffold = safe_scaffold(canonical, mol)
        cleaned.append(
            {
                "smiles": canonical,
                "pactivity": y,
                "document_year": year,
                "standard_type": row.get("standard_type"),
                "activity_id": row.get("activity_id"),
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "document_chembl_id": row.get("document_chembl_id"),
                "assay_chembl_id": row.get("assay_chembl_id"),
                "scaffold": scaffold,
            }
        )
        audit["eligible_measurements"] += 1
    frame = pd.DataFrame(cleaned)
    if frame.empty:
        return frame, dict(audit)
    return frame.sort_values(["smiles", "document_year", "activity_id"]).reset_index(drop=True), dict(audit)


def aggregate_compounds(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for smiles, group in frame.groupby("smiles", sort=False):
        values = group["pactivity"].to_numpy(float)
        known_years = group["document_year"].dropna().astype(int)
        records.append(
            {
                "smiles": smiles,
                "pactivity": float(np.median(values)),
                "n_measurements": int(len(group)),
                "pactivity_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "pactivity_span": float(values.max() - values.min()),
                "first_document_year": int(known_years.min()) if len(known_years) else np.nan,
                "last_document_year": int(known_years.max()) if len(known_years) else np.nan,
                "unknown_year_measurements": int(group["document_year"].isna().sum()),
                "scaffold": group["scaffold"].iloc[0],
                "pmids": ";".join(sorted(set(group.get("pmid", pd.Series(dtype=str)).dropna().astype(str)))),
            }
        )
    return pd.DataFrame(records).sort_values(["first_document_year", "smiles"]).reset_index(drop=True)


def fingerprints(
    smiles: Iterable[str], use_chirality: bool = True
) -> tuple[np.ndarray, list[DataStructs.ExplicitBitVect]]:
    generator = AllChem.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=use_chirality
    )
    arrays: list[np.ndarray] = []
    bitvectors: list[DataStructs.ExplicitBitVect] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES encountered: {smi}")
        bitvector = generator.GetFingerprint(mol)
        array = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bitvector, array)
        arrays.append(array)
        bitvectors.append(bitvector)
    return np.asarray(arrays, dtype=np.uint8), bitvectors


def safe_correlation(function, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(function(y_true, y_pred).statistic)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    slope, intercept = (float("nan"), float("nan"))
    if len(y_true) >= 3 and np.std(y_pred) > 0:
        slope, intercept = np.polyfit(y_pred, y_true, 1)
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": safe_correlation(pearsonr, y_true, y_pred),
        "spearman_rho": safe_correlation(spearmanr, y_true, y_pred),
        "mean_bias_pred_minus_obs": float(np.mean(y_pred - y_true)),
        "calibration_slope_obs_on_pred": float(slope),
        "calibration_intercept_obs_on_pred": float(intercept),
    }


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = (y_true >= threshold).astype(int)
    predictions = (y_score >= threshold).astype(int)
    result: dict[str, float | int] = {
        "threshold": threshold,
        "positives": int(labels.sum()),
        "negatives": int((1 - labels).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)) if len(np.unique(labels)) == 2 else float("nan"),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_hard_threshold": float(brier_score_loss(labels, predictions)),
    }
    if len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, y_score))
        result["auprc"] = float(average_precision_score(labels, y_score))
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")
    return result


def predict_locked(model: Any, x: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    q05s: list[np.ndarray] = []
    q95s: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size]
        tree_predictions = np.asarray([tree.predict(xb) for tree in model.estimators_], dtype=np.float32)
        means.append(tree_predictions.mean(axis=0))
        stds.append(tree_predictions.std(axis=0))
        q05s.append(np.quantile(tree_predictions, 0.05, axis=0))
        q95s.append(np.quantile(tree_predictions, 0.95, axis=0))
    return tuple(np.concatenate(parts) for parts in (means, stds, q05s, q95s))  # type: ignore[return-value]


def max_similarity(query: list[DataStructs.ExplicitBitVect], reference: list[DataStructs.ExplicitBitVect]) -> np.ndarray:
    if not reference:
        return np.full(len(query), np.nan)
    values = [max(DataStructs.BulkTanimotoSimilarity(fp, reference)) for fp in query]
    return np.asarray(values, dtype=float)


def evaluate_frame(
    frame: pd.DataFrame,
    prediction_column: str,
    cohort: str,
    thresholds: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_true = frame["pactivity"].to_numpy(float)
    y_pred = frame[prediction_column].to_numpy(float)
    reg = {"cohort": cohort, **regression_metrics(y_true, y_pred)}
    cls = [
        {"cohort": cohort, **classification_metrics(y_true, y_pred, threshold)}
        for threshold in thresholds
    ]
    return reg, cls


def scaffold_split(frame: pd.DataFrame, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_idx, test_idx = next(splitter.split(frame, groups=frame["scaffold"]))
    return train_idx, test_idx


def conformal_audit(frame: pd.DataFrame, coverage: float, seed: int) -> dict[str, Any]:
    if len(frame) < 20:
        return {"n": len(frame), "status": "insufficient_data"}
    cal_idx, test_idx = scaffold_split(frame, 0.5, seed)
    cal = frame.iloc[cal_idx]
    test = frame.iloc[test_idx]
    residuals = np.abs(cal["pactivity"].to_numpy(float) - cal["locked_prediction"].to_numpy(float))
    n = len(residuals)
    quantile_level = min(1.0, math.ceil((n + 1) * coverage) / n)
    radius = float(np.quantile(residuals, quantile_level, method="higher"))
    errors = np.abs(test["pactivity"].to_numpy(float) - test["locked_prediction"].to_numpy(float))
    return {
        "n_calibration": int(len(cal)),
        "n_test": int(len(test)),
        "nominal_coverage": coverage,
        "residual_radius_pactivity": radius,
        "empirical_test_coverage": float(np.mean(errors <= radius)),
        "split": "scaffold-grouped 50/50 calibration/test",
    }


def make_plots(predictions: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, (target, frame) in zip(axes, predictions.items()):
        cohort = frame[frame["cohort_post_publication"]]
        ax.scatter(cohort["pactivity"], cohort["locked_prediction"], c=cohort["max_similarity_historical"],
                   cmap="viridis", s=24, alpha=0.75, edgecolors="none")
        low = min(cohort["pactivity"].min(), cohort["locked_prediction"].min()) if len(cohort) else 3
        high = max(cohort["pactivity"].max(), cohort["locked_prediction"].max()) if len(cohort) else 10
        ax.plot([low, high], [low, high], "--", color="0.35", linewidth=1)
        ax.set(title=f"{target}: locked RF", xlabel="Observed pActivity", ylabel="Predicted pActivity")
    fig.savefig(output / "locked_rf_observed_vs_predicted.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, (target, frame) in zip(axes, predictions.items()):
        cohort = frame[frame["cohort_post_publication"]]
        error = np.abs(cohort["locked_prediction"] - cohort["pactivity"])
        ax.scatter(cohort["max_similarity_historical"], error, s=22, alpha=0.65)
        ax.axvline(0.6, linestyle="--", color="0.35", linewidth=1)
        ax.set(title=f"{target}: chemical-distance audit", xlabel="Max Tanimoto to historical ChEMBL", ylabel="Absolute error")
    fig.savefig(output / "error_vs_historical_similarity.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, (target, frame) in zip(axes, predictions.items()):
        cohort = frame[frame["cohort_post_publication"]]
        error = np.abs(cohort["locked_prediction"] - cohort["pactivity"])
        ax.scatter(cohort["locked_tree_std"], error, s=22, alpha=0.65)
        ax.set(title=f"{target}: RF dispersion", xlabel="Tree prediction SD", ylabel="Absolute error")
    fig.savefig(output / "error_vs_tree_dispersion.png", dpi=300)
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def write_report(
    output: Path,
    config: dict[str, Any],
    data_audit: list[dict[str, Any]],
    locked_metrics: list[dict[str, Any]],
    reference_metrics: list[dict[str, Any]],
    uncertainty: list[dict[str, Any]],
    conformal: dict[str, Any],
    model_metadata: dict[str, Any],
) -> None:
    post = [row for row in locked_metrics if row["cohort"] == "post_publication_novel"]
    strict = [row for row in locked_metrics if row["cohort"] == "strict_recent_novel"]
    lines = [
        "# EGFR/VEGFR2 predictor validation report",
        "",
        "## Evidence boundary",
        "",
        "The exact compounds used to train the locked POLYGON random-forest models are not present in the repository. "
        "Therefore, this report does not claim a reconstruction of their original cross-validation. Locked-model results are "
        "reported on exact-numeric BindingDB IC50/Kd records linked to PubMed, using compounds whose first known publication is "
        "from 2024 or later and for which no eligible measurement has an unknown publication year. The 2025-only subset is a stricter "
        "sensitivity analysis. This reduces but cannot mathematically exclude overlap with the undisclosed original training chemistry.",
        "",
        "Historical-BindingDB similarity is a chemical-domain proxy, not the true RF training-set applicability domain. The scaffold "
        "and temporal split experiments use newly trained reference RFs and validate the ECFP4-RF recipe, not the locked model weights.",
        "",
        "## Locked-model post-publication concordance",
        "",
        "| Target | Fingerprint mode | Cohort | n | R2 | RMSE | MAE | Pearson r | Spearman rho | Bias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in post + strict:
        lines.append(
            f"| {row['target']} | {row['model']} | {row['cohort']} | {row['n']} | {fmt(row['r2'])} | {fmt(row['rmse'])} | "
            f"{fmt(row['mae'])} | {fmt(row['pearson_r'])} | {fmt(row['spearman_rho'])} | "
            f"{fmt(row['mean_bias_pred_minus_obs'])} |"
        )
    lines += [
        "",
        "## Reconstructed ECFP4-RF validation",
        "",
        "| Target | Split | n | R2 | RMSE | MAE | Pearson r | Spearman rho |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reference_metrics:
        lines.append(
            f"| {row['target']} | {row['cohort']} | {row['n']} | {fmt(row['r2'])} | {fmt(row['rmse'])} | "
            f"{fmt(row['mae'])} | {fmt(row['pearson_r'])} | {fmt(row['spearman_rho'])} |"
        )
    lines += [
        "",
        "## Reliability diagnostics",
        "",
        "| Target | n | rho(tree SD, absolute error) | q05-q95 coverage | Mean historical similarity | Similarity below 0.60 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in uncertainty:
        lines.append(
            f"| {row['target']} | {row['n']} | {fmt(row['tree_std_abs_error_spearman'])} | "
            f"{fmt(row['tree_q05_q95_empirical_coverage'])} | {fmt(row['mean_max_similarity_historical'])} | "
            f"{fmt(row['rate_similarity_below_0_60'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- R2 may be negative on a temporal chemical-shift cohort even when rank correlation is positive; report both rather than hiding the shift.",
        "- RF tree quantiles are descriptive and are not calibrated prediction intervals. Scaffold-split conformal residual calibration is reported separately.",
        "- A weak relationship between tree dispersion and absolute error means tree disagreement should not be used as the sole acceptance filter.",
        "- The original POLYGON training script used achiral ECFP4, while this project's formal oracle bridge used chirality-enabled ECFP4. Both locked-model modes are reported as a sensitivity analysis.",
        "- Generated-molecule claims should remain 'predicted activity' unless supported by docking, higher-accuracy rescoring or experimental assays.",
        "",
        "## Reproducibility",
        "",
        f"- Historical reference through: {config['reference_end_year']}",
        f"- Post-publication proxy starts: {config['post_publication_start_year']}",
        f"- Strict recent sensitivity starts: {config['strict_recent_start_year']}",
        f"- Random seed: {config['random_seed']}",
        f"- scikit-learn: {sklearn.__version__}; RDKit: {rdBase.rdkitVersion}; NumPy: {np.__version__}; SciPy: {scipy.__version__}",
    ]
    for target, metadata in model_metadata.items():
        lines.append(f"- {target} model SHA256: `{metadata['sha256']}`")
    lines += ["", "Conformal calibration details are stored in `conformal_summary.json`; complete row filtering is stored in `data_audit.csv`."]
    (output / "predictor_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "config" / "predictor_validation.json")
    parser.add_argument("--output", type=Path, default=root / "results" / "predictor_validation_20260729")
    parser.add_argument("--cache", type=Path, default=root / "data" / "external" / "bindingdb")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    (args.output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    snapshots: dict[str, list[dict[str, Any]]] = {}
    all_pmids: set[str] = set()
    source_metadata: dict[str, Any] = {"data_source": config["data_source"], "snapshots": {}}
    for target, target_config in config["targets"].items():
        snapshot_path = root / target_config["bindingdb_snapshot"]
        snapshots[target] = load_bindingdb_snapshot(snapshot_path)
        all_pmids.update(
            str(row.get("pmid")) for row in snapshots[target]
            if str(row.get("pmid") or "").isdigit()
        )
        source_metadata["snapshots"][target] = {
            "path": str(snapshot_path.resolve()),
            "sha256": sha256(snapshot_path),
            "rows": len(snapshots[target]),
            "retrieved_date": "2026-07-29",
            "endpoint": f"{config['bindingdb_api_base']}?uniprot={target_config['uniprot_id']}&cutoff=1000000000&response=application/json",
        }
    pubmed_cache = args.cache / "pubmed_publication_years.json"
    pubmed_years = fetch_pubmed_years(
        all_pmids, pubmed_cache, config["pubmed_esummary_url"],
        config["request_timeout_seconds"], config["request_retries"], args.refresh,
    )
    source_metadata["pubmed_year_cache"] = {
        "path": str(pubmed_cache.resolve()),
        "sha256": sha256(pubmed_cache),
        "years_resolved": len(pubmed_years),
    }
    (args.output / "source_metadata.json").write_text(json.dumps(source_metadata, indent=2), encoding="utf-8")

    data_audit: list[dict[str, Any]] = []
    locked_regression: list[dict[str, Any]] = []
    locked_classification: list[dict[str, Any]] = []
    reference_regression: list[dict[str, Any]] = []
    reference_classification: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    similarity_bin_rows: list[dict[str, Any]] = []
    conformal_rows: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}

    for target, target_config in config["targets"].items():
        raw = snapshots[target]
        cleaned, audit = clean_bindingdb_activities(raw, pubmed_years, config)
        compounds = aggregate_compounds(cleaned)
        historical_selector = compounds["first_document_year"].le(config["reference_end_year"])
        post_selector = (
            compounds["first_document_year"].ge(config["post_publication_start_year"])
            & compounds["unknown_year_measurements"].eq(0)
        )
        strict_selector = (
            compounds["first_document_year"].ge(config["strict_recent_start_year"])
            & compounds["unknown_year_measurements"].eq(0)
        )
        audit.update(
            {
                "target": target,
                "unique_compounds": int(len(compounds)),
                "historical_compounds": int(historical_selector.sum()),
                "post_publication_novel_compounds": int(post_selector.sum()),
                "strict_recent_novel_compounds": int(strict_selector.sum()),
                "compounds_with_unknown_year_measurements": int((compounds["unknown_year_measurements"] > 0).sum()),
                "compounds_with_span_gt_1": int((compounds["pactivity_span"] > 1.0).sum()),
            }
        )
        data_audit.append(audit)
        cleaned.to_csv(args.output / f"{target.lower()}_eligible_measurements.csv", index=False, encoding="utf-8-sig")
        compounds.to_csv(args.output / f"{target.lower()}_aggregated_compounds.csv", index=False, encoding="utf-8-sig")

        x, _ = fingerprints(compounds["smiles"], use_chirality=True)
        x_training_faithful, bitvectors = fingerprints(compounds["smiles"], use_chirality=False)
        model_path = root / target_config["model"]
        with model_path.open("rb") as handle:
            locked_model = pickle.load(handle)
        root_counts = [float(tree.tree_.weighted_n_node_samples[0]) for tree in locked_model.estimators_]
        metadata[target] = {
            "path": str(model_path.resolve()),
            "sha256": sha256(model_path),
            "class": type(locked_model).__name__,
            "n_estimators": len(locked_model.estimators_),
            "n_features_in": int(locked_model.n_features_in_),
            "training_samples_retained_by_model": int(round(root_counts[0])),
            "sklearn_runtime": sklearn.__version__,
            "training_script_fingerprint": "Morgan radius 2, 2048 bits, useChirality=False (RDKit default)",
            "formal_project_fingerprint": "Morgan radius 2, 2048 bits, useChirality=True",
        }
        mean, std, q05, q95 = predict_locked(locked_model, x)
        compounds["locked_prediction"] = mean
        compounds["locked_tree_std"] = std
        compounds["locked_tree_q05"] = q05
        compounds["locked_tree_q95"] = q95
        faithful_mean, _, _, _ = predict_locked(locked_model, x_training_faithful)
        compounds["locked_prediction_training_faithful"] = faithful_mean
        compounds["operational_vs_training_faithful_abs_diff"] = np.abs(mean - faithful_mean)
        compounds["cohort_historical"] = historical_selector
        compounds["cohort_post_publication"] = post_selector
        compounds["cohort_strict_recent"] = strict_selector

        historical_mask = compounds["cohort_historical"].to_numpy(bool)
        post_mask = compounds["cohort_post_publication"].to_numpy(bool)
        compounds["max_similarity_historical"] = np.nan
        compounds.loc[post_mask, "max_similarity_historical"] = max_similarity(
            [bitvectors[i] for i in np.flatnonzero(post_mask)],
            [bitvectors[i] for i in np.flatnonzero(historical_mask)],
        )

        cohorts = {
            "post_publication_novel": compounds[compounds["cohort_post_publication"]].copy(),
            "strict_recent_novel": compounds[compounds["cohort_strict_recent"]].copy(),
            "all_eligible_diagnostic": compounds.copy(),
        }
        locked_modes = {
            "locked_operational_chiral": "locked_prediction",
            "locked_training_faithful_achiral": "locked_prediction_training_faithful",
        }
        for model_label, prediction_column in locked_modes.items():
            for cohort_name, cohort_frame in cohorts.items():
                if cohort_frame.empty:
                    continue
                reg, cls = evaluate_frame(cohort_frame, prediction_column, cohort_name, config["activity_thresholds"])
                locked_regression.append({"target": target, "model": model_label, **reg})
                locked_classification.extend({"target": target, "model": model_label, **row} for row in cls)

        post = cohorts["post_publication_novel"]
        absolute_error = np.abs(post["locked_prediction"].to_numpy(float) - post["pactivity"].to_numpy(float))
        uncertainty_rows.append(
            {
                "target": target,
                "n": int(len(post)),
                "tree_std_abs_error_spearman": safe_correlation(spearmanr, post["locked_tree_std"].to_numpy(float), absolute_error),
                "tree_q05_q95_empirical_coverage": float(np.mean((post["pactivity"] >= post["locked_tree_q05"]) & (post["pactivity"] <= post["locked_tree_q95"]))),
                "mean_tree_q05_q95_width": float(np.mean(post["locked_tree_q95"] - post["locked_tree_q05"])),
                "mean_max_similarity_historical": float(post["max_similarity_historical"].mean()),
                "median_max_similarity_historical": float(post["max_similarity_historical"].median()),
                "rate_similarity_below_0_60": float((post["max_similarity_historical"] < 0.60).mean()),
                "mean_operational_vs_training_faithful_abs_diff": float(post["operational_vs_training_faithful_abs_diff"].mean()),
                "max_operational_vs_training_faithful_abs_diff": float(post["operational_vs_training_faithful_abs_diff"].max()),
            }
        )
        for label, lower, upper in (("lt_0.4", -np.inf, 0.4), ("0.4_to_0.6", 0.4, 0.6), ("0.6_to_0.8", 0.6, 0.8), ("ge_0.8", 0.8, np.inf)):
            subset = post[(post["max_similarity_historical"] >= lower) & (post["max_similarity_historical"] < upper)]
            if len(subset):
                similarity_bin_rows.append({"target": target, "similarity_bin": label, **regression_metrics(subset["pactivity"].to_numpy(float), subset["locked_prediction"].to_numpy(float))})

        conformal_rows[target] = conformal_audit(post, config["conformal_coverage"], config["random_seed"])

        historical = compounds[compounds["cohort_historical"]].reset_index(drop=True)
        temporal = compounds[compounds["cohort_post_publication"]].reset_index(drop=True)
        x_historical, _ = fingerprints(historical["smiles"], use_chirality=False)
        train_idx, test_idx = scaffold_split(historical, config["scaffold_test_fraction"], config["random_seed"])
        reference_scaffold = RandomForestRegressor(
            n_estimators=config["reference_rf_estimators"], random_state=config["random_seed"], n_jobs=-1
        )
        reference_scaffold.fit(x_historical[train_idx], historical.iloc[train_idx]["pactivity"].to_numpy(float))
        scaffold_prediction = reference_scaffold.predict(x_historical[test_idx])
        scaffold_test = historical.iloc[test_idx].copy()
        scaffold_test["reference_prediction"] = scaffold_prediction
        reg, cls = evaluate_frame(scaffold_test, "reference_prediction", "reconstructed_scaffold_split", config["activity_thresholds"])
        reference_regression.append({"target": target, "model": "reconstructed_ecfp4_rf", **reg})
        reference_classification.extend({"target": target, "model": "reconstructed_ecfp4_rf", **row} for row in cls)

        reference_temporal = RandomForestRegressor(
            n_estimators=config["reference_rf_estimators"], random_state=config["random_seed"], n_jobs=-1
        )
        reference_temporal.fit(x_historical, historical["pactivity"].to_numpy(float))
        temporal_indices = np.flatnonzero(post_mask)
        temporal_prediction = reference_temporal.predict(x_training_faithful[temporal_indices])
        temporal["reference_prediction"] = temporal_prediction
        reg, cls = evaluate_frame(temporal, "reference_prediction", "reconstructed_temporal_split", config["activity_thresholds"])
        reference_regression.append({"target": target, "model": "reconstructed_ecfp4_rf", **reg})
        reference_classification.extend({"target": target, "model": "reconstructed_ecfp4_rf", **row} for row in cls)

        compounds.to_csv(args.output / f"{target.lower()}_locked_predictions.csv", index=False, encoding="utf-8-sig")
        prediction_frames[target] = compounds

    pd.DataFrame(data_audit).to_csv(args.output / "data_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(locked_regression).to_csv(args.output / "locked_model_regression_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(locked_classification).to_csv(args.output / "locked_model_classification_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_regression).to_csv(args.output / "reference_model_regression_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_classification).to_csv(args.output / "reference_model_classification_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(uncertainty_rows).to_csv(args.output / "uncertainty_and_domain_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(similarity_bin_rows).to_csv(args.output / "historical_similarity_bin_metrics.csv", index=False, encoding="utf-8-sig")
    (args.output / "conformal_summary.json").write_text(json.dumps(conformal_rows, indent=2), encoding="utf-8")
    (args.output / "locked_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    make_plots(prediction_frames, args.output)
    write_report(args.output, config, data_audit, locked_regression, reference_regression, uncertainty_rows, conformal_rows, metadata)
    print(f"Validation complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
