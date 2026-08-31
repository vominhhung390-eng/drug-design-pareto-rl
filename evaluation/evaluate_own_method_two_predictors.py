#!/usr/bin/env python
"""Blindly rescore formal own-method outputs with original RF and V4.1.

The molecular generation runs are not repeated.  The same frozen V4-B outputs
from seeds 42--51 are evaluated under both predictor systems so that predictor
effects are not confounded with generation stochasticity.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesClassifier

from benchmark_predictor_round2 import featurize


ROOT = Path(__file__).resolve().parents[1]
OWN = ROOT / "results" / "own_method_v4" / "common_seeds_42_51_10240"
V3 = ROOT / "results" / "predictor_retraining_v3_20260731" / "data"
V41 = ROOT / "results" / "predictor_v41_20260802"
EGFR_MODELS = V41 / "egfr_bindingdb_external_v2"
OUT = ROOT / "results" / "own_method_v4" / "predictor_crosscheck_20260802"
WRAPPER = ROOT / "evaluation" / "run_chemprop_utf8.py"
SEEDS = tuple(range(42, 52))


def load_formal_outputs() -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        run = OWN / f"v4_b_raw_mean_seed{seed}"
        standardized = pd.read_csv(run / "evaluation" / "standardized_molecules.csv")
        quality = pd.read_csv(
            run
            / "evaluation"
            / "quality_constrained"
            / "quality_annotated_molecules.csv"
        )[["smiles", "quality_pass"]].drop_duplicates("smiles")
        frame = standardized.merge(quality, on="smiles", how="left", validate="one_to_one")
        frame["seed"] = seed
        frame["method"] = "V4-B"
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    output["quality_pass"] = output.quality_pass.fillna(False).astype(bool)
    return output


def chemprop_predict(input_csv: Path, model_dir: Path, output_csv: Path, morgan: bool) -> np.ndarray:
    if not output_csv.exists():
        command = [
            sys.executable,
            "-X",
            "utf8",
            str(WRAPPER),
            "predict",
            "--test-path",
            str(input_csv),
            "--smiles-columns",
            "smiles",
            "--model-paths",
            str(model_dir),
            "--preds-path",
            str(output_csv),
            "--batch-size",
            "512",
            "--num-workers",
            "0",
            "--accelerator",
            "gpu",
            "--devices",
            "1",
        ]
        if morgan:
            command += ["--molecule-featurizers", "morgan_binary"]
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "RICH_FORCE_TERMINAL": "false",
            }
        )
        log_path = output_csv.with_suffix(".log")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if process.returncode:
            raise RuntimeError(log_path.read_text(encoding="utf-8", errors="replace")[-8000:])
    frame = pd.read_csv(output_csv)
    return frame.active_label.to_numpy(float)


def egfr_similarity_probability(smiles: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    reference = pd.read_csv(
        V3
        / "egfr"
        / "single_protein_assay_ge10"
        / "development_through_2023.csv"
    )
    reference = reference[
        (reference.pactivity <= 5.5) | (reference.pactivity >= 7.5)
    ].reset_index(drop=True)
    labels = (reference.pactivity >= 7.5).astype(int).to_numpy()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048)
    reference_fp = [
        generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in reference.smiles
    ]
    probability = np.empty(len(smiles), dtype=float)
    max_similarity = np.empty(len(smiles), dtype=float)
    for index, value in enumerate(smiles):
        fp = generator.GetFingerprint(Chem.MolFromSmiles(value))
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fp, reference_fp), dtype=float
        )
        nearest = np.argpartition(similarities, -20)[-20:]
        weights = np.maximum(similarities[nearest], 1e-6) ** 6
        probability[index] = np.average(labels[nearest], weights=weights)
        max_similarity[index] = similarities[nearest].max()
        if (index + 1) % 10000 == 0:
            print(f"EGFR similarity {index + 1}/{len(smiles)}", flush=True)
    return probability, max_similarity


def vegfr2_probability(smiles: pd.Series) -> np.ndarray:
    train = pd.read_csv(
        V3
        / "vegfr2"
        / "single_protein_wt_or_unspecified"
        / "development_through_2023.csv"
    )
    train = train[(train.pactivity <= 5.75) | (train.pactivity >= 7.25)].reset_index(
        drop=True
    )
    labels = (train.pactivity >= 7.25).astype(int).to_numpy()
    model = ExtraTreesClassifier(
        n_estimators=2000,
        max_features="sqrt",
        min_samples_leaf=1,
        criterion="entropy",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(featurize(train.smiles), labels)
    output = np.empty(len(smiles), dtype=float)
    chunk_size = 4096
    for start in range(0, len(smiles), chunk_size):
        stop = min(start + chunk_size, len(smiles))
        output[start:stop] = model.predict_proba(featurize(smiles.iloc[start:stop]))[:, 1]
        print(f"VEGFR2 forest {stop}/{len(smiles)}", flush=True)
    return output


def safe_correlation(function, a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(function(a.to_numpy(float), b.to_numpy(float)).statistic)


def pareto_front_2d(points: np.ndarray) -> np.ndarray:
    """Return the nondominated front for a two-objective maximization in O(n log n)."""
    points = np.asarray(points, dtype=float)
    if not len(points):
        return points.reshape(0, 2)
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    keep = []
    best_y = -np.inf
    for index in order:
        y_value = points[index, 1]
        if y_value > best_y:
            keep.append(index)
            best_y = y_value
    return points[np.asarray(keep, dtype=int)]


def unit_hypervolume(points: np.ndarray) -> float:
    front = pareto_front_2d(np.clip(np.asarray(points, dtype=float), 0.0, 1.0))
    front = front[np.all(front > 0.0, axis=1)]
    if not len(front):
        return 0.0
    front = front[np.argsort(front[:, 0])]
    result = 0.0
    best_y = 0.0
    for x_value, y_value in front[::-1]:
        if y_value > best_y:
            result += x_value * (y_value - best_y)
            best_y = y_value
    return float(result)


def summarize(frame: pd.DataFrame, label: str) -> dict:
    original_min = frame[["egfr", "vegfr2"]].min(axis=1)
    v41_min = frame[["v41_egfr_probability", "v41_vegfr2_probability"]].min(axis=1)
    original_dual6 = (frame.egfr >= 6.0) & (frame.vegfr2 >= 6.0)
    original_dual7 = (frame.egfr >= 7.0) & (frame.vegfr2 >= 7.0)
    v41_egfr_hit = frame.v41_egfr_probability >= 0.595
    v41_vegfr2_hit = frame.v41_vegfr2_probability >= 0.5
    v41_dual = v41_egfr_hit & v41_vegfr2_hit
    top_n = min(100, len(frame))
    original_top = original_min.nlargest(top_n).index
    v41_top = v41_min.nlargest(top_n).index
    return {
        "subset": label,
        "n": int(len(frame)),
        "original_egfr_mean": float(frame.egfr.mean()),
        "original_vegfr2_mean": float(frame.vegfr2.mean()),
        "original_min_mean": float(original_min.mean()),
        "original_min_p90": float(original_min.quantile(0.90)),
        "original_dual6_rate": float(original_dual6.mean()),
        "original_dual6_count": int(original_dual6.sum()),
        "original_dual7_rate": float(original_dual7.mean()),
        "original_dual7_count": int(original_dual7.sum()),
        "v41_egfr_probability_mean": float(frame.v41_egfr_probability.mean()),
        "v41_vegfr2_probability_mean": float(frame.v41_vegfr2_probability.mean()),
        "v41_min_probability_mean": float(v41_min.mean()),
        "v41_min_probability_p90": float(v41_min.quantile(0.90)),
        "v41_egfr_hit_rate": float(v41_egfr_hit.mean()),
        "v41_egfr_hit_count": int(v41_egfr_hit.sum()),
        "v41_vegfr2_hit_rate": float(v41_vegfr2_hit.mean()),
        "v41_vegfr2_hit_count": int(v41_vegfr2_hit.sum()),
        "v41_dual_hit_rate": float(v41_dual.mean()),
        "v41_dual_hit_count": int(v41_dual.sum()),
        "consensus_original7_v41_dual_rate": float((original_dual7 & v41_dual).mean()),
        "consensus_original7_v41_dual_count": int((original_dual7 & v41_dual).sum()),
        "v41_support_among_original_dual7": (
            float(v41_dual[original_dual7].mean()) if original_dual7.any() else None
        ),
        "original7_support_among_v41_dual": (
            float(original_dual7[v41_dual].mean()) if v41_dual.any() else None
        ),
        "v41_dual_rate_in_original_top100": float(v41_dual.loc[original_top].mean()),
        "original_dual7_rate_in_v41_top100": float(original_dual7.loc[v41_top].mean()),
        "egfr_spearman": safe_correlation(spearmanr, frame.egfr, frame.v41_egfr_probability),
        "vegfr2_spearman": safe_correlation(
            spearmanr, frame.vegfr2, frame.v41_vegfr2_probability
        ),
        "minimum_score_spearman": safe_correlation(spearmanr, original_min, v41_min),
        "egfr_pearson": safe_correlation(pearsonr, frame.egfr, frame.v41_egfr_probability),
        "vegfr2_pearson": safe_correlation(
            pearsonr, frame.vegfr2, frame.v41_vegfr2_probability
        ),
        "v41_probability_hypervolume": unit_hypervolume(
            frame[["v41_egfr_probability", "v41_vegfr2_probability"]].to_numpy(float)
        ),
        "v41_pareto_size": int(
            len(
                pareto_front_2d(
                    frame[["v41_egfr_probability", "v41_vegfr2_probability"]].to_numpy(
                        float
                    )
                )
            )
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    occurrences = load_formal_outputs()
    unique = occurrences.drop_duplicates("smiles").reset_index(drop=True).copy()
    input_csv = OUT / "unique_smiles.csv"
    unique[["smiles"]].to_csv(input_csv, index=False)
    print(
        f"formal occurrences={len(occurrences)} unique_smiles={len(unique)}",
        flush=True,
    )

    prediction_columns = [
        "smiles",
        "v41_egfr_dmpnn",
        "v41_egfr_morgan",
        "v41_egfr_knn",
        "v41_egfr_max_train_similarity",
        "v41_egfr_probability",
        "v41_vegfr2_probability",
    ]
    prediction_cache = OUT / "unique_predictions.csv"
    if prediction_cache.exists():
        cached = pd.read_csv(prediction_cache)
        if len(cached) != len(unique) or not set(prediction_columns).issubset(cached):
            raise RuntimeError("Existing unique_predictions.csv is incomplete or mismatched")
        unique = cached
        print("Reusing complete unique prediction cache", flush=True)
    else:
        dmpnn = chemprop_predict(
            input_csv, EGFR_MODELS / "dmpnn", OUT / "egfr_dmpnn_predictions.csv", False
        )
        print("EGFR D-MPNN prediction complete", flush=True)
        morgan = chemprop_predict(
            input_csv,
            EGFR_MODELS / "dmpnn_morgan",
            OUT / "egfr_morgan_predictions.csv",
            True,
        )
        print("EGFR Morgan-DMPNN prediction complete", flush=True)
        knn, max_similarity = egfr_similarity_probability(unique.smiles)
        unique["v41_egfr_dmpnn"] = dmpnn
        unique["v41_egfr_morgan"] = morgan
        unique["v41_egfr_knn"] = knn
        unique["v41_egfr_max_train_similarity"] = max_similarity
        unique["v41_egfr_probability"] = 0.7 * dmpnn + 0.1 * morgan + 0.2 * knn
        unique["v41_vegfr2_probability"] = vegfr2_probability(unique.smiles)
        unique.to_csv(prediction_cache, index=False)

    scored = occurrences.merge(
        unique[prediction_columns], on="smiles", how="left", validate="many_to_one"
    )
    scored.to_csv(OUT / "scored_occurrences.csv", index=False)

    rows = []
    for seed, frame in scored.groupby("seed"):
        rows.append({"seed": int(seed), **summarize(frame, "all")})
        quality_frame = frame[frame.quality_pass].copy()
        rows.append({"seed": int(seed), **summarize(quality_frame, "quality_pass")})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "per_seed_metrics.csv", index=False)

    pooled = scored.drop_duplicates("smiles").reset_index(drop=True)
    pooled["original_min_activity"] = pooled[["egfr", "vegfr2"]].min(axis=1)
    pooled["v41_min_probability"] = pooled[
        ["v41_egfr_probability", "v41_vegfr2_probability"]
    ].min(axis=1)
    pooled["consensus_rank_score"] = (
        pooled.original_min_activity.rank(pct=True)
        + pooled.v41_min_probability.rank(pct=True)
    ) / 2.0
    consensus_mask = (
        pooled.quality_pass
        & pooled.egfr.ge(7.0)
        & pooled.vegfr2.ge(7.0)
        & pooled.v41_egfr_probability.ge(0.595)
        & pooled.v41_vegfr2_probability.ge(0.5)
    )
    pooled.loc[consensus_mask].sort_values(
        ["consensus_rank_score", "v41_min_probability", "original_min_activity"],
        ascending=False,
    ).to_csv(OUT / "consensus_quality_candidates.csv", index=False)
    result = {
        "method": "V4-B raw mean",
        "protocol": "blind rescoring of frozen generated molecules; no regeneration",
        "seeds": list(SEEDS),
        "budget_per_seed": 10240,
        "occurrences": int(len(scored)),
        "unique_smiles": int(len(pooled)),
        "pooled_unique_all": summarize(pooled, "all"),
        "pooled_unique_quality_pass": summarize(
            pooled[pooled.quality_pass].copy(), "quality_pass"
        ),
        "per_seed_mean": {
            subset: {
                column: float(group[column].mean())
                for column in (
                    "n",
                    "original_dual6_rate",
                    "original_dual7_rate",
                    "v41_egfr_hit_rate",
                    "v41_vegfr2_hit_rate",
                    "v41_dual_hit_rate",
                    "consensus_original7_v41_dual_rate",
                    "v41_support_among_original_dual7",
                    "v41_dual_rate_in_original_top100",
                    "egfr_spearman",
                    "vegfr2_spearman",
                    "minimum_score_spearman",
                )
            }
            for subset, group in per_seed.groupby("subset")
        },
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
