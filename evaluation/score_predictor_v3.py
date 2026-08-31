#!/usr/bin/env python
"""Score SMILES with the frozen V3 candidate predictor and applicability domain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from benchmark_predictor_round2 import featurize


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "results" / "predictor_retraining_v3_20260731" / "final_candidate" / "deployment_all_available"
RDLogger.DisableLog("rdApp.error")


def ad_label(similarity: float, tree_std: float) -> str:
    if similarity >= 0.45 and tree_std <= 0.75:
        return "high"
    if similarity >= 0.30 and tree_std <= 1.00:
        return "medium"
    return "low"


def forest_mean_std(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree_predictions = np.asarray([tree.predict(x) for tree in model.estimators_])
    return tree_predictions.mean(axis=0), tree_predictions.std(axis=0)


def max_similarity(fp, references) -> float:
    return float(max(DataStructs.BulkTanimotoSimilarity(fp, references)))


def hybrid_prediction(model, references, targets, fps, x, k=20, power=3):
    tree_mean, tree_std = forest_mean_std(model, x)
    knn, similarity = [], []
    targets = np.asarray(targets)
    for fp in fps:
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, references), dtype=float)
        order = np.argsort(sims)[::-1][:k]
        weights = np.maximum(sims[order], 1e-6) ** power
        knn.append(float(np.average(targets[order], weights=weights)))
        similarity.append(float(sims[order[0]]))
    return 0.5 * tree_mean + 0.5 * np.asarray(knn), tree_std, np.asarray(similarity)


def score_file(input_csv: Path, output_csv: Path, smiles_column: str) -> None:
    frame = pd.read_csv(input_csv)
    if smiles_column not in frame.columns:
        raise ValueError(f"Missing SMILES column: {smiles_column}")
    mols = [Chem.MolFromSmiles(str(s)) for s in frame[smiles_column]]
    valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
    invalid = [i for i, mol in enumerate(mols) if mol is None]
    if not valid_indices:
        raise ValueError("No valid SMILES were found")
    valid_mols = [mols[i] for i in valid_indices]
    valid_canonical = [Chem.MolToSmiles(mol, canonical=True) for mol in valid_mols]
    x = featurize(pd.Series(valid_canonical))
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fpgen.GetFingerprint(mol) for mol in valid_mols]

    egfr = joblib.load(MODEL_DIR / "egfr_v3_consensus.joblib")
    egfr_chembl, egfr_tree_std, egfr_similarity = hybrid_prediction(
        egfr["chembl_regressor"], egfr["chembl_knn_fingerprints"], egfr["chembl_knn_targets"], fps, x
    )
    egfr_cross, egfr_cross_std, egfr_cross_similarity = hybrid_prediction(
        egfr["cross_source_regressor"], egfr["cross_source_knn_fingerprints"], egfr["cross_source_knn_targets"], fps, x
    )
    egfr_prediction = 0.5 * egfr_chembl + 0.5 * egfr_cross
    egfr_disagreement = np.abs(egfr_chembl - egfr_cross)

    veg = joblib.load(MODEL_DIR / "vegfr2_v3_candidate.joblib")
    veg_prediction, veg_tree_std = forest_mean_std(veg["regressor"], x)
    veg_probability = veg["classifier"].predict_proba(x)[:, 1]
    veg_similarity = [max_similarity(fp, veg["ad_fingerprints"]) for fp in fps]

    def numeric_full(values) -> np.ndarray:
        output = np.full(len(frame), np.nan, dtype=float)
        output[valid_indices] = values
        return output

    def label_full(values, invalid_label: str = "invalid_smiles") -> np.ndarray:
        output = np.full(len(frame), invalid_label, dtype=object)
        output[valid_indices] = values
        return output

    result = frame.copy()
    result["predictor_input_valid"] = False
    result.loc[valid_indices, "predictor_input_valid"] = True
    result["canonical_smiles_v3"] = label_full(valid_canonical, "")
    result["egfr_chembl_score"] = numeric_full(egfr_chembl)
    result["egfr_cross_source_score"] = numeric_full(egfr_cross)
    result["egfr_predicted_pIC50"] = numeric_full(egfr_prediction)
    result["egfr_ranking_score"] = numeric_full(egfr_prediction)
    result["egfr_model_disagreement"] = numeric_full(egfr_disagreement)
    result["egfr_tree_std"] = numeric_full(egfr_tree_std)
    result["egfr_max_train_similarity"] = numeric_full(egfr_similarity)
    result["egfr_ad"] = label_full([ad_label(s, u) for s, u in zip(egfr_similarity, egfr_tree_std)])
    cross_ad = [ad_label(s, u) for s, u in zip(egfr_cross_similarity, egfr_cross_std)]
    consensus_supported = [
        d <= 0.25 and first in {"high", "medium"} and second in {"high", "medium"}
        for d, first, second in zip(egfr_disagreement, [ad_label(s, u) for s, u in zip(egfr_similarity, egfr_tree_std)], cross_ad)
    ]
    result["egfr_evidence_tier"] = label_full(["consensus_supported" if ok else "review" for ok in consensus_supported])
    result["egfr_absolute_value_validated"] = False
    result["vegfr2_predicted_pIC50"] = numeric_full(veg_prediction)
    result["vegfr2_active_probability_pIC50_ge_6_5"] = numeric_full(veg_probability)
    result["vegfr2_tree_std"] = numeric_full(veg_tree_std)
    result["vegfr2_max_train_similarity"] = numeric_full(veg_similarity)
    result["vegfr2_ad"] = label_full([ad_label(s, u) for s, u in zip(veg_similarity, veg_tree_std)])
    valid_joint = ["usable" if ok and v in {"high", "medium"} else "review" for ok, v in zip(consensus_supported, result.loc[valid_indices, "vegfr2_ad"])]
    result["joint_ad"] = label_full(valid_joint)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(json.dumps({
        "rows": len(result), "valid_smiles": len(valid_indices), "invalid_smiles": len(invalid),
        "output": str(output_csv), "joint_ad_counts": result["joint_ad"].value_counts().to_dict()
    }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    args = parser.parse_args()
    score_file(args.input, args.output, args.smiles_column)


if __name__ == "__main__":
    main()
