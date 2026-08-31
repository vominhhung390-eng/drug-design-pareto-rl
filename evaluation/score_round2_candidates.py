#!/usr/bin/env python
"""Candidate-only EGFR/VEGFR2 scoring with mandatory domain flags."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator


def features_and_fps(smiles: pd.Series):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    features = np.empty((len(smiles), 2058), dtype=np.float32)
    fps = []
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {index}: {smi}")
        fp = generator.GetFingerprint(mol)
        fps.append(fp)
        DataStructs.ConvertToNumpyArray(fp, features[index, :2048])
        features[index, 2048:] = (
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            Lipinski.FractionCSP3(mol),
            Lipinski.HeavyAtomCount(mol),
            Chem.GetFormalCharge(mol),
        )
    return features, fps


def reference_fps(smiles: pd.Series):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    return [generator.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in smiles]


def max_similarity(queries, references) -> np.ndarray:
    return np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(query, references)) for query in queries]
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if args.smiles_column not in frame.columns:
        raise ValueError(f"Missing SMILES column: {args.smiles_column}")
    smiles = frame[args.smiles_column].astype(str)
    run_dir = root / "results" / "predictor_retraining_round2_20260730"

    multitask_models = (
        run_dir
        / "final_candidates"
        / "multitask_dmpnn"
        / "research_through_2023"
    )
    wrapper = root / "evaluation" / "run_chemprop_utf8.py"
    with tempfile.TemporaryDirectory(prefix="round2_predict_") as temporary:
        temporary = Path(temporary)
        chemprop_input = temporary / "input.csv"
        chemprop_output = temporary / "output.csv"
        pd.DataFrame({"smiles": smiles}).to_csv(chemprop_input, index=False)
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "RICH_FORCE_TERMINAL": "false",
            }
        )
        subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                str(wrapper),
                "predict",
                "-q",
                "--test-path",
                str(chemprop_input),
                "--smiles-columns",
                "smiles",
                "--model-paths",
                str(multitask_models),
                "--preds-path",
                str(chemprop_output),
                "--uncertainty-method",
                "ensemble",
                "--batch-size",
                "256",
                "--num-workers",
                "0",
                "--accelerator",
                "gpu",
                "--devices",
                "1",
            ],
            check=True,
            env=environment,
        )
        chemprop_predictions = pd.read_csv(chemprop_output)

    features, query_fps = features_and_fps(smiles)
    vegfr2_model_dir = (
        run_dir
        / "final_candidates"
        / "vegfr2"
        / "research_through_2023"
    )
    vegfr2_models = [
        joblib.load(path)
        for path in sorted(vegfr2_model_dir.glob("model_seed_*.joblib"))
    ]
    if len(vegfr2_models) != 5:
        raise RuntimeError(f"Expected 5 VEGFR2 models, found {len(vegfr2_models)}")
    vegfr2_matrix = np.vstack([model.predict(features) for model in vegfr2_models])

    egfr_reference = pd.read_csv(
        run_dir / "data" / "egfr" / "development_through_2023.csv"
    )
    vegfr2_reference = pd.read_csv(
        run_dir / "data" / "vegfr2" / "development_through_2023.csv"
    )
    egfr_similarity = max_similarity(
        query_fps, reference_fps(egfr_reference["smiles"])
    )
    vegfr2_similarity = max_similarity(
        query_fps, reference_fps(vegfr2_reference["smiles"])
    )

    output = frame.copy()
    output["candidate_egfr_pactivity"] = chemprop_predictions["egfr_pactivity"]
    output["candidate_egfr_ensemble_unc"] = chemprop_predictions[
        "egfr_pactivity_unc"
    ]
    output["candidate_vegfr2_pactivity"] = vegfr2_matrix.mean(axis=0)
    output["candidate_vegfr2_ensemble_std"] = vegfr2_matrix.std(axis=0, ddof=1)
    output["egfr_max_tanimoto_to_development"] = egfr_similarity
    output["vegfr2_max_tanimoto_to_development"] = vegfr2_similarity
    output["within_joint_applicability_domain"] = (
        (egfr_similarity >= 0.60) & (vegfr2_similarity >= 0.60)
    )
    output["formal_use_allowed"] = False
    output["use_policy"] = (
        "candidate diagnostics only; not allowed as formal reward or paper oracle"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(output)} candidate scores to {args.output}")


if __name__ == "__main__":
    main()
