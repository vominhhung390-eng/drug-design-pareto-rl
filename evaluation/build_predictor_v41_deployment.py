#!/usr/bin/env python
"""Build and verify the frozen V4.1 VEGFR2 deployment classifier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from benchmark_predictor_round2 import featurize


ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "results"
    / "predictor_retraining_v3_20260731"
    / "data"
    / "vegfr2"
    / "single_protein_wt_or_unspecified"
    / "development_through_2023.csv"
)
OUT = ROOT / "results" / "predictor_v41_20260802" / "deployment"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    frame = frame[(frame.pactivity <= 5.75) | (frame.pactivity >= 7.25)].reset_index(
        drop=True
    )
    labels = (frame.pactivity >= 7.25).astype(int).to_numpy()
    model = ExtraTreesClassifier(
        n_estimators=2000,
        max_features="sqrt",
        min_samples_leaf=1,
        criterion="entropy",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(featurize(frame.smiles), labels)
    model_path = OUT / "vegfr2_extratrees.pkl"
    joblib.dump(model, model_path, compress=3)
    metadata = {
        "schema_version": "predictor-v4.1-deployment",
        "target": "VEGFR2",
        "task": "high-confidence binary classification",
        "inactive_pactivity_max": 5.75,
        "active_pactivity_min": 7.25,
        "training_rows": int(len(frame)),
        "active_rows": int(labels.sum()),
        "inactive_rows": int((1 - labels).sum()),
        "feature_contract": "Morgan radius=2, 2048 bits, chirality plus 10 descriptors",
        "estimator": "ExtraTreesClassifier(n_estimators=2000, criterion=entropy, class_weight=balanced, random_state=42)",
        "training_data": str(DATA),
        "training_data_sha256": sha256(DATA),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
