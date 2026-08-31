#!/usr/bin/env python
"""Train the four shared RF ranking oracles used by every method.

PARP1/BRD4 use the packaged ChEMBL 37 formal training tables.  The exact
historical EGFR/VEGFR2 rows were not recovered; their packaged BindingDB API
snapshots are therefore available only behind an explicit opt-in flag.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")
ROOT = Path(__file__).resolve().parents[1]
FIRST_PAIR = ROOT / "data/predictor_target_pairs/01_EGFR_VEGFR2_第一组_恢复相关数据_NOT_EXACT_ORIGINAL"
SECOND_PAIR = ROOT / "data/predictor_target_pairs/02_PARP1_BRD4_第二组_当前正式预测器数据_ChEMBL37"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if fragments:
        mol = max(fragments, key=lambda value: value.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def parse_nm(value: object) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return None
    parsed = float(match.group(0))
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def load_recovered_bindingdb(path: Path, exact_query: str) -> tuple[pd.DataFrame, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["getLindsByUniprotsResponse"]["affinities"]
    values: list[dict[str, object]] = []
    for row in rows:
        if str(row.get("query", "")) != exact_query:
            continue
        if str(row.get("affinity_type", "")) not in {"IC50", "Kd"}:
            continue
        nm = parse_nm(row.get("affinity"))
        smiles = canonical(str(row.get("smile", "")))
        if nm is None or smiles is None:
            continue
        values.append({"smiles": smiles, "pactivity": -math.log10(nm * 1e-9)})
    frame = pd.DataFrame(values)
    if frame.empty:
        raise RuntimeError(f"No usable IC50/Kd records in {path}")
    # Matches the legacy intent: the smallest available nM value per molecule.
    frame = frame.groupby("smiles", as_index=False)["pactivity"].max()
    return frame, {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
        "provenance": "RECOVERED_RELATED_NOT_EXACT_ORIGINAL",
        "filter": f"query == {exact_query!r}; affinity_type in [IC50, Kd]; minimum nM per canonical molecule",
        "rows": int(len(frame)),
    }


def load_formal_csv(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path, usecols=["smiles", "pactivity"])
    frame["smiles"] = frame["smiles"].map(canonical)
    frame["pactivity"] = pd.to_numeric(frame["pactivity"], errors="coerce")
    frame = frame.dropna(subset=["smiles", "pactivity"])
    frame = frame.groupby("smiles", as_index=False)["pactivity"].mean()
    return frame, {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
        "provenance": "FORMAL_PACKAGED_TRAINING_TABLE",
        "rows": int(len(frame)),
    }


def fingerprints(smiles: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    matrix = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Unexpected invalid canonical SMILES: {value}")
        fp = generator.GetFingerprint(mol)
        matrix[index, list(fp.GetOnBits())] = 1
    return matrix


def fit_target(target: str, frame: pd.DataFrame, output: Path) -> tuple[Path, dict[str, object]]:
    x = fingerprints(frame["smiles"].astype(str).tolist())
    y = frame["pactivity"].to_numpy(dtype=float)
    model = RandomForestRegressor(
        n_estimators=1000,
        max_features=1.0,
        min_samples_leaf=1,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(x, y)
    prediction = model.predict(x)
    path = output / f"target_{target}_model.pkl"
    with path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path, {
        "target": target,
        "training_rows": int(len(frame)),
        "training_rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "training_r2": float(r2_score(y, prediction)),
        "model_sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "models/reproduced_oracles")
    parser.add_argument("--allow-recovered-egfr-vegfr2", action="store_true")
    args = parser.parse_args()
    if not args.allow_recovered_egfr_vegfr2:
        raise SystemExit(
            "BLOCKED: exact historical EGFR/VEGFR2 training rows are unavailable. "
            "Use --allow-recovered-egfr-vegfr2 only if the documented BindingDB recovery is acceptable."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    specs = {
        "EGFR": lambda: load_recovered_bindingdb(
            FIRST_PAIR / "EGFR_P00533_BindingDB_API_snapshot_20260712.json",
            "Epidermal growth factor receptor",
        ),
        "VEGFR2": lambda: load_recovered_bindingdb(
            FIRST_PAIR / "VEGFR2_P35968_BindingDB_API_snapshot_20260712.json",
            "Vascular endothelial growth factor receptor 2",
        ),
        "PARP1": lambda: load_formal_csv(
            SECOND_PAIR / "PARP1_CHEMBL3105_train_through_2023_n2538.csv"
        ),
        "BRD4": lambda: load_formal_csv(
            SECOND_PAIR / "BRD4_CHEMBL1163125_train_through_2023_n5245.csv"
        ),
    }
    rows: list[dict[str, object]] = []
    sources: dict[str, object] = {}
    model_hashes: list[dict[str, str]] = []
    for target, loader in specs.items():
        frame, source = loader()
        model_path, metrics = fit_target(target, frame, args.output)
        rows.append(metrics)
        sources[target] = source
        model_hashes.append({"file": model_path.name, "sha256": sha256(model_path)})
        print(f"trained={target} rows={len(frame)} model={model_path}", flush=True)

    pd.DataFrame(rows).to_csv(args.output / "training_metrics.csv", index=False, encoding="utf-8-sig")
    with (args.output / "SHA256SUMS.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "sha256"])
        writer.writeheader()
        writer.writerows(model_hashes)
    metadata = {
        "schema_version": "four-shared-rf-oracles-v1",
        "warning": (
            "EGFR/VEGFR2 use recovered BindingDB API snapshots and are not an exact row-for-row "
            "reconstruction of the historical oracle training data."
        ),
        "features": {"type": "Morgan/ECFP4 bit vector", "radius": 2, "n_bits": 2048, "include_chirality": True},
        "model": {
            "class": "sklearn.ensemble.RandomForestRegressor",
            "n_estimators": 1000,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "random_state": 0,
            "n_jobs": -1,
            "other_hyperparameters": "scikit-learn defaults",
        },
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "sources": sources,
        "models": {row["target"]: f"target_{row['target']}_model.pkl" for row in rows},
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
