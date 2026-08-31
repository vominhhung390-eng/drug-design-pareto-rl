"""Shared EGFR/VEGFR2 oracle used by every external baseline.

The module deliberately owns fingerprint construction so that individual
baseline repositories cannot silently use different QSAR preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EGFR_MODEL = PROJECT_ROOT / "models" / "oracles" / "target_EGFR_model.pkl"
DEFAULT_VEGFR2_MODEL = PROJECT_ROOT / "models" / "oracles" / "target_VEGFR2_model.pkl"


@dataclass(frozen=True)
class OracleResult:
    smiles: str
    canonical_smiles: str
    valid: bool
    egfr: float
    vegfr2: float


def canonicalize_smiles(smiles: str) -> str | None:
    """Desalt by retaining the largest fragment, then canonicalize."""

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        # Some decoder outputs pass MolFromSmiles but fail during fragment
        # sanitization/kekulization.  They are invalid proposals, not fatal
        # oracle errors, and still count against the caller's exact budget.
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not fragments:
            return None
        mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except (ValueError, RuntimeError, Chem.rdchem.KekulizeException):
        return None


def fingerprint(canonical_smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {canonical_smiles!r}")
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=2048, useChirality=True
    )
    array = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


class DualTargetOracle:
    def __init__(
        self,
        egfr_model: Path | None = None,
        vegfr2_model: Path | None = None,
    ) -> None:
        # The historical EGFR/VEGFR2 pair remains the no-argument default.
        # A formal target-pair runner may override both paths through the
        # environment so every unmodified baseline adapter receives exactly
        # the same second-pair oracle without changing its native algorithm.
        self.egfr_model_path = Path(
            egfr_model
            or os.environ.get("DUAL_TARGET_MODEL_1", DEFAULT_EGFR_MODEL)
        )
        self.vegfr2_model_path = Path(
            vegfr2_model
            or os.environ.get("DUAL_TARGET_MODEL_2", DEFAULT_VEGFR2_MODEL)
        )
        self.target_names = (
            os.environ.get("DUAL_TARGET_NAME_1", "EGFR"),
            os.environ.get("DUAL_TARGET_NAME_2", "VEGFR2"),
        )
        with self.egfr_model_path.open("rb") as handle:
            self.egfr_model = pickle.load(handle)
        with self.vegfr2_model_path.open("rb") as handle:
            self.vegfr2_model = pickle.load(handle)
        for name, model in zip(self.target_names, (self.egfr_model, self.vegfr2_model)):
            feature_count = getattr(model, "n_features_in_", 2048)
            if int(feature_count) != 2048:
                raise ValueError(
                    f"{name} oracle expects {feature_count} features; shared bridge provides 2048"
                )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "target_1": self.target_names[0],
            "target_2": self.target_names[1],
            "target_1_model": str(self.egfr_model_path.resolve()),
            "target_2_model": str(self.vegfr2_model_path.resolve()),
            "compatibility_columns": {"egfr": self.target_names[0], "vegfr2": self.target_names[1]},
        }

    def score_many(self, smiles: Sequence[str]) -> list[OracleResult]:
        canonical = [canonicalize_smiles(item) for item in smiles]
        valid_indices = [index for index, item in enumerate(canonical) if item is not None]
        scores: dict[int, tuple[float, float]] = {}
        if valid_indices:
            matrix = np.stack([fingerprint(canonical[index]) for index in valid_indices])
            egfr = self.egfr_model.predict(matrix)
            vegfr2 = self.vegfr2_model.predict(matrix)
            scores = {
                index: (float(egfr[pos]), float(vegfr2[pos]))
                for pos, index in enumerate(valid_indices)
            }
        return [
            OracleResult(
                smiles=item,
                canonical_smiles=canonical[index] or "",
                valid=index in scores,
                egfr=scores.get(index, (0.0, 0.0))[0],
                vegfr2=scores.get(index, (0.0, 0.0))[1],
            )
            for index, item in enumerate(smiles)
        ]

    def score(self, smiles: str) -> OracleResult:
        return self.score_many([smiles])[0]


def read_smiles(path: Path) -> Iterable[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            key = "smiles" if "smiles" in (reader.fieldnames or []) else "Smiles"
            for row in reader:
                yield row[key]
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip().split()[0] if line.strip() else ""
                if value:
                    yield value


def main() -> None:
    parser = argparse.ArgumentParser(description="Score SMILES with the shared dual-target oracle")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    oracle = DualTargetOracle()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["smiles", "canonical_smiles", "valid", "egfr", "vegfr2"],
        )
        writer.writeheader()
        batch: list[str] = []
        for item in read_smiles(args.input):
            batch.append(item)
            if len(batch) >= args.batch_size:
                writer.writerows(result.__dict__ for result in oracle.score_many(batch))
                batch.clear()
        if batch:
            writer.writerows(result.__dict__ for result in oracle.score_many(batch))


if __name__ == "__main__":
    main()
