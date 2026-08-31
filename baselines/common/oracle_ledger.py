"""Exact oracle-budget accounting shared by all five baseline adapters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from baselines.common.oracle_bridge import DualTargetOracle, OracleResult


FIELDS = [
    "oracle_index",
    "phase",
    "iteration",
    "sample_index",
    "smiles",
    "canonical_smiles",
    "valid",
    "egfr",
    "vegfr2",
    "egfr_desirability",
    "vegfr2_desirability",
]


def desirability(value: float, lower: float = 3.0, upper: float = 6.5) -> float:
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


class OracleLedger:
    """Score every generated row and stop exactly at the registered budget.

    Invalid and duplicate generations still consume one terminal-generation
    budget row, matching the primary method's ``generated_rows == budget``
    accounting.  Invalid rows receive zero raw scores and zero desirability.
    """

    def __init__(
        self,
        budget: int,
        output_csv: Path,
        oracle: DualTargetOracle | None = None,
        *,
        resume: bool = False,
    ):
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.budget = int(budget)
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.oracle = oracle or DualTargetOracle()
        self.records: list[dict] = []
        if resume and self.output_csv.exists():
            with self.output_csv.open("r", encoding="utf-8", newline="") as handle:
                for expected_index, row in enumerate(csv.DictReader(handle)):
                    oracle_index = int(row["oracle_index"])
                    if oracle_index != expected_index:
                        raise ValueError(
                            f"Non-contiguous oracle ledger at row {expected_index}: {oracle_index}"
                        )
                    self.records.append(
                        {
                            "oracle_index": oracle_index,
                            "phase": row["phase"],
                            "iteration": int(row["iteration"]),
                            "sample_index": int(row["sample_index"]),
                            "smiles": row["smiles"],
                            "canonical_smiles": row["canonical_smiles"],
                            "valid": int(row["valid"]),
                            "egfr": float(row["egfr"]),
                            "vegfr2": float(row["vegfr2"]),
                            "egfr_desirability": float(row["egfr_desirability"]),
                            "vegfr2_desirability": float(row["vegfr2_desirability"]),
                        }
                    )
            if len(self.records) > self.budget:
                raise ValueError(
                    f"Existing oracle ledger has {len(self.records)} rows, above budget {self.budget}"
                )

    @property
    def used(self) -> int:
        return len(self.records)

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0

    def score(
        self,
        smiles: Sequence[str],
        *,
        phase: str,
        iteration: int,
    ) -> tuple[list[OracleResult], np.ndarray]:
        selected = list(smiles[: self.remaining])
        if not selected:
            return [], np.empty((0, 2), dtype=np.float32)
        results = self.oracle.score_many(selected)
        reward_view = np.zeros((len(results), 2), dtype=np.float32)
        start = self.used
        for position, result in enumerate(results):
            if result.valid:
                reward_view[position, 0] = desirability(result.egfr)
                reward_view[position, 1] = desirability(result.vegfr2)
            self.records.append(
                {
                    "oracle_index": start + position,
                    "phase": phase,
                    "iteration": iteration,
                    "sample_index": position,
                    "smiles": result.smiles,
                    "canonical_smiles": result.canonical_smiles,
                    "valid": int(result.valid),
                    "egfr": result.egfr,
                    "vegfr2": result.vegfr2,
                    "egfr_desirability": float(reward_view[position, 0]),
                    "vegfr2_desirability": float(reward_view[position, 1]),
                }
            )
        self.flush()
        return results, reward_view

    def flush(self) -> None:
        with self.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.records)

    def write_metadata(self, path: Path, **extra) -> None:
        valid = sum(record["valid"] for record in self.records)
        unique_valid = len(
            {record["canonical_smiles"] for record in self.records if record["valid"]}
        )
        payload = {
            "budget": self.budget,
            "used": self.used,
            "complete": self.exhausted,
            "valid_rows": valid,
            "unique_valid": unique_valid,
            "oracle": self.oracle.metadata,
            "reward_view": {
                "type": "linear_clipped_desirability",
                "lower_pactivity": 3.0,
                "upper_pactivity": 6.5,
            },
            **extra,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
