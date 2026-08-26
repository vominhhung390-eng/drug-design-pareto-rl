#!/usr/bin/env python
"""Compute pre-registered quality-constrained Pareto metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYGON_ROOT = PROJECT_ROOT / "vendor" / "polygon-main"
if str(POLYGON_ROOT) not in sys.path:
    sys.path.insert(0, str(POLYGON_ROOT))
from polygon.utils.custom_scoring_fcn import SAScorer

from multiobjective_metrics import hypervolume_2d, pareto_front

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


def front_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    points = frame[["egfr", "vegfr2"]].dropna().to_numpy(float)
    front = pareto_front(points) if len(points) else np.empty((0, 2))
    return {
        f"{prefix}_molecules": len(frame),
        f"{prefix}_pareto_size": len(front),
        f"{prefix}_hypervolume": hypervolume_2d(front),
        f"{prefix}_best_min_activity": float(np.minimum(points[:, 0], points[:, 1]).max()) if len(points) else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="standardized_molecules.csv from evaluate_experiment.py")
    p.add_argument("output", type=Path)
    p.add_argument("--qed-min", type=float, default=0.60)
    p.add_argument("--sa-max", type=float, default=4.0)
    p.add_argument("--fscores", type=Path, default=POLYGON_ROOT / "data" / "fpscores.pkl.gz")
    args = p.parse_args()

    frame = pd.read_csv(args.input)
    required = {"smiles", "egfr", "vegfr2", "qed", "structural_alert"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    extra = []
    keep = []
    sa_scorer = SAScorer(score_modifier=None, fscores=str(args.fscores))
    for i, smi in enumerate(frame["smiles"].astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        sa = float(sa_scorer.raw_score(smi))
        keep.append(i)
        extra.append({
            "sa": sa,
            "hbd": hbd,
            "hba": hba,
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "lipinski_pass": bool(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
        })
    frame = frame.iloc[keep].reset_index(drop=True)
    frame = pd.concat([frame, pd.DataFrame(extra)], axis=1)
    alert_free = ~frame["structural_alert"].astype(bool)
    qed_pass = frame["qed"] >= args.qed_min
    sa_pass = frame["sa"] <= args.sa_max
    lipinski_pass = frame["lipinski_pass"].astype(bool)
    frame["quality_pass"] = alert_free & qed_pass & sa_pass & lipinski_pass
    frame["dual_active_6"] = (frame["egfr"] >= 6.0) & (frame["vegfr2"] >= 6.0)
    frame["dual_active_7"] = (frame["egfr"] >= 7.0) & (frame["vegfr2"] >= 7.0)

    summary: dict[str, object] = {
        "qed_min": args.qed_min,
        "sa_max": args.sa_max,
        "quality_definition": "QED>=0.60; SA<=4.0; no PAINS/Brenk alert; Lipinski pass",
        "total_unique_valid": len(frame),
        "qed_pass_rate": float(qed_pass.mean()) if len(frame) else 0.0,
        "sa_pass_rate": float(sa_pass.mean()) if len(frame) else 0.0,
        "alert_free_rate": float(alert_free.mean()) if len(frame) else 0.0,
        "lipinski_pass_rate": float(lipinski_pass.mean()) if len(frame) else 0.0,
        "quality_pass_rate": float(frame["quality_pass"].mean()) if len(frame) else 0.0,
        "dual_active_6_rate": float(frame["dual_active_6"].mean()) if len(frame) else 0.0,
        "dual_active_7_rate": float(frame["dual_active_7"].mean()) if len(frame) else 0.0,
    }
    summary.update(front_metrics(frame, "raw"))
    summary.update(front_metrics(frame[qed_pass], "qed_constrained"))
    summary.update(front_metrics(frame[alert_free], "alert_free"))
    summary.update(front_metrics(frame[frame["quality_pass"]], "quality_constrained"))

    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "quality_annotated_molecules.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(args.output / "quality_constrained_summary.csv", index=False, encoding="utf-8-sig")
    (args.output / "quality_constrained_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
