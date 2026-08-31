#!/usr/bin/env python
"""Measure whether the ECFP4 chirality mismatch changes formal method rankings."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from multiobjective_metrics import hypervolume_2d
from validate_target_predictors import fingerprints


def has_stereo(smiles: str) -> bool:
    text = str(smiles)
    return "@" in text or "/" in text or "\\" in text


def score_map(smiles: list[str], model, use_chirality: bool) -> dict[str, float]:
    if not smiles:
        return {}
    x, _ = fingerprints(smiles, use_chirality=use_chirality)
    prediction = model.predict(x)
    return dict(zip(smiles, prediction.astype(float), strict=True))


def rescore(frame: pd.DataFrame, maps: dict[str, dict[str, float]]) -> pd.DataFrame:
    result = frame.copy()
    stereo = result["smiles"].astype(str).map(has_stereo)
    for target in ("egfr", "vegfr2"):
        result.loc[stereo, target] = result.loc[stereo, "smiles"].map(maps[target]).to_numpy(float)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "outputs" / "019f5762-58d7-7670-9168-54fe5fbeb2b3" / "metric_validation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "predictor_validation_20260729" / "generated_chirality_audit",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checks = manifest["input_checks"]

    stereo_smiles: set[str] = set()
    for check in checks:
        frame = pd.read_csv(check["molecules"], usecols=["smiles"])
        stereo_smiles.update(smi for smi in frame["smiles"].astype(str) if has_stereo(smi))
    stereo_list = sorted(stereo_smiles)

    models = {}
    for target, filename in {
        "egfr": "target_EGFR_model.pkl",
        "vegfr2": "target_VEGFR2_model.pkl",
    }.items():
        with (root / "models" / "oracles" / filename).open("rb") as handle:
            models[target] = pickle.load(handle)
    achiral_maps = {target: score_map(stereo_list, model, False) for target, model in models.items()}
    chiral_maps = {target: score_map(stereo_list, model, True) for target, model in models.items()}

    molecule_audit = pd.DataFrame({"smiles": stereo_list})
    for target in ("egfr", "vegfr2"):
        molecule_audit[f"{target}_chiral"] = molecule_audit["smiles"].map(chiral_maps[target])
        molecule_audit[f"{target}_achiral"] = molecule_audit["smiles"].map(achiral_maps[target])
        molecule_audit[f"{target}_abs_diff"] = np.abs(
            molecule_audit[f"{target}_chiral"] - molecule_audit[f"{target}_achiral"]
        )
    molecule_audit.to_csv(args.output / "stereochemical_molecule_score_differences.csv", index=False, encoding="utf-8-sig")

    rows: list[dict] = []
    for check in checks:
        frame = pd.read_csv(check["molecules"])
        quality = pd.read_csv(check["quality"])
        stereo_mask = frame["smiles"].astype(str).map(has_stereo)
        quality_stereo_mask = quality["smiles"].astype(str).map(has_stereo)
        achiral = rescore(frame, achiral_maps)
        achiral_quality = rescore(quality, achiral_maps)
        quality_operational = quality[quality["quality_pass"].astype(bool)]
        quality_achiral = achiral_quality[achiral_quality["quality_pass"].astype(bool)]

        stored_vs_chiral = []
        for target in ("egfr", "vegfr2"):
            if stereo_mask.any():
                expected = frame.loc[stereo_mask, "smiles"].map(chiral_maps[target]).to_numpy(float)
                stored_vs_chiral.extend(np.abs(frame.loc[stereo_mask, target].to_numpy(float) - expected))
        rows.append(
            {
                "method": check["method"],
                "seed": int(check["seed"]),
                "molecules": len(frame),
                "stereo_molecules": int(stereo_mask.sum()),
                "stereo_rate": float(stereo_mask.mean()),
                "quality_stereo_rate": float(quality_stereo_mask.mean()),
                "stored_vs_recomputed_chiral_max_abs_diff": float(np.max(stored_vs_chiral)) if stored_vs_chiral else 0.0,
                "operational_hv": hypervolume_2d(frame[["egfr", "vegfr2"]].to_numpy(float)),
                "training_faithful_achiral_hv": hypervolume_2d(achiral[["egfr", "vegfr2"]].to_numpy(float)),
                "operational_quality_hv": hypervolume_2d(quality_operational[["egfr", "vegfr2"]].to_numpy(float)),
                "training_faithful_achiral_quality_hv": hypervolume_2d(quality_achiral[["egfr", "vegfr2"]].to_numpy(float)),
                "egfr_mean_abs_score_change": float(np.mean(np.abs(achiral["egfr"] - frame["egfr"]))),
                "vegfr2_mean_abs_score_change": float(np.mean(np.abs(achiral["vegfr2"] - frame["vegfr2"]))),
                "egfr_max_abs_score_change": float(np.max(np.abs(achiral["egfr"] - frame["egfr"]))),
                "vegfr2_max_abs_score_change": float(np.max(np.abs(achiral["vegfr2"] - frame["vegfr2"]))),
            }
        )

    per_run = pd.DataFrame(rows)
    per_run["hv_change_achiral_minus_operational"] = per_run["training_faithful_achiral_hv"] - per_run["operational_hv"]
    per_run["quality_hv_change_achiral_minus_operational"] = per_run["training_faithful_achiral_quality_hv"] - per_run["operational_quality_hv"]
    per_run.to_csv(args.output / "per_run_chirality_impact.csv", index=False, encoding="utf-8-sig")

    aggregate = per_run.groupby("method", as_index=False).agg(
        n=("seed", "count"),
        stereo_rate_mean=("stereo_rate", "mean"),
        operational_hv_mean=("operational_hv", "mean"),
        operational_hv_sd=("operational_hv", "std"),
        achiral_hv_mean=("training_faithful_achiral_hv", "mean"),
        achiral_hv_sd=("training_faithful_achiral_hv", "std"),
        hv_change_mean=("hv_change_achiral_minus_operational", "mean"),
        operational_quality_hv_mean=("operational_quality_hv", "mean"),
        achiral_quality_hv_mean=("training_faithful_achiral_quality_hv", "mean"),
        quality_hv_change_mean=("quality_hv_change_achiral_minus_operational", "mean"),
        stored_recompute_max=("stored_vs_recomputed_chiral_max_abs_diff", "max"),
        egfr_score_change_mean=("egfr_mean_abs_score_change", "mean"),
        vegfr2_score_change_mean=("vegfr2_mean_abs_score_change", "mean"),
    )
    aggregate["operational_rank"] = aggregate["operational_hv_mean"].rank(method="min", ascending=False).astype(int)
    aggregate["achiral_rank"] = aggregate["achiral_hv_mean"].rank(method="min", ascending=False).astype(int)
    aggregate = aggregate.sort_values("operational_rank")
    aggregate.to_csv(args.output / "method_chirality_impact_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Generated-molecule fingerprint chirality audit",
        "",
        f"Across the 60 formal common-seed runs, {len(stereo_list):,} unique stereochemistry-bearing SMILES were rescored with the achiral ECFP4 representation used by the original POLYGON RF training script.",
        "",
        "| Method | Stereo rate | Operational HV | Achiral HV | Mean change | Rank before -> after |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.stereo_rate_mean:.4f} | {row.operational_hv_mean:.4f} | {row.achiral_hv_mean:.4f} | {row.hv_change_mean:+.4f} | {row.operational_rank} -> {row.achiral_rank} |"
        )
    rank_changed = bool((aggregate["operational_rank"] != aggregate["achiral_rank"]).any())
    lines += [
        "",
        f"Method ordering changed: **{rank_changed}**.",
        "The formal results remain the operational chiral-oracle experiment. This audit only tests representation sensitivity and must not be substituted silently for a newly optimized achiral-oracle run.",
    ]
    (args.output / "generated_chirality_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated chirality audit complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
