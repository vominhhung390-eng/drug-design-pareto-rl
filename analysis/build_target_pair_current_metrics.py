from __future__ import annotations

import argparse
import gzip
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


SEEDS = tuple(range(42, 52))
SAMPLE_SIZE = 2000
METRICS = (
    "validity",
    "uniqueness",
    "novelty",
    "diversity",
    "hypervolume",
    "igd_plus",
    "pareto_size",
    "dual_at_6",
    "dual_at_6_5",
    "quality_pass",
    "alert_free",
    "scaffold_diversity",
    "qc_hypervolume",
    "qc_dual_at_6",
    "qc_dual_at_6_5",
    "qc_dual_at_7",
    "qc_best_min",
)


def normalize(points: np.ndarray) -> np.ndarray:
    """Map pActivity 3--10 to the registered [0, 1] objective range."""
    return np.clip((np.asarray(points, dtype=float) - 3.0) / 7.0, 0.0, 1.0)


def pareto_front_2d(points: np.ndarray) -> np.ndarray:
    """Return nondominated rows for two maximization objectives."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    sorted_points = points[order]
    keep: list[int] = []
    prior_max_y = -math.inf
    start = 0
    while start < len(sorted_points):
        x = sorted_points[start, 0]
        stop = start + 1
        while stop < len(sorted_points) and sorted_points[stop, 0] == x:
            stop += 1
        group = sorted_points[start:stop]
        group_max_y = float(group[:, 1].max())
        if group_max_y > prior_max_y:
            local = np.flatnonzero(group[:, 1] == group_max_y)
            keep.extend((start + local).tolist())
        prior_max_y = max(prior_max_y, group_max_y)
        start = stop
    return sorted_points[np.asarray(keep, dtype=int)]


def hypervolume_2d(normalized_points: np.ndarray) -> float:
    front = pareto_front_2d(normalized_points)
    front = front[np.all(front > 0.0, axis=1)]
    if len(front) == 0:
        return 0.0
    front = np.unique(front, axis=0)
    front = front[np.argsort(front[:, 0])]
    hv = 0.0
    best_y = 0.0
    for x, y in front[::-1]:
        if y > best_y:
            hv += float(x) * float(y - best_y)
            best_y = float(y)
    return hv


def igd_plus(approximation: np.ndarray, reference_front: np.ndarray) -> float:
    if len(approximation) == 0 or len(reference_front) == 0:
        return math.nan
    distances = []
    for target in reference_front:
        deficits = np.maximum(target[None, :] - approximation, 0.0)
        distances.append(float(np.linalg.norm(deficits, axis=1).min()))
    return float(np.mean(distances))


def internal_diversity(smiles: list[str], method: str, seed: int) -> tuple[float, int]:
    if len(smiles) < 2:
        return 0.0, len(smiles)
    rng = np.random.default_rng((zlib.crc32(method.encode("utf-8")) & 0xFFFFFFFF) + seed)
    if len(smiles) > SAMPLE_SIZE:
        indices = np.sort(rng.choice(len(smiles), size=SAMPLE_SIZE, replace=False))
        smiles = [smiles[int(index)] for index in indices]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is not None:
            fingerprints.append(generator.GetFingerprint(mol))
    if len(fingerprints) < 2:
        return 0.0, len(fingerprints)
    similarity_sum = 0.0
    pair_count = 0
    for index, fingerprint in enumerate(fingerprints[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[index + 1 :])
        similarity_sum += float(sum(similarities))
        pair_count += len(similarities)
    return 1.0 - similarity_sum / pair_count, len(fingerprints)


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_training_cache(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Canonical training cache not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {line.rstrip("\n") for line in handle if line.strip()}


def method_specs(experiment_root: Path) -> list[tuple[str, Path]]:
    return [
        ("Ours (V4-B)", experiment_root / "own_method"),
        ("POLYGON", experiment_root / "baselines" / "polygon_original"),
        ("REINVENT4", experiment_root / "baselines" / "reinvent4"),
        ("DrugEx v2", experiment_root / "baselines" / "drugex_v2"),
        ("MO-LSO", experiment_root / "baselines" / "mo_lso"),
        ("GraphPareto-NSGA-II", experiment_root / "baselines" / "graphpareto_nsga2"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("training_cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    args.output.mkdir(parents=True, exist_ok=True)
    training = load_training_cache(args.training_cache)

    records: list[dict[str, object]] = []
    fronts: dict[tuple[str, int], np.ndarray] = {}
    completion: list[dict[str, object]] = []

    for method, method_root in method_specs(args.experiment_root):
        completed_seeds = 0
        for seed in SEEDS:
            evaluation = method_root / f"formal_10240_seed{seed}" / "anytime" / "budget_10240"
            molecule_path = evaluation / "standardized_molecules.csv"
            summary_path = evaluation / "evaluation_summary.json"
            quality_path = evaluation / "quality_constrained" / "quality_annotated_molecules.csv"
            if not all(path.exists() for path in (molecule_path, summary_path, quality_path)):
                continue

            completed_seeds += 1
            frame = pd.read_csv(molecule_path, encoding="utf-8-sig")
            quality = pd.read_csv(quality_path, encoding="utf-8-sig")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if len(frame) != int(summary["unique_valid"]):
                raise ValueError(f"{method} seed {seed}: standardized row count mismatch")
            if len(quality) != len(frame):
                raise ValueError(f"{method} seed {seed}: quality row count mismatch")
            if set(frame["smiles"].astype(str)) != set(quality["smiles"].astype(str)):
                raise ValueError(f"{method} seed {seed}: quality molecule identities mismatch")

            points = frame[["egfr", "vegfr2"]].to_numpy(float)
            normalized_points = normalize(points)
            front = pareto_front_2d(normalized_points)
            fronts[(method, seed)] = np.unique(front, axis=0)
            smiles = frame["smiles"].astype(str).tolist()
            diversity, diversity_sample_n = internal_diversity(smiles, method, seed)

            quality_pass = parse_bool(quality["quality_pass"])
            alert_free = ~parse_bool(quality["structural_alert"])
            qc = quality.loc[quality_pass]
            qc_points = qc[["egfr", "vegfr2"]].to_numpy(float) if len(qc) else np.empty((0, 2))
            nonempty_scaffolds = frame["scaffold"].fillna("").astype(str).str.len() > 0
            scaffold_count = frame.loc[nonempty_scaffolds, "scaffold"].nunique()

            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "generated_rows": int(summary["generated_rows"]),
                    "valid_rows": int(summary["valid_rows"]),
                    "unique_valid": int(summary["unique_valid"]),
                    "diversity_sample_n": diversity_sample_n,
                    "validity": float(summary["validity"]),
                    "uniqueness": float(summary["uniqueness_valid"]),
                    "novelty": sum(value not in training for value in smiles) / len(smiles) if smiles else 0.0,
                    "diversity": diversity,
                    "hypervolume": hypervolume_2d(normalized_points),
                    "igd_plus": math.nan,
                    "pareto_size": int(len(front)),
                    "dual_at_6": float(((frame["egfr"] >= 6.0) & (frame["vegfr2"] >= 6.0)).mean()) if len(frame) else 0.0,
                    "dual_at_6_5": float(((frame["egfr"] >= 6.5) & (frame["vegfr2"] >= 6.5)).mean()) if len(frame) else 0.0,
                    "quality_pass": float(quality_pass.mean()) if len(quality) else 0.0,
                    "alert_free": float(alert_free.mean()) if len(quality) else 0.0,
                    "scaffold_diversity": float(scaffold_count / len(frame)) if len(frame) else 0.0,
                    "qc_hypervolume": hypervolume_2d(normalize(qc_points)) if len(qc) else 0.0,
                    "qc_dual_at_6": float(((qc["egfr"] >= 6.0) & (qc["vegfr2"] >= 6.0)).mean()) if len(qc) else 0.0,
                    "qc_dual_at_6_5": float(((qc["egfr"] >= 6.5) & (qc["vegfr2"] >= 6.5)).mean()) if len(qc) else 0.0,
                    "qc_dual_at_7": float(((qc["egfr"] >= 7.0) & (qc["vegfr2"] >= 7.0)).mean()) if len(qc) else 0.0,
                    "qc_best_min": float(np.minimum(qc["egfr"], qc["vegfr2"]).max()) if len(qc) else math.nan,
                }
            )
            print(f"{method} seed {seed} complete", flush=True)
        completion.append({"method": method, "completed_seeds": completed_seeds, "planned_seeds": len(SEEDS)})

    if not records:
        raise RuntimeError("No completed formal seed evaluations were found")
    pooled = np.vstack(list(fronts.values()))
    reference_front = np.unique(pareto_front_2d(np.unique(pooled, axis=0)), axis=0)
    reference_front = reference_front[np.argsort(reference_front[:, 0])]
    for record in records:
        key = (str(record["method"]), int(record["seed"]))
        record["igd_plus"] = igd_plus(fronts[key], reference_front)

    per_seed = pd.DataFrame(records)
    aggregate_rows = []
    for method, _ in method_specs(args.experiment_root):
        subset = per_seed[per_seed["method"] == method]
        if subset.empty:
            continue
        for metric in METRICS:
            values = subset[metric].astype(float)
            aggregate_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "n": int(values.notna().sum()),
                }
            )

    per_seed.to_csv(args.output / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(aggregate_rows).to_csv(args.output / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(completion).to_csv(args.output / "completion.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_front, columns=["parp1_normalized", "brd4_normalized"]).to_csv(
        args.output / "igd_reference_front.csv", index=False, encoding="utf-8-sig"
    )
    (args.output / "metric_metadata.json").write_text(
        json.dumps(
            {
                "target_mapping": {"egfr": "PARP1", "vegfr2": "BRD4"},
                "budget_per_seed": 10240,
                "planned_seeds": list(SEEDS),
                "training_cache": str(args.training_cache),
                "training_unique_canonical": len(training),
                "igd_reference_points": len(reference_front),
                "igd_reference_methods": sorted(per_seed["method"].unique().tolist()),
                "diversity_sample_size_cap": SAMPLE_SIZE,
                "quality_definition": "QED>=0.60; SA<=4.0; no PAINS/Brenk alert; Lipinski pass",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(per_seed)} seed rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
