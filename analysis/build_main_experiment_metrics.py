from __future__ import annotations

import gzip
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "019f5762-58d7-7670-9168-54fe5fbeb2b3"
TRAIN_FILE = ROOT / "data" / "train_smiles_only.txt"
CACHE_FILE = OUT / "training_smiles_canonical.txt.gz"
SEEDS = list(range(42, 52))
SAMPLE_SIZE = 2000

METHODS = [
    {
        "method": "Ours (V4)",
        "run": lambda seed: ROOT
        / "results"
        / "own_method_v4"
        / "common_seeds_42_51_10240"
        / f"v4_b_raw_mean_seed{seed}",
        "evaluation": lambda run: run / "anytime" / "budget_10240",
        "quality": lambda run: run / "evaluation" / "quality_constrained",
    },
    {
        "method": "POLYGON",
        "run": lambda seed: ROOT / "results" / "baselines" / "polygon_original" / f"formal_10240_seed{seed}",
        "evaluation": lambda run: run / "anytime" / "budget_10240",
        "quality": lambda run: run / "anytime" / "budget_10240" / "quality_constrained",
    },
    {
        "method": "REINVENT4",
        "run": lambda seed: ROOT / "results" / "baselines" / "reinvent4" / f"formal_10240_seed{seed}",
        "evaluation": lambda run: run / "anytime" / "budget_10240",
        "quality": lambda run: run / "anytime" / "budget_10240" / "quality_constrained",
    },
    {
        "method": "DrugEx v2",
        "run": lambda seed: ROOT / "results" / "baselines" / "drugex_v2" / f"formal_10240_seed{seed}",
        "evaluation": lambda run: run / "anytime" / "budget_10240",
        "quality": lambda run: run / "anytime" / "budget_10240" / "quality_constrained",
    },
    {
        "method": "MO-LSO",
        "run": lambda seed: ROOT / "results" / "baselines" / "mo_lso" / f"formal_10240_seed{seed}",
        "evaluation": lambda run: run / "anytime" / "budget_10240",
        "quality": lambda run: run / "anytime" / "budget_10240" / "quality_constrained",
    },
    {
        "method": "GraphPareto–NSGA-II",
        "run": lambda seed: ROOT / "results" / "baselines" / "graphpareto_nsga2" / f"formal_10240_seed{seed}",
        "evaluation": lambda run: run / "anytime" / "budget_10240",
        "quality": lambda run: run / "anytime" / "budget_10240" / "quality_constrained",
    },
]

METRICS = [
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
]


def canonicalize(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not fragments:
            return None
        mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except (ValueError, RuntimeError, Chem.rdchem.KekulizeException):
        return None


def load_training_smiles() -> set[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and CACHE_FILE.stat().st_mtime >= TRAIN_FILE.stat().st_mtime:
        print(f"Loading canonical training cache: {CACHE_FILE}", flush=True)
        with gzip.open(CACHE_FILE, "rt", encoding="utf-8") as handle:
            return {line.rstrip("\n") for line in handle if line.strip()}

    print("Canonicalizing the common training set for novelty...", flush=True)
    canonical: set[str] = set()
    total = 0
    invalid = 0
    with TRAIN_FILE.open("r", encoding="utf-8") as handle:
        for total, line in enumerate(handle, start=1):
            value = canonicalize(line.strip())
            if value is None:
                invalid += 1
            else:
                canonical.add(value)
            if total % 250_000 == 0:
                print(f"  training rows {total:,}; unique canonical {len(canonical):,}", flush=True)

    with gzip.open(CACHE_FILE, "wt", encoding="utf-8", compresslevel=5) as handle:
        for value in sorted(canonical):
            handle.write(value + "\n")
    print(
        f"Training set ready: rows={total:,}, unique canonical={len(canonical):,}, invalid={invalid:,}",
        flush=True,
    )
    return canonical


def normalize(points: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(points, dtype=float) - 3.0) / 7.0, 0.0, 1.0)


def pareto_front_2d(points: np.ndarray) -> np.ndarray:
    """Return all nondominated rows for two maximization objectives.

    Exact duplicate objective rows are retained, matching the project's formal
    evaluator, while the implementation is O(n log n).
    """
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


def hypervolume_2d_normalized(normalized_points: np.ndarray) -> float:
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
    distances: list[float] = []
    for target in reference_front:
        deficits = np.maximum(target[None, :] - approximation, 0.0)
        distances.append(float(np.linalg.norm(deficits, axis=1).min()))
    return float(np.mean(distances))


def internal_diversity(smiles: list[str], method: str, seed: int) -> tuple[float, int]:
    if len(smiles) < 2:
        return 0.0, len(smiles)
    method_offset = zlib.crc32(method.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(method_offset + seed)
    if len(smiles) > SAMPLE_SIZE:
        indices = np.sort(rng.choice(len(smiles), size=SAMPLE_SIZE, replace=False))
        sampled = [smiles[int(i)] for i in indices]
    else:
        sampled = smiles

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for value in sampled:
        mol = Chem.MolFromSmiles(value)
        if mol is not None:
            fps.append(generator.GetFingerprint(mol))
    if len(fps) < 2:
        return 0.0, len(fps)

    similarity_sum = 0.0
    pair_count = 0
    for index, fp in enumerate(fps[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, fps[index + 1 :])
        similarity_sum += float(sum(similarities))
        pair_count += len(similarities)
    return 1.0 - similarity_sum / pair_count, len(fps)


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main() -> None:
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    OUT.mkdir(parents=True, exist_ok=True)
    training = load_training_smiles()

    records: list[dict[str, object]] = []
    fronts: dict[tuple[str, int], np.ndarray] = {}
    validation: list[dict[str, object]] = []

    for method_spec in METHODS:
        method = method_spec["method"]
        for seed in SEEDS:
            run = method_spec["run"](seed)
            evaluation = method_spec["evaluation"](run)
            quality_dir = method_spec["quality"](run)
            molecule_path = evaluation / "standardized_molecules.csv"
            summary_path = evaluation / "evaluation_summary.json"
            quality_path = quality_dir / "quality_annotated_molecules.csv"
            for required in (molecule_path, summary_path, quality_path):
                if not required.exists():
                    raise FileNotFoundError(required)

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
            hv = hypervolume_2d_normalized(normalized_points)
            summary_hv = float(summary["hypervolume"])
            hv_matches = math.isclose(hv, summary_hv, rel_tol=1e-9, abs_tol=1e-10)
            legacy_unnormalized_hv = summary_hv > 1.0
            if not hv_matches and not legacy_unnormalized_hv:
                raise ValueError(f"{method} seed {seed}: HV mismatch {hv} vs {summary_hv}")
            if len(front) != int(summary["pareto_size"]):
                raise ValueError(
                    f"{method} seed {seed}: Pareto size mismatch {len(front)} vs {summary['pareto_size']}"
                )

            smiles = frame["smiles"].astype(str).tolist()
            novelty = sum(value not in training for value in smiles) / len(smiles) if smiles else 0.0
            diversity, diversity_sample_n = internal_diversity(smiles, method, seed)

            quality_pass = parse_bool(quality["quality_pass"])
            alert_free = ~parse_bool(quality["structural_alert"])
            qc = quality.loc[quality_pass]
            qc_points = qc[["egfr", "vegfr2"]].to_numpy(float) if len(qc) else np.empty((0, 2))
            nonempty_scaffolds = frame["scaffold"].fillna("").astype(str).str.len() > 0
            scaffold_count = frame.loc[nonempty_scaffolds, "scaffold"].nunique()

            record: dict[str, object] = {
                "method": method,
                "seed": seed,
                "generated_rows": int(summary["generated_rows"]),
                "valid_rows": int(summary["valid_rows"]),
                "unique_valid": int(summary["unique_valid"]),
                "diversity_sample_n": diversity_sample_n,
                "validity": float(summary["validity"]),
                "uniqueness": float(summary["uniqueness_valid"]),
                "novelty": float(novelty),
                "diversity": float(diversity),
                "hypervolume": float(hv),
                "igd_plus": math.nan,
                "pareto_size": int(len(front)),
                "dual_at_6": float(((frame["egfr"] >= 6.0) & (frame["vegfr2"] >= 6.0)).mean())
                if len(frame)
                else 0.0,
                "dual_at_6_5": float(((frame["egfr"] >= 6.5) & (frame["vegfr2"] >= 6.5)).mean())
                if len(frame)
                else 0.0,
                "quality_pass": float(quality_pass.mean()) if len(quality) else 0.0,
                "alert_free": float(alert_free.mean()) if len(quality) else 0.0,
                "scaffold_diversity": float(scaffold_count / len(frame)) if len(frame) else 0.0,
                "qc_hypervolume": hypervolume_2d_normalized(normalize(qc_points)) if len(qc) else 0.0,
                "qc_dual_at_6": float(((qc["egfr"] >= 6.0) & (qc["vegfr2"] >= 6.0)).mean())
                if len(qc)
                else 0.0,
                "qc_dual_at_6_5": float(((qc["egfr"] >= 6.5) & (qc["vegfr2"] >= 6.5)).mean())
                if len(qc)
                else 0.0,
                "qc_dual_at_7": float(((qc["egfr"] >= 7.0) & (qc["vegfr2"] >= 7.0)).mean())
                if len(qc)
                else 0.0,
                "qc_best_min": float(np.minimum(qc["egfr"], qc["vegfr2"]).max()) if len(qc) else math.nan,
            }
            records.append(record)
            validation.append(
                {
                    "method": method,
                    "seed": seed,
                    "molecules": str(molecule_path),
                    "quality": str(quality_path),
                    "summary": str(summary_path),
                    "formal_summary_hv": summary_hv,
                    "recomputed_normalized_hv": hv,
                    "hv_matches_formal_summary": hv_matches,
                    "legacy_unnormalized_hv_detected": legacy_unnormalized_hv,
                    "pareto_size_matches_formal_summary": True,
                    "quality_identity_matches": True,
                }
            )
            print(
                f"{method:10s} seed {seed}: valid={record['validity']:.4f}, "
                f"novel={record['novelty']:.4f}, diversity={record['diversity']:.4f}",
                flush=True,
            )

    pooled = np.vstack(list(fronts.values()))
    reference_front = np.unique(pareto_front_2d(np.unique(pooled, axis=0)), axis=0)
    reference_front = reference_front[np.argsort(reference_front[:, 0])]
    for record in records:
        key = (str(record["method"]), int(record["seed"]))
        record["igd_plus"] = igd_plus(fronts[key], reference_front)

    per_seed = pd.DataFrame(records)
    method_order = [spec["method"] for spec in METHODS]
    per_seed["method"] = pd.Categorical(per_seed["method"], categories=method_order, ordered=True)
    per_seed = per_seed.sort_values(["method", "seed"]).reset_index(drop=True)
    per_seed["method"] = per_seed["method"].astype(str)

    aggregate_rows: list[dict[str, object]] = []
    for method in method_order:
        method_frame = per_seed[per_seed["method"] == method]
        for metric in METRICS:
            values = method_frame[metric].astype(float)
            aggregate_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "n": int(values.notna().sum()),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)

    per_seed.to_csv(OUT / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_front, columns=["egfr_normalized", "vegfr2_normalized"]).to_csv(
        OUT / "igd_reference_front.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "metric_validation.json").write_text(
        json.dumps(
            {
                "methods": method_order,
                "seeds": SEEDS,
                "budget_per_seed": 10240,
                "training_file": str(TRAIN_FILE),
                "training_unique_canonical": len(training),
                "igd_reference_points": len(reference_front),
                "diversity_sample_size_cap": SAMPLE_SIZE,
                "input_checks": validation,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(per_seed)} per-seed rows and {len(aggregate)} aggregate rows to {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
