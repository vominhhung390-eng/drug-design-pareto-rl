from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "docking/seed_top10_two_pairs_20260830"
TRAINING = PROJECT / "data/train_smiles_only.txt"
METHOD_ORDER = (
    "CLOVER-Mol",
    "POLYGON",
    "REINVENT4",
    "DrugEx v2",
    "MO-LSO",
    "GraphPareto-NSGA-II",
)
PAIR_ORDER = ("pairA", "pairB")
TARGETS = {"pairA": ("EGFR", "VEGFR2"), "pairB": ("PARP1", "BRD4")}
PASS_THRESHOLD = -7.0


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def load_compounds() -> pd.DataFrame:
    selected = pd.read_csv(ROOT / "selected_compounds.csv", encoding="utf-8-sig")
    raw = pd.DataFrame(json.loads((ROOT / "docking_raw_results.json").read_text(encoding="utf-8")))
    if len(selected) != 1200 or selected["compound_id"].nunique() != 1200:
        raise RuntimeError("Expected 1,200 uniquely identified candidate records")
    if len(raw) != 2400 or not raw["status"].eq("ok").all():
        raise RuntimeError("Expected 2,400 successful docking records")
    if raw["best_affinity"].isna().any() or raw["task_key"].duplicated().any():
        raise RuntimeError("Raw docking matrix has missing scores or duplicate tasks")
    per_candidate = raw.groupby("compound_id")["target"].nunique()
    if len(per_candidate) != 1200 or not per_candidate.eq(2).all():
        raise RuntimeError("Every candidate must have exactly two target results")
    pivot = raw.pivot(index="compound_id", columns="target", values="best_affinity").reset_index()
    compounds = selected.merge(pivot, on="compound_id", how="inner", validate="one_to_one")
    compounds["target_1_vina"] = np.where(compounds["pair_key"].eq("pairA"), compounds["EGFR"], compounds["PARP1"])
    compounds["target_2_vina"] = np.where(compounds["pair_key"].eq("pairA"), compounds["VEGFR2"], compounds["BRD4"])
    compounds["dual_worst_vina"] = compounds[["target_1_vina", "target_2_vina"]].max(axis=1)
    compounds["dual_mean_vina"] = compounds[["target_1_vina", "target_2_vina"]].mean(axis=1)
    compounds["dual_pass"] = compounds["target_1_vina"].le(PASS_THRESHOLD) & compounds["target_2_vina"].le(PASS_THRESHOLD)
    return compounds


def cached_novelty_by_smiles() -> dict[str, float]:
    return {}


def scan_uncached(smiles: list[str]) -> dict[str, float]:
    if not smiles:
        return {}
    RDLogger.DisableLog("rdApp.*")
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    query_fps = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise RuntimeError(f"Invalid selected SMILES: {value}")
        query_fps.append(fpgen.GetFingerprint(mol))
    maxima = np.full(len(query_fps), -1.0, dtype=np.float64)
    supplier = Chem.MultithreadedSmilesMolSupplier(
        str(TRAINING), delimiter=" \t", smilesColumn=0, nameColumn=-1,
        titleLine=False, sanitize=True, numWriterThreads=8,
        sizeInputQueue=4000, sizeOutputQueue=4000,
    )
    chunk_size = 20_000
    chunk_fps = []
    valid = 0
    started = time.time()
    workers = 8
    partitions = [np.asarray(x, dtype=int) for x in np.array_split(np.arange(len(query_fps)), workers) if len(x)]

    def evaluate(indices: np.ndarray, fps: list) -> tuple[np.ndarray, np.ndarray]:
        values = np.empty(len(indices), dtype=np.float64)
        for position, query_index in enumerate(indices):
            values[position] = max(DataStructs.BulkTanimotoSimilarity(query_fps[int(query_index)], fps))
        return indices, values

    def update(pool: ThreadPoolExecutor) -> None:
        if not chunk_fps:
            return
        futures = [pool.submit(evaluate, indices, chunk_fps) for indices in partitions]
        for future in futures:
            indices, values = future.result()
            maxima[indices] = np.maximum(maxima[indices], values)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for mol in supplier:
            if mol is None:
                continue
            valid += 1
            chunk_fps.append(fpgen.GetFingerprint(mol))
            if len(chunk_fps) >= chunk_size:
                update(pool)
                chunk_fps.clear()
            if valid % 100_000 == 0:
                elapsed = time.time() - started
                print(f"novelty_training_processed={valid:,} elapsed_s={elapsed:.1f}", flush=True)
        update(pool)
    print(f"novelty_training_complete={valid:,} elapsed_s={time.time()-started:.1f}", flush=True)
    return dict(zip(smiles, maxima.astype(float), strict=True))


def attach_novelty(compounds: pd.DataFrame) -> pd.DataFrame:
    final_path = ROOT / "training_novelty_top10_candidates.csv"
    if final_path.exists():
        existing = pd.read_csv(final_path, encoding="utf-8-sig")
        if set(existing["compound_id"].astype(str)) == set(compounds["compound_id"].astype(str)):
            return compounds.merge(existing, on="compound_id", how="left", validate="one_to_one")
    cache = cached_novelty_by_smiles()
    unique_smiles = compounds["canonical_smiles"].drop_duplicates().astype(str).tolist()
    uncached = [value for value in unique_smiles if value not in cache]
    print(f"novelty_unique={len(unique_smiles)} cached={len(unique_smiles)-len(uncached)} uncached={len(uncached)}", flush=True)
    cache.update(scan_uncached(uncached))
    novelty = compounds[["compound_id", "canonical_smiles"]].copy()
    novelty["nearest_train_similarity"] = novelty["canonical_smiles"].map(cache).astype(float)
    novelty["average_novelty_1_minus_tanimoto"] = 1.0 - novelty["nearest_train_similarity"]
    novelty["exact_training_structure_novel"] = novelty["nearest_train_similarity"].lt(1.0 - 1e-12)
    atomic_csv(novelty[["compound_id", "nearest_train_similarity", "average_novelty_1_minus_tanimoto", "exact_training_structure_novel"]], final_path)
    return compounds.merge(
        novelty[["compound_id", "nearest_train_similarity", "average_novelty_1_minus_tanimoto", "exact_training_structure_novel"]],
        on="compound_id", how="left", validate="one_to_one",
    )


def summarize(compounds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = compounds.groupby(["pair_key", "target_pair", "method", "source_seed"], sort=False).agg(
        n=("dual_worst_vina", "size"),
        best=("dual_worst_vina", "min"),
        mean=("dual_worst_vina", "mean"),
        median=("dual_worst_vina", "median"),
        variance=("dual_worst_vina", "var"),
        dual_pass_n=("dual_pass", "sum"),
        dual_pass_rate=("dual_pass", "mean"),
        qed=("qed", "mean"), sa=("sa", "mean"), logp=("logp", "mean"),
        average_novelty=("average_novelty_1_minus_tanimoto", "mean"),
        exact_structure_novelty_rate=("exact_training_structure_novel", "mean"),
    ).reset_index()
    if len(seed) != 120 or not seed["n"].eq(10).all():
        raise RuntimeError("Expected 120 seed rows with n=10")
    method = compounds.groupby(["pair_key", "target_pair", "method"], sort=False).agg(
        n=("dual_worst_vina", "size"),
        best=("dual_worst_vina", "min"),
        mean=("dual_worst_vina", "mean"),
        median=("dual_worst_vina", "median"),
        variance=("dual_worst_vina", "var"),
        dual_pass_n=("dual_pass", "sum"),
        dual_pass_rate=("dual_pass", "mean"),
        qed=("qed", "mean"), sa=("sa", "mean"), logp=("logp", "mean"),
        average_novelty=("average_novelty_1_minus_tanimoto", "mean"),
        exact_structure_novelty_rate=("exact_training_structure_novel", "mean"),
    ).reset_index()
    stability = seed.groupby(["pair_key", "method"], sort=False).agg(
        seed_mean_score_sd=("mean", "std"),
        seed_mean_score_min=("mean", "min"),
        seed_mean_score_max=("mean", "max"),
        seed_pass_rate_sd=("dual_pass_rate", "std"),
        seeds_with_at_least_8_of_10_pass=("dual_pass_n", lambda x: int((x >= 8).sum())),
        seeds_with_10_of_10_pass=("dual_pass_n", lambda x: int((x == 10).sum())),
    ).reset_index()
    method = method.merge(stability, on=["pair_key", "method"], validate="one_to_one")
    method["method"] = pd.Categorical(method["method"], METHOD_ORDER, ordered=True)
    seed["method"] = pd.Categorical(seed["method"], METHOD_ORDER, ordered=True)
    method["pair_key"] = pd.Categorical(method["pair_key"], PAIR_ORDER, ordered=True)
    seed["pair_key"] = pd.Categorical(seed["pair_key"], PAIR_ORDER, ordered=True)
    method = method.sort_values(["pair_key", "method"]).reset_index(drop=True)
    seed = seed.sort_values(["pair_key", "method", "source_seed"]).reset_index(drop=True)
    return method, seed


def main() -> None:
    compounds = attach_novelty(load_compounds())
    method, seed = summarize(compounds)
    atomic_csv(compounds, ROOT / "docking_top10_compound_summary.csv")
    atomic_csv(seed, ROOT / "docking_top10_seed_summary.csv")
    atomic_csv(method, ROOT / "docking_top10_method_summary.csv")
    print("METHOD_SUMMARY_BEGIN", flush=True)
    print(method.to_csv(index=False), flush=True)
    print("METHOD_SUMMARY_END", flush=True)


if __name__ == "__main__":
    main()
