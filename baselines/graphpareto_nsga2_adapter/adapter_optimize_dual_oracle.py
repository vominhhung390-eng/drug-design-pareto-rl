"""Run the official MolecularGraphPareto NSGA-II under the shared oracle protocol.

The upstream checkout is kept pristine.  This adapter imports and uses its
Molecule, Mutator, Crossover, and Arbiter classes directly.  The Pareto-front
and crowding-distance routines below are a source-faithful port of the methods
in upstream_official/nsga-ii/nsga2.py at commit 826e533b.  Only data loading,
the two-objective oracle, exact-budget accounting, and restartable logging are
adapted to the common experiment protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import gmean


ADAPTER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ADAPTER_ROOT.parents[1]
UPSTREAM_ROOT = ADAPTER_ROOT / "upstream_official"
NSGA_ROOT = UPSTREAM_ROOT / "nsga-ii"
sys.path.insert(0, str(NSGA_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common.oracle_bridge import DualTargetOracle  # noqa: E402
from baselines.common.oracle_ledger import OracleLedger  # noqa: E402
from nsga2.base import Molecule  # noqa: E402
from nsga2.infrastructure import Arbiter  # noqa: E402
from nsga2.operations import Crossover, Mutator  # noqa: E402


UPSTREAM_COMMIT = "826e533b1b3995a8944e7c5cefe087806ff8c03f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
            text=True,
            timeout=10,
        ).strip()
    except Exception:
        return "unavailable"


def read_smiles(path: Path) -> list[str]:
    values: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip().split()[0] if line.strip() else ""
            if value:
                values.append(value)
    return values


def initial_candidates(dataset: Path, initial_size: int) -> list[Molecule]:
    values = read_smiles(dataset)
    if len(values) < initial_size:
        raise ValueError(f"Dataset has {len(values)} rows, below initial_size={initial_size}")
    indices = np.random.choice(len(values), size=initial_size, replace=False)
    molecules: list[Molecule] = []
    for index in indices:
        mol = Chem.MolFromSmiles(values[int(index)])
        if mol is not None:
            molecules.append(Molecule(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)))
    if not molecules:
        raise RuntimeError("No valid molecules were sampled from the source dataset")
    return molecules


def calculate_fronts(molecules: Sequence[Molecule]) -> list[list[Molecule]]:
    """Official NSGA-II non-dominated sorting, preserved from nsga2.py."""

    fitnesses = np.array([molecule.fitnesses for molecule in molecules])
    domination_sets = []
    domination_counts = []
    for fitnesses_a in fitnesses:
        current_domination_set = set()
        domination_counts.append(0)
        for index, fitnesses_b in enumerate(fitnesses):
            if np.all(fitnesses_a >= fitnesses_b) and np.any(fitnesses_a > fitnesses_b):
                current_domination_set.add(index)
            elif np.all(fitnesses_b >= fitnesses_a) and np.any(fitnesses_b > fitnesses_a):
                domination_counts[-1] += 1
        domination_sets.append(current_domination_set)
    domination_counts = np.array(domination_counts)
    fronts = []
    while True:
        current_front = np.where(domination_counts == 0)[0]
        if len(current_front) == 0:
            break
        fronts.append(current_front)
        for individual in current_front:
            domination_counts[individual] = -1
            for dominated_by_current in domination_sets[individual]:
                domination_counts[dominated_by_current] -= 1
    molecular_fronts = [list(map(molecules.__getitem__, front)) for front in fronts]
    for index, molecular_front in enumerate(molecular_fronts):
        for molecule in molecular_front:
            molecule.rank = index
    return molecular_fronts


def extrema(molecules: Sequence[Molecule]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([molecule.fitnesses for molecule in molecules], axis=0)
    return stacked.min(axis=0), stacked.max(axis=0)


def assign_crowding_distance(
    fronts: Sequence[list[Molecule]], minimal_values: np.ndarray, maximal_values: np.ndarray
) -> None:
    """Official crowding-distance calculation, including its persistent field."""

    for front in fronts:
        for dimension in range(len(fronts[0][0].fitnesses)):
            value_range = maximal_values[dimension] - minimal_values[dimension]
            if value_range == 0.0:
                value_range = 1.0
            normalized = (
                np.array([molecule.fitnesses[dimension] for molecule in front])
                - minimal_values[dimension]
            ) / value_range
            ordered = [
                (value, molecule)
                for value, molecule in sorted(zip(normalized, front), key=lambda item: item[0])
            ]
            if len(ordered) == 1:
                ordered[0][1].crowding_distance = np.inf
                continue
            for index, pair in enumerate(ordered[1:-1]):
                pair[1].crowding_distance += ordered[index + 2][0] - ordered[index][0]
            ordered[0][1].crowding_distance = np.inf
            ordered[-1][1].crowding_distance = np.inf


def selection(molecules: list[Molecule], population_size: int) -> list[Molecule]:
    """Official front-first, crowding-distance NSGA-II survivor selection."""

    if not molecules:
        raise RuntimeError("NSGA-II selection received an empty candidate set")
    selected: list[Molecule] = []
    fronts = calculate_fronts(molecules)
    minimal_values, maximal_values = extrema(molecules)
    assign_crowding_distance(fronts, minimal_values, maximal_values)
    for front in fronts:
        if len(selected) + len(front) > population_size:
            front.sort(key=lambda item: item.crowding_distance, reverse=True)
            selected.extend(front[: population_size - len(selected)])
        else:
            selected.extend(front)
    return selected


def sample_mutation_parents(molecules: list[Molecule], batch_size: int) -> list[Molecule]:
    pairs = [(molecule, float(np.mean(molecule.fitnesses))) for molecule in molecules]
    candidates, weights = map(list, zip(*pairs))
    return random.choices(candidates, k=batch_size, weights=weights)


def sample_crossover_pairs(
    molecules: list[Molecule], batch_size: int
) -> list[tuple[Molecule, Molecule]]:
    pairs = [(molecule, float(gmean(molecule.fitnesses))) for molecule in molecules]
    candidates, weights = map(list, zip(*pairs))
    sampled = random.choices(candidates, k=batch_size, weights=weights)
    array = np.empty(len(sampled), dtype=object)
    array[:] = sampled
    sampled_pairs = np.random.choice(array, size=(batch_size, 2), replace=True)
    return [tuple(pair) for pair in sampled_pairs]


def generate_offspring(
    population: list[Molecule], mutator: Mutator, crossover: Crossover, batch_size: int
) -> tuple[list[Molecule], int, int]:
    generated: list[Molecule] = []
    mutation_count = 0
    crossover_count = 0
    for parent in sample_mutation_parents(population, batch_size):
        products = mutator(parent)
        mutation_count += len(products)
        generated.extend(products)
    for pair in sample_crossover_pairs(population, batch_size):
        products = crossover(pair)
        crossover_count += len(products)
        generated.extend(products)
    return generated, mutation_count, crossover_count


def score_then_arbitrate(
    molecules: list[Molecule],
    arbiter: Arbiter,
    ledger: OracleLedger,
    *,
    phase: str,
    iteration: int,
) -> list[Molecule]:
    """Count terminal proposals, then apply GraphPareto's native arbiter."""

    if not molecules or ledger.exhausted:
        return []
    molecules = molecules[: ledger.remaining]
    molecules = Arbiter.neutralize(molecules)
    results, raw_view = ledger.score(
        [molecule.smiles for molecule in molecules], phase=phase, iteration=iteration
    )
    valid_molecules: list[Molecule] = []
    for molecule, result in zip(molecules, results):
        if result.valid:
            molecule.fitnesses = [float(result.egfr), float(result.vegfr2)]
            valid_molecules.append(molecule)
    return arbiter(valid_molecules)


def pareto_size(population: list[Molecule]) -> int:
    return len(calculate_fronts(population)[0]) if population else 0


def write_archive(path: Path, population: list[Molecule]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["smiles", "egfr", "vegfr2", "rank", "crowding_distance"],
        )
        writer.writeheader()
        for molecule in population:
            writer.writerow(
                {
                    "smiles": molecule.smiles,
                    "egfr": molecule.fitnesses[0],
                    "vegfr2": molecule.fitnesses[1],
                    "rank": molecule.rank,
                    "crowding_distance": molecule.crowding_distance,
                }
            )


def atomic_pickle(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_metrics(path: Path, metrics: Iterable[dict]) -> None:
    path.write_text(json.dumps(list(metrics), indent=2), encoding="utf-8")


def metadata_extras(args: argparse.Namespace, *, generation: int, elapsed: float) -> dict:
    source_files = {
        "nsga2.py": NSGA_ROOT / "nsga2.py",
        "operations.py": NSGA_ROOT / "nsga2" / "operations.py",
        "infrastructure.py": NSGA_ROOT / "nsga2" / "infrastructure.py",
    }
    return {
        "method": "GraphPareto-NSGA-II",
        "seed": args.seed,
        "dataset": str(args.dataset.resolve()),
        "upstream_repository": "https://github.com/Jonas-Verhellen/MolecularGraphPareto",
        "upstream_commit": upstream_commit(),
        "expected_upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": {name: sha256(path) for name, path in source_files.items()},
        "algorithm": "official graph mutation + crossover + NSGA-II survivor selection",
        "objective_handling": "two raw frozen-oracle outputs; maximize both; no scalar reward",
        "budget_unit": "terminal proposals before method-native deduplication and structural filtering",
        "initial_size": args.initial_size,
        "population_size": args.population_size,
        "batch_size": args.batch_size,
        "generation": generation,
        "elapsed_seconds": elapsed,
        "native_arbiter_rules": ["Glaxo", "halogenicity", "Veber", "MW/logP/TPSA/MR"],
        "adapter_changes": [
            "shared source SMILES dataset",
            "shared frozen dual-target oracle selected through the common bridge",
            "exact terminal-proposal budget",
            "seed control, checkpointing, and common result schema",
            "portable Python reporting in place of pygmo/MultipleComparisons statistics only",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "train_smiles_only.txt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-size", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--oracle-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-stalled-generations", type=int, default=100)
    args = parser.parse_args()

    if args.budget <= 0:
        raise ValueError("budget must be positive")
    if args.initial_size <= 0 or args.population_size <= 0 or args.batch_size <= 0:
        raise ValueError("initial, population, and batch sizes must be positive")
    if upstream_commit() != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Official checkout revision mismatch: {upstream_commit()} != {UPSTREAM_COMMIT}"
        )

    RDLogger.DisableLog("rdApp.*")
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pkl"
    generated_path = output_dir / "generated.csv"
    metadata_path = output_dir / "metadata.json"
    metrics_path = output_dir / "iterations.json"

    oracle = DualTargetOracle()
    for model in (oracle.egfr_model, oracle.vegfr2_model):
        if hasattr(model, "n_jobs"):
            model.n_jobs = args.oracle_threads
    ledger = OracleLedger(args.budget, generated_path, oracle=oracle, resume=False)

    original_cwd = Path.cwd()
    os.chdir(NSGA_ROOT)
    try:
        arbiter = Arbiter(SimpleNamespace(rules=["Glaxo"]))
        mutator = Mutator(
            SimpleNamespace(data_file=str((NSGA_ROOT / "data/smarts/mutation_collection.tsv").resolve()))
        )
        crossover = Crossover()

        elapsed_before = 0.0
        if args.resume and checkpoint_path.exists():
            with checkpoint_path.open("rb") as handle:
                state = pickle.load(handle)
            if state["budget"] != args.budget or state["seed"] != args.seed:
                raise RuntimeError("Checkpoint does not match requested seed/budget")
            population = state["population"]
            ledger.records = state["ledger_records"]
            ledger.flush()
            arbiter.cache_smiles = state["arbiter_cache"]
            random.setstate(state["python_random_state"])
            np.random.set_state(state["numpy_random_state"])
            generation = state["generation"]
            stalled = state["stalled"]
            metrics = state["metrics"]
            elapsed_before = state["elapsed_seconds"]
        else:
            raw_initial = initial_candidates(args.dataset.resolve(), args.initial_size)
            accepted_initial = score_then_arbitrate(
                raw_initial, arbiter, ledger, phase="initialization", iteration=0
            )
            if not accepted_initial:
                raise RuntimeError("GraphPareto arbiter rejected the entire initial sample")
            population = selection(accepted_initial, args.population_size)
            generation = 1
            stalled = 0
            metrics: list[dict] = []

        started = time.time()
        while not ledger.exhausted:
            before = ledger.used
            raw_offspring, mutation_count, crossover_count = generate_offspring(
                population, mutator, crossover, args.batch_size
            )
            accepted = score_then_arbitrate(
                raw_offspring,
                arbiter,
                ledger,
                phase="optimization",
                iteration=generation,
            )
            if accepted:
                population = selection(population + accepted, args.population_size)
            stalled = stalled + 1 if ledger.used == before else 0
            if stalled >= args.max_stalled_generations:
                raise RuntimeError(
                    f"No terminal proposals for {stalled} consecutive generations"
                )

            elapsed = elapsed_before + time.time() - started
            row = {
                "generation": generation,
                "oracle_used": ledger.used,
                "oracle_budget": ledger.budget,
                "progress_percent": 100.0 * ledger.used / ledger.budget,
                "mutation_products": mutation_count,
                "crossover_products": crossover_count,
                "accepted_offspring": len(accepted),
                "population_size": len(population),
                "pareto_size": pareto_size(population),
                "best_min_raw_activity": max(
                    min(molecule.fitnesses) for molecule in population
                ),
                "elapsed_seconds": elapsed,
            }
            metrics.append(row)
            write_metrics(metrics_path, metrics)
            write_archive(output_dir / "archive_latest.csv", population)
            generation += 1
            state = {
                "budget": args.budget,
                "seed": args.seed,
                "population": population,
                "ledger_records": ledger.records,
                "arbiter_cache": arbiter.cache_smiles,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "generation": generation,
                "stalled": stalled,
                "metrics": metrics,
                "elapsed_seconds": elapsed,
            }
            atomic_pickle(checkpoint_path, state)
            ledger.write_metadata(
                metadata_path,
                **metadata_extras(args, generation=generation, elapsed=elapsed),
                population_size_final=len(population),
                pareto_size_final=row["pareto_size"],
            )
            print(json.dumps(row), flush=True)

        elapsed = elapsed_before + time.time() - started
        final_front = calculate_fronts(population)[0]
        write_archive(output_dir / "pareto_front.csv", final_front)
        ledger.write_metadata(
            metadata_path,
            **metadata_extras(args, generation=generation, elapsed=elapsed),
            population_size_final=len(population),
            pareto_size_final=len(final_front),
            status="complete",
        )
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
