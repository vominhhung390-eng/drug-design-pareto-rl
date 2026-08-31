from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from rdkit import Chem


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "docking/seed_top10_two_pairs_20260830"
TOP_K = 10
SEEDS = range(42, 52)
METHODS = (
    ("CLOVER-Mol", "CLV", "own"),
    ("POLYGON", "POL", "polygon_original"),
    ("REINVENT4", "RNV", "reinvent4"),
    ("DrugEx v2", "DRX", "drugex_v2"),
    ("MO-LSO", "MLS", "mo_lso"),
    ("GraphPareto-NSGA-II", "GPN", "graphpareto_nsga2"),
)


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if fragments:
        mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def quality_path(pair: str, method_kind: str, seed: int) -> Path:
    if pair == "pairA":
        if method_kind == "own":
            return (
                PROJECT
                / "results/own_method_v4/common_seeds_42_51_10240"
                / f"v4_b_raw_mean_seed{seed}"
                / "evaluation/quality_constrained/quality_annotated_molecules.csv"
            )
        return (
            PROJECT
            / "results/baselines"
            / method_kind
            / f"formal_10240_seed{seed}"
            / "anytime/budget_10240/quality_constrained/quality_annotated_molecules.csv"
        )
    base = PROJECT / "results/target_pairs/parp1_brd4_egfr_vegfr2_aligned_20260827"
    if method_kind == "own":
        return (
            base
            / "own_method"
            / f"formal_10240_seed{seed}"
            / "anytime/budget_10240/quality_constrained/quality_annotated_molecules.csv"
        )
    return (
        base
        / "baselines"
        / method_kind
        / f"formal_10240_seed{seed}"
        / "anytime/budget_10240/quality_constrained/quality_annotated_molecules.csv"
    )


def select_method(pair: str, method: str, prefix: str, kind: str) -> tuple[list[dict], dict]:
    selected: list[dict] = []
    audit = {"method": method, "pair": pair, "seeds": {}, "selected": 0}
    target_pair_label = "EGFR/VEGFR2" if pair == "pairA" else "PARP1/BRD4"
    targets = ("egfr", "vegfr2")

    for seed in SEEDS:
        # The seed is the independent experimental unit.  Deduplicate only
        # within that seed; a structure rediscovered by another seed remains a
        # valid repeated outcome (its docking calculation can be cached later).
        seen: set[str] = set()
        path = quality_path(pair, kind, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"smiles", "quality_pass", "qed", "sa", "logp", *targets}
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        frame = frame[frame["quality_pass"].astype(str).str.lower().eq("true")].copy()
        frame["canonical_smiles"] = frame["smiles"].map(canonicalize)
        frame = frame.dropna(subset=["canonical_smiles"])
        frame["min_predicted_activity"] = frame[list(targets)].astype(float).min(axis=1)
        frame["mean_predicted_activity"] = frame[list(targets)].astype(float).mean(axis=1)
        frame = frame.sort_values(
            ["min_predicted_activity", "mean_predicted_activity", "qed", "sa"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        before = len(frame)
        seed_selected = 0
        for row in frame.itertuples(index=False):
            smiles = str(row.canonical_smiles)
            if smiles in seen:
                continue
            seen.add(smiles)
            seed_selected += 1
            selected.append(
                {
                    "pair_key": pair,
                    "target_pair": target_pair_label,
                    "method": method,
                    "method_key": kind,
                    "source_seed": seed,
                    "within_seed_rank": seed_selected,
                    "compound_id": f"{prefix}_{pair[-1]}S{seed}_{seed_selected:02d}",
                    "canonical_smiles": smiles,
                    "predicted_target_1": float(getattr(row, targets[0])),
                    "predicted_target_2": float(getattr(row, targets[1])),
                    "min_predicted_activity": float(row.min_predicted_activity),
                    "mean_predicted_activity": float(row.mean_predicted_activity),
                    "qed": float(row.qed),
                    "sa": float(row.sa),
                    "logp": float(row.logp),
                    "source_file": str(path),
                }
            )
            if seed_selected == TOP_K:
                break
        audit["seeds"][str(seed)] = {
            "quality_pass_rows_available": int(before),
            "selected": int(seed_selected),
        }
        if seed_selected != TOP_K:
            raise RuntimeError(f"{pair}/{method}/seed{seed}: expected {TOP_K}, got {seed_selected}")
    audit["selected"] = len(selected)
    expected = len(SEEDS) * TOP_K
    if len(selected) != expected:
        raise RuntimeError(f"{pair}/{method}: expected {expected}, got {len(selected)}")
    return selected, audit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    audits: list[dict] = []
    jobs = [
        (pair, method, prefix, kind)
        for pair in ("pairA", "pairB")
        for method, prefix, kind in METHODS
    ]
    # Each pair/method panel is independent.  Process-level parallelism avoids
    # making the complete quality-table scan a single-core bottleneck.
    with ProcessPoolExecutor(max_workers=6) as pool:
        for picked, audit in pool.map(lambda_args_select_method, jobs):
            rows.extend(picked)
            audits.append(audit)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "selected_compounds.csv", index=False, encoding="utf-8-sig")
    (OUT / "selection_audit.json").write_text(
        json.dumps(
            {
                "selection_rule": (
                    "per-seed Top-10 among quality_pass molecules after largest-fragment "
                    "canonical SMILES; duplicates removed within each seed but retained across seeds; "
                    "ranked by min target activity, mean target activity, QED descending, SA ascending"
                ),
                "seed_ids": list(SEEDS),
                "methods": [m[0] for m in METHODS],
                "top_k_per_seed": TOP_K,
                "selected_total": int(len(frame)),
                "rows_per_pair_method": len(SEEDS) * TOP_K,
                "pairs": audits,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(frame.groupby(["pair_key", "method"]).size().to_string())
    print(f"selected_total={len(frame)}")


def lambda_args_select_method(args):
    return select_method(*args)


if __name__ == "__main__":
    main()
