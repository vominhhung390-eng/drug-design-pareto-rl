from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
SEEDS = range(42, 52)


def specs(pair: str):
    if pair == "EGFR-VEGFR2":
        own_root = PROJECT / "results/own_method_v4/common_seeds_42_51_10240"
        baseline_root = PROJECT / "results/baselines"
        own_run = "v4_b_raw_mean_seed{seed}"
        own_rel = "evaluation/quality_constrained/quality_annotated_molecules.csv"
    else:
        base = PROJECT / "results/target_pairs/parp1_brd4_egfr_vegfr2_aligned_20260827"
        own_root = base / "own_method"
        baseline_root = base / "baselines"
        own_run = "formal_10240_seed{seed}"
        own_rel = "anytime/budget_10240/quality_constrained/quality_annotated_molecules.csv"

    yield "CLOVER-Mol", own_root, own_run, own_rel
    for label, folder in (
        ("POLYGON", "polygon_original"),
        ("REINVENT4", "reinvent4"),
        ("DrugEx v2", "drugex_v2"),
        ("MO-LSO", "mo_lso"),
        ("GraphPareto-NSGA-II", "graphpareto_nsga2"),
    ):
        yield (
            label,
            baseline_root / folder,
            "formal_10240_seed{seed}",
            "anytime/budget_10240/quality_constrained/quality_annotated_molecules.csv",
        )


rows = []
for pair in ("EGFR-VEGFR2", "PARP1-BRD4"):
    for method, root, run_pattern, relative in specs(pair):
        passed_smiles = []
        dual6_smiles = []
        dual7_smiles = []
        rows_all = 0
        rows_pass = 0
        files = 0
        for seed in SEEDS:
            path = root / run_pattern.format(seed=seed) / relative
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(
                path,
                usecols=["smiles", "quality_pass", "dual_active_6", "dual_active_7"],
            )
            files += 1
            rows_all += len(frame)
            passed = frame.loc[
                frame["quality_pass"].astype(str).str.lower().eq("true"), "smiles"
            ].dropna()
            rows_pass += len(passed)
            passed_smiles.extend(passed.astype(str).tolist())
            quality_mask = frame["quality_pass"].astype(str).str.lower().eq("true")
            dual6_mask = frame["dual_active_6"].astype(str).str.lower().eq("true")
            dual7_mask = frame["dual_active_7"].astype(str).str.lower().eq("true")
            dual6_smiles.extend(
                frame.loc[quality_mask & dual6_mask, "smiles"].dropna().astype(str).tolist()
            )
            dual7_smiles.extend(
                frame.loc[quality_mask & dual7_mask, "smiles"].dropna().astype(str).tolist()
            )
        rows.append(
            {
                "target_pair": pair,
                "method": method,
                "seed_files": files,
                "valid_unique_rows_across_seeds": rows_all,
                "quality_pass_rows": rows_pass,
                "quality_pass_unique_smiles": len(set(passed_smiles)),
                "quality_pass_dual6_unique": len(set(dual6_smiles)),
                "quality_pass_dual7_unique": len(set(dual7_smiles)),
            }
        )

result = pd.DataFrame(rows)
print(result.to_csv(index=False))
