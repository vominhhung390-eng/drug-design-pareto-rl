#!/usr/bin/env python
"""Build the two main paper tables from per-seed formal outputs."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EGFR_WORK = ROOT / "outputs/019f5762-58d7-7670-9168-54fe5fbeb2b3"
PARP_EXPERIMENT = ROOT / "results/target_pairs/parp1_brd4_egfr_vegfr2_aligned_20260827"
METHOD_ORDER = ["CLOVER-Mol", "POLYGON", "REINVENT4", "DrugEx v2", "MO-LSO", "GraphPareto-NSGA-II"]
METHOD_ALIASES = {
    "Ours (V4)": "CLOVER-Mol",
    "Ours (V4-B)": "CLOVER-Mol",
    "GraphPareto–NSGA-II": "GraphPareto-NSGA-II",
}


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def normalize_methods(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["method"] = frame["method"].astype(str).replace(METHOD_ALIASES)
    return frame


def lookup(aggregate: pd.DataFrame, method: str, metric: str) -> tuple[float, float, int]:
    row = aggregate[(aggregate["method"] == method) & (aggregate["metric"] == metric)]
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one row for {method}/{metric}; found {len(row)}")
    item = row.iloc[0]
    return float(item["mean"]), float(item["sd"]), int(item["n"])


def fmt(value: tuple[float, float, int], percent: bool = False, digits: int = 4) -> str:
    mean, sd, _ = value
    if percent:
        return f"{100 * mean:.2f}±{100 * sd:.2f}%"
    return f"{mean:.{digits}f}±{sd:.{digits}f}"


def make_markdown(pair: str, aggregate: pd.DataFrame) -> str:
    lines = [
        f"# {pair} 主实验表",
        "",
        "数值为10个正式种子（42–51）的均值±样本标准差。POLYGON使用与CLOVER-Mol共享的增强数据VAE起点，不代表无增强的严格原版POLYGON。",
        "",
        "## Table 2：主要生成性能",
        "",
        "| Method | Validity↑ | Uniqueness↑ | Novelty↑ | Diversity↑ | HV↑ | IGD+↓ | Pareto Size↑ | Dual@6↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        lines.append(
            f"| {method} | {fmt(lookup(aggregate, method, 'validity'), True)} | "
            f"{fmt(lookup(aggregate, method, 'uniqueness'), True)} | "
            f"{fmt(lookup(aggregate, method, 'novelty'), True)} | "
            f"{fmt(lookup(aggregate, method, 'diversity'))} | "
            f"{fmt(lookup(aggregate, method, 'hypervolume'))} | "
            f"{fmt(lookup(aggregate, method, 'igd_plus'))} | "
            f"{fmt(lookup(aggregate, method, 'pareto_size'), digits=1)} | "
            f"{fmt(lookup(aggregate, method, 'dual_at_6'), True)} |"
        )
    lines.extend([
        "", "## Table 3：质量约束后的结果", "",
        "| Method | Quality pass↑ | Alert-free↑ | Scaffold diversity↑ | QC-HV↑ | QC-Dual@6↑ | QC-Dual@7↑ | QC best-min↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in METHOD_ORDER:
        lines.append(
            f"| {method} | {fmt(lookup(aggregate, method, 'quality_pass'), True)} | "
            f"{fmt(lookup(aggregate, method, 'alert_free'), True)} | "
            f"{fmt(lookup(aggregate, method, 'scaffold_diversity'), True)} | "
            f"{fmt(lookup(aggregate, method, 'qc_hypervolume'))} | "
            f"{fmt(lookup(aggregate, method, 'qc_dual_at_6'), True)} | "
            f"{fmt(lookup(aggregate, method, 'qc_dual_at_7'), True)} | "
            f"{fmt(lookup(aggregate, method, 'qc_best_min'), digits=3)} |"
        )
    return "\n".join(lines) + "\n"


def validate_completion(frame: pd.DataFrame, pair: str) -> None:
    frame = normalize_methods(frame)
    missing = []
    for method in METHOD_ORDER:
        row = frame[frame["method"] == method]
        if len(row) != 1 or int(row.iloc[0]["completed_seeds"]) != 10:
            value = "absent" if row.empty else str(int(row.iloc[0]["completed_seeds"]))
            missing.append(f"{method}={value}/10")
    if missing:
        raise RuntimeError(f"{pair} incomplete: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper_tables")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    run([sys.executable, str(ROOT / "analysis/build_main_experiment_metrics.py")])
    cache = EGFR_WORK / "training_smiles_canonical.txt.gz"
    parp_out = output / "parp1_brd4_intermediate"
    run([
        sys.executable,
        str(ROOT / "analysis/build_target_pair_current_metrics.py"),
        str(PARP_EXPERIMENT), str(cache), str(parp_out),
    ])

    egfr_aggregate = normalize_methods(pd.read_csv(EGFR_WORK / "aggregate_metrics.csv"))
    egfr_per_seed = normalize_methods(pd.read_csv(EGFR_WORK / "per_seed_metrics.csv"))
    egfr_completion = normalize_methods(pd.DataFrame({
        "method": METHOD_ORDER,
        "completed_seeds": [int((egfr_per_seed["method"] == method).sum()) for method in METHOD_ORDER],
        "planned_seeds": 10,
    }))
    parp_aggregate = normalize_methods(pd.read_csv(parp_out / "aggregate_metrics.csv"))
    parp_per_seed = normalize_methods(pd.read_csv(parp_out / "per_seed_metrics.csv"))
    parp_completion = normalize_methods(pd.read_csv(parp_out / "completion.csv"))
    validate_completion(egfr_completion, "EGFR/VEGFR2")
    validate_completion(parp_completion, "PARP1/BRD4")

    egfr_aggregate.assign(target_pair="EGFR_VEGFR2").to_csv(
        output / "EGFR_VEGFR2_aggregate_metrics.csv", index=False, encoding="utf-8-sig"
    )
    parp_aggregate.assign(target_pair="PARP1_BRD4").to_csv(
        output / "PARP1_BRD4_aggregate_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([
        egfr_per_seed.assign(target_pair="EGFR_VEGFR2"),
        parp_per_seed.assign(target_pair="PARP1_BRD4"),
    ], ignore_index=True).to_csv(output / "all_per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "Table2_Table3_EGFR_VEGFR2.md").write_text(
        make_markdown("EGFR/VEGFR2", egfr_aggregate), encoding="utf-8"
    )
    (output / "Table2_Table3_PARP1_BRD4.md").write_text(
        make_markdown("PARP1/BRD4", parp_aggregate), encoding="utf-8"
    )
    shutil.copy2(EGFR_WORK / "igd_reference_front.csv", output / "EGFR_VEGFR2_igd_reference_front.csv")
    shutil.copy2(parp_out / "igd_reference_front.csv", output / "PARP1_BRD4_igd_reference_front.csv")
    metadata = {
        "seeds": list(range(42, 52)),
        "budget_per_seed": 10240,
        "methods": METHOD_ORDER,
        "replication_unit": "formal seed; never pooled before metric calculation",
        "polygon_identity": "POLYGON adapter using the shared augmented-data VAE initialization",
        "igd_reference": "pooled nondominated front within each target pair across all six methods and ten seeds",
        "quality_gate": "QED>=0.60; SA<=4.0; no PAINS/Brenk alert; Lipinski pass",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TABLES_COMPLETE output={output}")


if __name__ == "__main__":
    main()
