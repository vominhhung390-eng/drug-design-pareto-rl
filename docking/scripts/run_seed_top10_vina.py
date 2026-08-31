from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import run_vina_base as base


PROJECT = Path(__file__).resolve().parents[2]
TOP10 = PROJECT / "docking/seed_top10_two_pairs_20260830"


def structure_key(task: dict) -> tuple[str, str, str]:
    return (str(task["pair_key"]), str(task["target"]), str(task["canonical_smiles"]))


def clone_result(source: dict, task: dict) -> dict:
    clone = dict(source)
    clone.update(
        {
            "task_key": task["task_key"],
            "pair_key": task["pair_key"],
            "target": task["target"],
            "compound_id": task["compound_id"],
            "method": task["method"],
            "source_seed": task["source_seed"],
            "canonical_smiles": task["canonical_smiles"],
            "ligand": task["ligand"],
            "out_file": task["out_file"],
            "log_file": task["log_file"],
            "stdout_file": task["stdout_file"],
            "reused_from_task_key": source["task_key"],
        }
    )
    for key in ("out_file", "log_file", "stdout_file"):
        src = Path(source[key])
        dst = Path(task[key])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dst.resolve() and not dst.exists():
                shutil.copy2(src, dst)
    return clone


def run_deduplicated() -> None:
    manifest = json.loads((TOP10 / "ligand_prep_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("errors"):
        raise SystemExit("ligand preparation manifest still contains errors")
    selected = pd.read_csv(TOP10 / "selected_compounds.csv", encoding="utf-8-sig")
    tasks = []
    index = 0
    for row in selected.itertuples(index=False):
        for target_name in base.TARGETS[str(row.pair_key)]:
            tasks.append(base.task_record(row, target_name, index))
            index += 1

    partial_path = TOP10 / "docking_raw_results.partial.json"
    records = {}
    if partial_path.exists():
        records = {
            str(item["task_key"]): item
            for item in json.loads(partial_path.read_text(encoding="utf-8"))
        }
    cache = {
        structure_key(item): item
        for item in records.values()
        if base.reusable(item)
    }
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for task in tasks:
        old = records.get(task["task_key"])
        if old and base.reusable(old):
            cache.setdefault(structure_key(task), old)
            continue
        cached = cache.get(structure_key(task))
        if cached:
            records[task["task_key"]] = clone_result(cached, task)
            continue
        groups.setdefault(structure_key(task), []).append(task)

    representatives = [items[0] for items in groups.values()]
    print(
        f"tasks_total={len(tasks)} reusable_or_cached={len(tasks)-sum(len(v) for v in groups.values())} "
        f"pending_records={sum(len(v) for v in groups.values())} unique_dockings={len(representatives)} "
        f"workers={base.MAX_PARALLEL}",
        flush=True,
    )
    completed_unique = 0
    if representatives:
        with ThreadPoolExecutor(max_workers=base.MAX_PARALLEL) as pool:
            futures = {pool.submit(base.run_one, task): task for task in representatives}
            for future in as_completed(futures):
                representative = futures[future]
                result = future.result()
                group = groups[structure_key(representative)]
                records[result["task_key"]] = result
                for duplicate in group[1:]:
                    records[duplicate["task_key"]] = clone_result(result, duplicate)
                completed_unique += 1
                base.atomic_json(
                    partial_path,
                    [records[t["task_key"]] for t in tasks if t["task_key"] in records],
                )
                errors = sum(1 for item in records.values() if item.get("status") == "error")
                print(
                    f"unique_completed={completed_unique}/{len(representatives)} "
                    f"records_persisted={len(records)}/{len(tasks)} errors={errors}",
                    flush=True,
                )

    ordered = [records[t["task_key"]] for t in tasks if t["task_key"] in records]
    base.atomic_json(partial_path, ordered)
    base.atomic_json(TOP10 / "docking_raw_results.json", ordered)
    errors = [item for item in ordered if item.get("status") != "ok"]
    if errors:
        base.atomic_json(TOP10 / "docking_errors.json", errors)
        raise SystemExit(f"docking completed with {len(errors)} errors")
    if len(ordered) != len(tasks):
        raise SystemExit(f"incomplete record set: {len(ordered)}/{len(tasks)}")
    print(f"docking_complete={len(ordered)}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    TOP10.mkdir(parents=True, exist_ok=True)
    base.HERE = TOP10
    base.MAX_PARALLEL = 8
    base.VINA_CPU = 4
    run_deduplicated()
