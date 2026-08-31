#!/usr/bin/env python
"""Build a local, DOI-ready reproducibility inventory for the V4-B paper.

This script does not publish files or invent an identifier.  It records the
critical inputs, frozen models, protocols, aggregate results, analysis code and
manuscript sources that should be included in a versioned public archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pandas
import rdkit
import scipy
import sklearn
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(rows: list[dict[str, object]], project: Path, path: Path, role: str, access: str) -> None:
    path = path.resolve()
    if not path.exists() or not path.is_file():
        rows.append(
            {
                "role": role,
                "relative_path": str(path),
                "access_route": access,
                "exists": False,
                "size_bytes": "",
                "sha256": "",
            }
        )
        return
    try:
        relative = str(path.relative_to(project))
    except ValueError:
        relative = str(path)
    rows.append(
        {
            "role": role,
            "relative_path": relative,
            "access_route": access,
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )


def gpu_info() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return f"unavailable: {exc}"


def completion_gate(project: Path) -> list[dict[str, object]]:
    seeds = range(42, 52)
    specs = [
        (
            "EGFR-VEGFR2",
            "MO-LSO",
            project / "results" / "baselines" / "mo_lso",
        ),
        (
            "PARP1-BRD4",
            "MO-LSO",
            project
            / "results"
            / "target_pairs"
            / "parp1_brd4_20260804"
            / "baselines"
            / "mo_lso",
        ),
    ]
    rows = []
    for pair, method, root in specs:
        for seed in seeds:
            run = root / f"formal_10240_seed{seed}"
            metadata_path = run / "metadata.json"
            summary_path = run / "anytime" / "budget_10240" / "evaluation_summary.json"
            metadata = {}
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "target_pair": pair,
                    "method": method,
                    "seed": seed,
                    "metadata_complete": bool(metadata.get("complete", False)),
                    "oracle_used": metadata.get("used", ""),
                    "evaluation_summary_exists": summary_path.exists(),
                    "formal_complete": bool(
                        metadata.get("complete", False)
                        and int(metadata.get("used", -1)) == 10240
                        and summary_path.exists()
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    paper = args.paper.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    critical = [
        (project / "data" / "train_smiles_only.txt", "common target-independent training dataset", "reused public source; redistribution licence to verify"),
        (project / "models" / "oracles" / "target_EGFR_model.pkl", "frozen EGFR oracle", "public archive after third-party licence audit"),
        (project / "models" / "oracles" / "target_VEGFR2_model.pkl", "frozen VEGFR2 oracle", "public archive after third-party licence audit"),
        (project / "models" / "oracles" / "parp1_brd4_20260804" / "target_PARP1_model.pkl", "frozen PARP1 oracle", "public archive after third-party licence audit"),
        (project / "models" / "oracles" / "parp1_brd4_20260804" / "target_BRD4_model.pkl", "frozen BRD4 oracle", "public archive after third-party licence audit"),
        (project / "models" / "polygon_vae_best_valid_novel_stable_020.pt", "locked VAE checkpoint shared by V4-B and POLYGON", "public archive if upstream terms permit"),
        (project / "config" / "formal_experiments.json", "primary formal protocol", "public"),
        (project / "config" / "v4_formal_ablation_10240.json", "registered single-factor ablation protocol", "public"),
        (project / "config" / "v4_p0_actor_vae_factorial_completion_10240.json", "exploratory factorial completion protocol", "public"),
        (project / "results" / "target_pairs" / "parp1_brd4_20260804" / "frozen_protocol.json", "second-pair frozen protocol", "public"),
        (project / "results" / "baselines" / "protocol_verification.json", "baseline protocol verification", "public"),
        (project / "baselines" / "COMPATIBILITY_PATCHES.md", "baseline compatibility and fidelity record", "public"),
        (project / "analysis" / "compare_v5_all_baselines_two_pairs.py", "two-pair metric reconstruction", "public"),
        (project / "analysis" / "summarize_actor_vae_factorial.py", "factorial analysis", "public"),
        (project / "analysis" / "audit_pairb_predictor_domain.py", "predictor applicability-domain analysis", "public"),
        (project / "results" / "own_method_v5_balancesync_20260806" / "analysis" / "all_baselines_two_pairs" / "aggregate_metrics.csv", "two-pair aggregate metrics", "public source data"),
        (project / "results" / "own_method_v5_balancesync_20260806" / "analysis" / "all_baselines_two_pairs" / "per_seed_metrics.csv", "two-pair per-seed metrics", "public source data"),
        (project / "results" / "own_method_v4" / "formal_ablation_10240" / "ablation_all_runs.csv", "registered ablation per-seed results", "public source data"),
        (project / "results" / "paper_p0_20260822" / "predictor_domain_audit" / "locked_2024plus_domain_summary.csv", "predictor domain summary", "public source data"),
        (project / "results" / "paper_p0_20260822" / "predictor_domain_audit" / "selected_candidates_domain_summary.csv", "candidate domain summary", "public source data"),
        (project / "docking" / "unified_7method_top5" / "docking_compound_results.csv", "EGFR-VEGFR2 docking source data", "public source data"),
        (project / "docking" / "parp1_brd4_unified_7method_top5" / "docking_compound_results.csv", "PARP1-BRD4 docking source data", "public source data"),
        (paper / "main.tex", "manuscript source", "public"),
        (paper / "references.bib", "bibliography", "public"),
    ]
    for path, role, access in critical:
        add_file(rows, project, path, role, access)
    for path in sorted((paper / "sections").glob("*.tex")):
        add_file(rows, project, path, "manuscript section", "public")
    for path in sorted((paper / "figures" / "source_data").glob("*.csv")):
        add_file(rows, project, path, "figure source data", "public source data")

    with (output / "critical_file_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    completion = completion_gate(project)
    pd_frame = pandas.DataFrame(completion)
    pd_frame.to_csv(output / "mo_lso_completion_gate.csv", index=False, encoding="utf-8-sig")
    all_complete = bool(pd_frame["formal_complete"].all())

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu_info(),
        "packages": {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }
    manifest = {
        "schema_version": "v4b-paper-reproducibility-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "doi_ready_local_inventory" if all_complete else "incomplete_experiment_inventory",
        "public_identifier": "[DOI TO BE ASSIGNED; not yet published]",
        "mo_lso_all_formal_complete": all_complete,
        "critical_files_present": all(bool(row["exists"]) for row in rows),
        "environment": environment,
        "archive_contents": {
            "data": "processed figure/table source data and reusable result summaries",
            "code": "V4-B implementation, adapters, evaluation and reconstruction scripts subject to licence audit",
            "models": "frozen oracles and checkpoints subject to upstream redistribution terms",
            "raw_third_party_data": "not redistributed unless original licences permit; stable source/version/filter metadata required",
        },
    }
    (output / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# V4-B reproducibility archive staging package

Status: `{manifest['status']}`

This local package inventories the evidence required to reproduce the manuscript. It is not a public repository and has no DOI yet.

## Required public-release structure

- `data/source_data/`: per-seed and aggregate values underlying every table and figure.
- `code/`: V4-B, baseline adapters, evaluation and plotting scripts.
- `models/`: frozen predictor and generator checkpoints that can legally be redistributed.
- `protocols/`: frozen budgets, seeds, hashes, configuration and compatibility records.
- `environment/`: package versions and installation instructions.
- `README.md`: one-command reconstruction instructions and third-party data download steps.

## Submission blockers

- Deposit the archive in Zenodo, Figshare, OSF or an institutional repository and replace `[DOI TO BE ASSIGNED]` with the assigned persistent identifier.
- Complete the third-party licence audit before redistributing MOSES/ChEMBL/BindingDB-derived files or upstream checkpoints.
- Test the archived workflow outside the author account.
- Confirm that every main and supplementary figure/table maps to an included source-data file.

## Current MO-LSO completion gate

- Both target-pair panels complete: `{all_complete}`
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    availability = """# Submission-ready availability text after repository deposit

## Data Availability

The processed source data supporting all main and supplementary figures and tables, including per-seed molecular-generation metrics, predictor validation results, ablation statistics and docking summaries, will be available in [REPOSITORY] at [DOI]. The target-independent training set was derived from the public MOSES benchmark; target activity records were obtained from the cited ChEMBL and BindingDB releases. Third-party records that cannot be redistributed will be represented by versioned download and filtering scripts, source identifiers and SHA-256 hashes.

## Code Availability

The V4-B implementation, frozen experiment configurations, five baseline adapters, oracle-budget ledger, statistical reconstruction scripts and figure-generation code will be archived in [REPOSITORY] at [DOI]. The archive will include an environment specification, model and data hashes, random seeds and instructions for reconstructing every reported table and figure. Upstream components remain subject to their original licences.

中文核对：仓库和 DOI 尚未创建，不能在正式投稿稿中把上述占位符写成已公开事实。
"""
    (output / "availability_statement_draft.md").write_text(availability, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
