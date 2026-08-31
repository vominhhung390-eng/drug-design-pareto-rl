#!/usr/bin/env python
"""Audit assay and variant heterogeneity before predictor V3 training."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


MUTATIONS = {
    "T790M": re.compile(r"\bT790M\b", re.I),
    "L858R": re.compile(r"\bL858R\b", re.I),
    "C797S": re.compile(r"\bC797S\b", re.I),
    "EXON19_DEL": re.compile(r"(?:exon\s*19|del(?:etion)?\s*19|del19|E746.{0,8}A750)", re.I),
    "OTHER_MUTANT": re.compile(
        r"\b(?:mutant|mutation|L861Q|G719[ACDSX]?|S768I|D770|V769|A763|ins(?:ertion)?)\b",
        re.I,
    ),
}
WT_PATTERN = re.compile(r"\b(?:wild[ -]?type|wt)[ -]?(?:EGFR)?\b", re.I)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def exact_base_filter(row: dict) -> bool:
    return (
        row.get("standard_type") == "IC50"
        and row.get("assay_type") == "B"
        and row.get("target_organism") == "Homo sapiens"
        and row.get("standard_flag") == 1
        and row.get("standard_relation") == "="
        and row.get("standard_units") == "nM"
        and row.get("data_validity_comment") in (None, "")
        and int(row.get("potential_duplicate") or 0) == 0
        and row.get("pchembl_value") not in (None, "")
        and row.get("document_year") not in (None, "")
        and row.get("canonical_smiles") not in (None, "")
    )


def variant_class(row: dict) -> str:
    explicit = str(row.get("assay_variant_mutation") or "").strip()
    description = str(row.get("assay_description") or "")
    text = f"{explicit} {description}"
    hits = [name for name, pattern in MUTATIONS.items() if pattern.search(text)]
    if hits:
        return "+".join(hits)
    if explicit:
        return "OTHER_EXPLICIT_VARIANT"
    if WT_PATTERN.search(description):
        return "EXPLICIT_WT"
    return "UNSPECIFIED"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "predictor_retraining_v3_20260731" / "audit",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {
        "EGFR": root / "data" / "external" / "chembl" / "CHEMBL203_activities.jsonl",
        "VEGFR2": root / "data" / "external" / "chembl" / "CHEMBL279_activities.jsonl",
    }
    summary = {}
    assay_rows = []
    for target, path in sources.items():
        raw = load_jsonl(path)
        base = [row for row in raw if exact_base_filter(row)]
        variants = Counter(variant_class(row) for row in base)
        formats = Counter(
            f"{row.get('bao_format') or 'NA'}|{row.get('bao_label') or 'NA'}" for row in base
        )
        assays: dict[str, list[dict]] = {}
        for row in base:
            assays.setdefault(str(row.get("assay_chembl_id")), []).append(row)
        for assay_id, rows in assays.items():
            values = pd.to_numeric(
                pd.Series([row.get("pchembl_value") for row in rows]), errors="coerce"
            ).dropna()
            first = rows[0]
            assay_rows.append(
                {
                    "target": target,
                    "assay_chembl_id": assay_id,
                    "n_measurements": len(rows),
                    "n_molecules": len({row.get("canonical_smiles") for row in rows}),
                    "variant_class": variant_class(first),
                    "bao_format": first.get("bao_format"),
                    "bao_label": first.get("bao_label"),
                    "document_year_min": min(int(row["document_year"]) for row in rows),
                    "document_year_max": max(int(row["document_year"]) for row in rows),
                    "pactivity_median": float(values.median()),
                    "pactivity_iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                    "description": first.get("assay_description"),
                }
            )
        summary[target] = {
            "raw_rows": len(raw),
            "exact_human_binding_ic50_rows": len(base),
            "unique_molecules": len({row.get("canonical_smiles") for row in base}),
            "unique_assays": len(assays),
            "variant_classes": dict(variants.most_common()),
            "bao_formats": dict(formats.most_common()),
            "assay_size": {
                "median": float(pd.Series([len(v) for v in assays.values()]).median()),
                "ge_10": sum(len(v) >= 10 for v in assays.values()),
                "ge_30": sum(len(v) >= 30 for v in assays.values()),
            },
        }
    pd.DataFrame(assay_rows).to_csv(
        args.output / "assay_audit.csv", index=False, encoding="utf-8-sig"
    )
    (args.output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
