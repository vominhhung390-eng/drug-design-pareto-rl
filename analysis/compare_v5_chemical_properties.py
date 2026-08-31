#!/usr/bin/env python
"""Comprehensive chemical-property audit for V4-B, V5 BalanceSync and POLYGON.

The unit of replication is a formal seed.  Each run contributes one summary
row calculated from its standardized unique-valid molecule set.  Aggregates
therefore report mean +/- SD across seeds rather than pooling molecules across
runs and treating molecules as independent replicates.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, FilterCatalog, Lipinski, rdMolDescriptors

from compare_v4c_balancesync_two_pairs import SEEDS, run_specs


PROPERTY_COLUMNS = (
    "qed_mean",
    "qed_p10",
    "qed_pass_rate",
    "sa_mean",
    "sa_p90",
    "sa_pass_rate",
    "mol_wt_mean",
    "mol_wt_p90",
    "mw_le_500_rate",
    "logp_mean",
    "logp_p90",
    "logp_le_5_rate",
    "tpsa_mean",
    "tpsa_p90",
    "tpsa_le_140_rate",
    "hbd_mean",
    "hbd_le_5_rate",
    "hba_mean",
    "hba_le_10_rate",
    "rotatable_bonds_mean",
    "rotatable_le_10_rate",
    "lipinski_pass_rate",
    "veber_pass_rate",
    "alert_free_rate",
    "quality_pass_rate",
    "scaffold_diversity",
    "scaffold_entropy",
)

SUBSET_COLUMNS = (
    "n",
    "fraction_of_unique_valid",
    "qed_mean",
    "sa_mean",
    "lipinski_pass_rate",
    "veber_pass_rate",
    "alert_free_rate",
    "quality_pass_rate",
    "scaffold_diversity",
)

RATE_SUFFIXES = ("_rate", "_diversity", "fraction_of_unique_valid")

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

_WORKER_PAINS = None
_WORKER_BRENK = None


def _init_extra_worker() -> None:
    global _WORKER_PAINS, _WORKER_BRENK
    pains_params = FilterCatalog.FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    brenk_params = FilterCatalog.FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    _WORKER_PAINS = FilterCatalog.FilterCatalog(pains_params)
    _WORKER_BRENK = FilterCatalog.FilterCatalog(brenk_params)


def _compute_extra_worker(smiles: str) -> tuple[float, ...]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (math.nan,) * 8
    return (
        float(_WORKER_PAINS.HasMatch(mol)),
        float(_WORKER_BRENK.HasMatch(mol)),
        float(Descriptors.HeavyAtomCount(mol)),
        float(Lipinski.RingCount(mol)),
        float(Lipinski.NumAromaticRings(mol)),
        float(rdMolDescriptors.CalcFractionCSP3(mol)),
        float(abs(Chem.GetFormalCharge(mol))),
        float(Chem.GetFormalCharge(mol) != 0),
    )


class ExtraDescriptorCache:
    """Cache descriptors that are absent from the quality CSVs by SMILES."""

    def __init__(self) -> None:
        pains_params = FilterCatalog.FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        brenk_params = FilterCatalog.FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        self.pains = FilterCatalog.FilterCatalog(pains_params)
        self.brenk = FilterCatalog.FilterCatalog(brenk_params)
        self.cache: dict[str, tuple[float, ...]] = {}

    def preload(self, smiles: set[str], workers: int = 6) -> None:
        missing = sorted(smiles.difference(self.cache))
        if not missing:
            return
        print(
            f"Precomputing PAINS/Brenk and extra descriptors for {len(missing)} SMILES "
            f"with {workers} workers",
            flush=True,
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_extra_worker,
        ) as executor:
            values = executor.map(_compute_extra_worker, missing, chunksize=256)
            for smiles_value, descriptor_value in zip(missing, values):
                self.cache[smiles_value] = descriptor_value

    def get(self, smiles: str) -> tuple[float, ...]:
        cached = self.cache.get(smiles)
        if cached is not None:
            return cached
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            value = (math.nan,) * 8
        else:
            value = (
                float(self.pains.HasMatch(mol)),
                float(self.brenk.HasMatch(mol)),
                float(Descriptors.HeavyAtomCount(mol)),
                float(Lipinski.RingCount(mol)),
                float(Lipinski.NumAromaticRings(mol)),
                float(rdMolDescriptors.CalcFractionCSP3(mol)),
                float(abs(Chem.GetFormalCharge(mol))),
                float(Chem.GetFormalCharge(mol) != 0),
            )
        self.cache[smiles] = value
        return value


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else math.nan


def safe_quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q)) if len(values) else math.nan


def scaffold_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    if not len(frame):
        return math.nan, math.nan
    scaffolds = frame["scaffold"].fillna("").astype(str)
    nonempty = scaffolds[scaffolds.str.len() > 0]
    diversity = float(nonempty.nunique() / len(frame))
    if not len(nonempty):
        return diversity, 0.0
    probabilities = nonempty.value_counts(normalize=True).to_numpy(float)
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return diversity, entropy


def summarize_subset(frame: pd.DataFrame, total: int) -> dict[str, float | int]:
    if not len(frame):
        return {name: (0 if name == "n" else math.nan) for name in SUBSET_COLUMNS}
    scaffold_diversity, _ = scaffold_metrics(frame)
    alert_free = ~as_bool(frame["structural_alert"])
    lipinski = as_bool(frame["lipinski_pass"])
    quality = as_bool(frame["quality_pass"])
    veber = frame["tpsa"].le(140.0) & frame["rotatable_bonds"].le(10)
    return {
        "n": int(len(frame)),
        "fraction_of_unique_valid": float(len(frame) / total) if total else math.nan,
        "qed_mean": safe_mean(frame["qed"]),
        "sa_mean": safe_mean(frame["sa"]),
        "lipinski_pass_rate": float(lipinski.mean()),
        "veber_pass_rate": float(veber.mean()),
        "alert_free_rate": float(alert_free.mean()),
        "quality_pass_rate": float(quality.mean()),
        "scaffold_diversity": scaffold_diversity,
    }


def summarize_run(
    frame: pd.DataFrame,
    extra_cache: ExtraDescriptorCache,
) -> dict[str, float | int]:
    required = {
        "smiles", "egfr", "vegfr2", "qed", "mol_wt", "logp", "scaffold",
        "structural_alert", "sa", "hbd", "hba", "rotatable_bonds", "tpsa",
        "lipinski_pass", "quality_pass",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing quality columns: {sorted(missing)}")
    if frame["smiles"].astype(str).duplicated().any():
        raise ValueError("Quality input is not unique by SMILES")

    alert_free = ~as_bool(frame["structural_alert"])
    lipinski = as_bool(frame["lipinski_pass"])
    quality = as_bool(frame["quality_pass"])
    qed_pass = frame["qed"].ge(0.60)
    sa_pass = frame["sa"].le(4.0)
    veber = frame["tpsa"].le(140.0) & frame["rotatable_bonds"].le(10)
    failures = pd.DataFrame({
        "qed": ~qed_pass,
        "sa": ~sa_pass,
        "alert": ~alert_free,
        "lipinski": ~lipinski,
    })
    failure_count = failures.sum(axis=1)
    scaffold_diversity, scaffold_entropy = scaffold_metrics(frame)
    min_activity = frame[["egfr", "vegfr2"]].min(axis=1)
    extra = np.asarray(
        [extra_cache.get(smiles) for smiles in frame["smiles"].astype(str)],
        dtype=float,
    )
    pains_alert = extra[:, 0].astype(bool)
    brenk_alert = extra[:, 1].astype(bool)
    stored_alert = as_bool(frame["structural_alert"]).to_numpy(bool)

    result: dict[str, float | int] = {
        "unique_valid": int(len(frame)),
        "qed_mean": safe_mean(frame["qed"]),
        "qed_p10": safe_quantile(frame["qed"], 0.10),
        "qed_pass_rate": float(qed_pass.mean()),
        "sa_mean": safe_mean(frame["sa"]),
        "sa_p90": safe_quantile(frame["sa"], 0.90),
        "sa_pass_rate": float(sa_pass.mean()),
        "mol_wt_mean": safe_mean(frame["mol_wt"]),
        "mol_wt_p90": safe_quantile(frame["mol_wt"], 0.90),
        "mw_le_500_rate": float(frame["mol_wt"].le(500.0).mean()),
        "logp_mean": safe_mean(frame["logp"]),
        "logp_p90": safe_quantile(frame["logp"], 0.90),
        "logp_le_5_rate": float(frame["logp"].le(5.0).mean()),
        "tpsa_mean": safe_mean(frame["tpsa"]),
        "tpsa_p90": safe_quantile(frame["tpsa"], 0.90),
        "tpsa_le_140_rate": float(frame["tpsa"].le(140.0).mean()),
        "hbd_mean": safe_mean(frame["hbd"]),
        "hbd_le_5_rate": float(frame["hbd"].le(5).mean()),
        "hba_mean": safe_mean(frame["hba"]),
        "hba_le_10_rate": float(frame["hba"].le(10).mean()),
        "rotatable_bonds_mean": safe_mean(frame["rotatable_bonds"]),
        "rotatable_le_10_rate": float(frame["rotatable_bonds"].le(10).mean()),
        "lipinski_pass_rate": float(lipinski.mean()),
        "veber_pass_rate": float(veber.mean()),
        "alert_free_rate": float(alert_free.mean()),
        "quality_pass_rate": float(quality.mean()),
        "qed_fail_rate": float(failures["qed"].mean()),
        "sa_fail_rate": float(failures["sa"].mean()),
        "alert_rate": float(failures["alert"].mean()),
        "lipinski_fail_rate": float(failures["lipinski"].mean()),
        "qed_only_fail_rate": float((failures["qed"] & failure_count.eq(1)).mean()),
        "sa_only_fail_rate": float((failures["sa"] & failure_count.eq(1)).mean()),
        "alert_only_fail_rate": float((failures["alert"] & failure_count.eq(1)).mean()),
        "lipinski_only_fail_rate": float((failures["lipinski"] & failure_count.eq(1)).mean()),
        "multiple_quality_fail_rate": float(failure_count.ge(2).mean()),
        "pains_alert_rate": float(pains_alert.mean()),
        "brenk_alert_rate": float(brenk_alert.mean()),
        "pains_and_brenk_alert_rate": float((pains_alert & brenk_alert).mean()),
        "combined_alert_mismatch_rate": float(
            np.not_equal(stored_alert, pains_alert | brenk_alert).mean()
        ),
        "heavy_atoms_mean": float(np.nanmean(extra[:, 2])),
        "ring_count_mean": float(np.nanmean(extra[:, 3])),
        "aromatic_ring_count_mean": float(np.nanmean(extra[:, 4])),
        "fraction_csp3_mean": float(np.nanmean(extra[:, 5])),
        "abs_formal_charge_mean": float(np.nanmean(extra[:, 6])),
        "charged_molecule_rate": float(np.nanmean(extra[:, 7])),
        "scaffold_diversity": scaffold_diversity,
        "scaffold_entropy": scaffold_entropy,
    }
    for prefix, threshold in (("dual6", 6.0), ("dual65", 6.5)):
        subset = frame.loc[min_activity.ge(threshold)]
        for name, value in summarize_subset(subset, len(frame)).items():
            result[f"{prefix}_{name}"] = value
    return result


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return math.nan
    observed = abs(float(differences.mean()))
    total = 0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += statistic >= observed - 1e-15
        total += 1
    return extreme / total


def collect(project: Path, config: dict) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    extra_cache = ExtraDescriptorCache()
    tasks: list[tuple[str, str, int, Path]] = []
    for pair_name in config["target_pairs"]:
        for spec in run_specs(project, config, pair_name):
            method = str(spec["method"])
            for seed in SEEDS:
                run = spec["run"](seed)
                quality_path = spec["quality"](run) / "quality_annotated_molecules.csv"
                if not quality_path.exists():
                    raise FileNotFoundError(quality_path)
                tasks.append((pair_name, method, seed, quality_path))
    all_smiles: set[str] = set()
    for _, _, _, quality_path in tasks:
        smiles = pd.read_csv(
            quality_path,
            usecols=["smiles"],
            encoding="utf-8-sig",
        )["smiles"].astype(str)
        all_smiles.update(smiles)
    extra_cache.preload(all_smiles)
    for pair_name, method, seed, quality_path in tasks:
        frame = pd.read_csv(quality_path, encoding="utf-8-sig")
        records.append({
            "target_pair": pair_name,
            "method": method,
            "seed": seed,
            **summarize_run(frame, extra_cache),
        })
        print(f"{pair_name} {method} seed {seed} audited", flush=True)
    print(f"Cached extra descriptors for {len(extra_cache.cache)} unique SMILES", flush=True)
    return pd.DataFrame(records)


def metric_names(frame: pd.DataFrame) -> list[str]:
    excluded = {"target_pair", "method", "seed"}
    return [name for name in frame.columns if name not in excluded]


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (pair, method), subset in per_seed.groupby(["target_pair", "method"], sort=False):
        for metric in metric_names(per_seed):
            values = pd.to_numeric(subset[metric], errors="coerce")
            rows.append({
                "target_pair": pair,
                "method": method,
                "metric": metric,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "n_seeds": int(values.notna().sum()),
            })
    return pd.DataFrame(rows)


def paired(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair, pair_frame in per_seed.groupby("target_pair", sort=False):
        v5 = pair_frame[pair_frame["method"] == "V5 BalanceSync"].set_index("seed")
        for reference_method in ("V4-B", "POLYGON"):
            reference = pair_frame[pair_frame["method"] == reference_method].set_index("seed")
            common = reference.index.intersection(v5.index)
            for metric in metric_names(per_seed):
                baseline = pd.to_numeric(reference.loc[common, metric], errors="coerce").to_numpy(float)
                candidate = pd.to_numeric(v5.loc[common, metric], errors="coerce").to_numpy(float)
                valid = np.isfinite(baseline) & np.isfinite(candidate)
                delta = candidate[valid] - baseline[valid]
                rows.append({
                    "target_pair": pair,
                    "comparison": f"V5 vs {reference_method}",
                    "metric": metric,
                    "reference_mean": float(np.mean(baseline[valid])) if valid.any() else math.nan,
                    "v5_mean": float(np.mean(candidate[valid])) if valid.any() else math.nan,
                    "delta_v5_minus_reference": float(np.mean(delta)) if len(delta) else math.nan,
                    "v5_seed_wins_raw_direction": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "n": int(len(delta)),
                    "exact_sign_flip_p": exact_sign_flip_pvalue(delta),
                })
    return pd.DataFrame(rows)


def fmt(mean: float, sd: float, percent: bool = False) -> str:
    if percent:
        return f"{100 * mean:.2f}+/-{100 * sd:.2f}%"
    return f"{mean:.3f}+/-{sd:.3f}"


def summary_markdown(aggregate_frame: pd.DataFrame, paired_frame: pd.DataFrame) -> str:
    lookup = aggregate_frame.set_index(["target_pair", "method", "metric"])
    lines = [
        "# V5 BalanceSync comprehensive chemical-property audit",
        "",
        "All values are mean +/- SD across formal seeds 42-51. Molecules are unique valid canonical SMILES within each run.",
        "Quality pass is QED>=0.60, SA<=4.0, Lipinski pass, and no PAINS/Brenk alert.",
    ]
    labels = {"egfr_vegfr2": "EGFR-VEGFR2", "parp1_brd4": "PARP1-BRD4"}
    methods = ("V4-B", "V5 BalanceSync", "POLYGON")
    for pair in labels:
        lines.extend([
            "", f"## {labels[pair]}", "",
            "| method | QED | SA (lower) | MW | cLogP | TPSA | Lipinski pass | Veber pass | alert-free | quality pass | scaffold diversity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in methods:
            def value(metric: str) -> tuple[float, float]:
                row = lookup.loc[(pair, method, metric)]
                return float(row["mean"]), float(row["sd"])
            q = value("qed_mean")
            sa = value("sa_mean")
            mw = value("mol_wt_mean")
            logp = value("logp_mean")
            tpsa = value("tpsa_mean")
            lip = value("lipinski_pass_rate")
            veb = value("veber_pass_rate")
            alert = value("alert_free_rate")
            quality = value("quality_pass_rate")
            scaffold = value("scaffold_diversity")
            lines.append(
                f"| {method} | {fmt(*q)} | {fmt(*sa)} | {fmt(*mw)} | {fmt(*logp)} | {fmt(*tpsa)} | "
                f"{fmt(*lip, percent=True)} | {fmt(*veb, percent=True)} | {fmt(*alert, percent=True)} | "
                f"{fmt(*quality, percent=True)} | {fmt(*scaffold, percent=True)} |"
            )
        lines.extend([
            "", "Structural composition and alert subtypes:", "",
            "| method | heavy atoms | rings | aromatic rings | fraction Csp3 | PAINS alert | Brenk alert | charged molecules |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in methods:
            def value(metric: str) -> tuple[float, float]:
                row = lookup.loc[(pair, method, metric)]
                return float(row["mean"]), float(row["sd"])
            heavy = value("heavy_atoms_mean")
            rings = value("ring_count_mean")
            aromatic = value("aromatic_ring_count_mean")
            csp3 = value("fraction_csp3_mean")
            pains = value("pains_alert_rate")
            brenk = value("brenk_alert_rate")
            charged = value("charged_molecule_rate")
            lines.append(
                f"| {method} | {fmt(*heavy)} | {fmt(*rings)} | {fmt(*aromatic)} | {fmt(*csp3)} | "
                f"{fmt(*pains, percent=True)} | {fmt(*brenk, percent=True)} | {fmt(*charged, percent=True)} |"
            )
        lines.extend([
            "", "Quality-failure decomposition:", "",
            "| method | QED fail | SA fail | combined alert | Lipinski fail | multiple failures |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for method in methods:
            def value(metric: str) -> tuple[float, float]:
                row = lookup.loc[(pair, method, metric)]
                return float(row["mean"]), float(row["sd"])
            qed_fail = value("qed_fail_rate")
            sa_fail = value("sa_fail_rate")
            alert_fail = value("alert_rate")
            lipinski_fail = value("lipinski_fail_rate")
            multiple = value("multiple_quality_fail_rate")
            lines.append(
                f"| {method} | {fmt(*qed_fail, percent=True)} | {fmt(*sa_fail, percent=True)} | "
                f"{fmt(*alert_fail, percent=True)} | {fmt(*lipinski_fail, percent=True)} | "
                f"{fmt(*multiple, percent=True)} |"
            )
        lines.extend([
            "", "Dual@6 subset:", "",
            "| method | fraction of all molecules | QED | SA (lower) | Lipinski pass | alert-free | quality pass | scaffold diversity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in methods:
            def value(metric: str) -> tuple[float, float]:
                row = lookup.loc[(pair, method, metric)]
                return float(row["mean"]), float(row["sd"])
            fraction = value("dual6_fraction_of_unique_valid")
            q = value("dual6_qed_mean")
            sa = value("dual6_sa_mean")
            lip = value("dual6_lipinski_pass_rate")
            alert = value("dual6_alert_free_rate")
            quality = value("dual6_quality_pass_rate")
            scaffold = value("dual6_scaffold_diversity")
            lines.append(
                f"| {method} | {fmt(*fraction, percent=True)} | {fmt(*q)} | {fmt(*sa)} | "
                f"{fmt(*lip, percent=True)} | {fmt(*alert, percent=True)} | {fmt(*quality, percent=True)} | "
                f"{fmt(*scaffold, percent=True)} |"
            )
        selected = paired_frame[
            (paired_frame["target_pair"] == pair)
            & (paired_frame["comparison"] == "V5 vs V4-B")
            & paired_frame["metric"].isin([
                "qed_mean", "sa_mean", "lipinski_pass_rate", "alert_free_rate",
                "quality_pass_rate", "dual6_quality_pass_rate", "scaffold_diversity",
            ])
        ]
        lines.extend(["", "Paired V5 minus V4-B:", "", "| metric | delta | exact p |", "|---|---:|---:|"])
        for row in selected.itertuples(index=False):
            percent = str(row.metric).endswith(RATE_SUFFIXES) or row.metric == "scaffold_diversity"
            delta = float(row.delta_v5_minus_reference)
            delta_text = f"{100 * delta:+.2f} pp" if percent else f"{delta:+.4f}"
            lines.append(f"| {row.metric} | {delta_text} | {row.exact_sign_flip_p:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/v5_balancesync_two_pair_20260806.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/own_method_v5_balancesync_20260806/analysis/chemical_properties"),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Rebuild aggregate and Markdown outputs from an existing per-seed CSV.",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project / args.config
    output = args.output if args.output.is_absolute() else project / args.output
    config = json.loads(config_path.read_text(encoding="utf-8"))

    per_seed_path = output / "chemical_properties_per_seed.csv"
    if args.reuse_existing:
        if not per_seed_path.exists():
            raise FileNotFoundError(per_seed_path)
        per_seed = pd.read_csv(per_seed_path, encoding="utf-8-sig")
    else:
        per_seed = collect(project, config)
    aggregate_frame = aggregate(per_seed)
    paired_frame = paired(per_seed)
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(per_seed_path, index=False, encoding="utf-8-sig")
    aggregate_frame.to_csv(output / "chemical_properties_aggregate.csv", index=False, encoding="utf-8-sig")
    paired_frame.to_csv(output / "chemical_properties_paired.csv", index=False, encoding="utf-8-sig")
    (output / "chemical_properties_summary.md").write_text(
        summary_markdown(aggregate_frame, paired_frame), encoding="utf-8"
    )
    metadata = {
        "methods": ["V4-B", "V5 BalanceSync", "POLYGON"],
        "target_pairs": list(config["target_pairs"]),
        "seeds": list(SEEDS),
        "unit_of_replication": "formal seed",
        "quality_definition": "QED>=0.60; SA<=4.0; Lipinski pass; no PAINS/Brenk alert",
        "veber_definition": "TPSA<=140 A^2 and rotatable bonds<=10",
    }
    (output / "chemical_properties_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote chemical-property audit to {output}")


if __name__ == "__main__":
    main()
