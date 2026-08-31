#!/usr/bin/env python
"""Audit agreement between activity predictors and the unified docking panel.

This analysis is diagnostic: docking is not treated as experimental ground truth.
It emits molecule-level and method-level tables with bootstrap confidence intervals
and permutation p-values so the evidence can be reported without cherry-picking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


RNG_SEED = 20260730


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spearman_summary(x: pd.Series, y: pd.Series, *, n_resamples: int = 10000) -> dict:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    xv = pair["x"].to_numpy(float)
    yv = pair["y"].to_numpy(float)
    n = len(pair)
    rho = float(spearmanr(xv, yv).statistic) if n >= 3 else np.nan
    rng = np.random.default_rng(RNG_SEED)
    boot = []
    if n >= 4:
        for _ in range(n_resamples):
            idx = rng.integers(0, n, n)
            value = spearmanr(xv[idx], yv[idx]).statistic
            if np.isfinite(value):
                boot.append(float(value))
    ci_low, ci_high = (
        np.quantile(boot, [0.025, 0.975]) if boot else (np.nan, np.nan)
    )
    perm = []
    if n >= 3 and np.isfinite(rho):
        for _ in range(n_resamples):
            perm.append(abs(float(spearmanr(xv, rng.permutation(yv)).statistic)))
        p_value = (1 + sum(value >= abs(rho) for value in perm)) / (1 + len(perm))
    else:
        p_value = np.nan
    return {
        "n": n,
        "spearman_rho": rho,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "permutation_p_two_sided": float(p_value),
        "bootstrap_resamples": n_resamples,
        "permutation_resamples": n_resamples,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=root
        / "results"
        / "predictor_evidence_minimal_20260730"
        / "selected_compounds_candidate_scores.csv",
    )
    parser.add_argument(
        "--docking",
        type=Path,
        default=root / "docking" / "unified_7method_top5" / "docking_compound_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "predictor_evidence_minimal_20260730",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores)
    docking = pd.read_csv(args.docking)
    dock_cols = [
        "compound_id",
        "egfr_vina_kcal_mol",
        "vegfr2_vina_kcal_mol",
        "dual_worst_vina_kcal_mol",
        "dual_mean_vina_kcal_mol",
        "dual_pass",
        "plausible_both_targets",
    ]
    merged = scores.merge(docking[dock_cols], on="compound_id", validate="one_to_one")
    if len(merged) != len(scores):
        raise RuntimeError(f"Docking merge lost rows: {len(scores)} -> {len(merged)}")

    # Convert Vina energies to a direction where larger means stronger predicted binding.
    merged["egfr_vina_strength"] = -merged["egfr_vina_kcal_mol"]
    merged["vegfr2_vina_strength"] = -merged["vegfr2_vina_kcal_mol"]
    merged["dual_vina_strength"] = -merged["dual_worst_vina_kcal_mol"]
    merged["candidate_min_pactivity"] = merged[
        ["candidate_egfr_pactivity", "candidate_vegfr2_pactivity"]
    ].min(axis=1)
    merged["candidate_mean_pactivity"] = merged[
        ["candidate_egfr_pactivity", "candidate_vegfr2_pactivity"]
    ].mean(axis=1)
    merged.to_csv(
        args.output_dir / "predictor_docking_molecule_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparisons = [
        ("EGFR", "locked_RF", "egfr", "egfr_vina_strength"),
        ("EGFR", "round2_candidate", "candidate_egfr_pactivity", "egfr_vina_strength"),
        ("VEGFR2", "locked_RF", "vegfr2", "vegfr2_vina_strength"),
        (
            "VEGFR2",
            "round2_candidate",
            "candidate_vegfr2_pactivity",
            "vegfr2_vina_strength",
        ),
        ("dual_worst", "locked_RF", "min_activity", "dual_vina_strength"),
        (
            "dual_worst",
            "round2_candidate",
            "candidate_min_pactivity",
            "dual_vina_strength",
        ),
    ]
    rows = []
    for endpoint, model, predictor, docking_strength in comparisons:
        subsets = [("all", merged)]
        subsets.append(
            (
                "joint_AD",
                merged[merged["within_joint_applicability_domain"].astype(bool)],
            )
        )
        for subset_name, subset in subsets:
            result = spearman_summary(subset[predictor], subset[docking_strength])
            rows.append(
                {
                    "endpoint": endpoint,
                    "model": model,
                    "subset": subset_name,
                    "predictor_column": predictor,
                    "docking_column": docking_strength,
                    **result,
                }
            )
    correlations = pd.DataFrame(rows)
    correlations.to_csv(
        args.output_dir / "predictor_docking_correlation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    method = (
        merged.groupby("method", as_index=False)
        .agg(
            n_compounds=("compound_id", "size"),
            locked_rf_dual_median=("min_activity", "median"),
            candidate_dual_median=("candidate_min_pactivity", "median"),
            vina_dual_strength_median=("dual_vina_strength", "median"),
            joint_ad_rate=("within_joint_applicability_domain", "mean"),
            dual_docking_pass_rate=("dual_pass", "mean"),
            plausible_both_rate=("plausible_both_targets", "mean"),
        )
        .sort_values("vina_dual_strength_median", ascending=False)
    )
    method.to_csv(
        args.output_dir / "predictor_docking_method_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall = {
        "analysis_role": "diagnostic cross-check; docking is not experimental ground truth",
        "n_molecules": int(len(merged)),
        "n_methods": int(merged["method"].nunique()),
        "joint_ad_n": int(merged["within_joint_applicability_domain"].sum()),
        "joint_ad_rate": float(merged["within_joint_applicability_domain"].mean()),
        "dual_docking_pass_n": int(merged["dual_pass"].sum()),
        "dual_docking_pass_rate": float(merged["dual_pass"].mean()),
        "plausible_both_n": int(merged["plausible_both_targets"].sum()),
        "plausible_both_rate": float(merged["plausible_both_targets"].mean()),
        "selection_caveat": (
            "The 35 molecules were selected using locked-RF activity and quality criteria; "
            "therefore correlations involving locked RF are descriptive and selection-biased."
        ),
    }
    with (args.output_dir / "predictor_docking_audit.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    validation_rows = [
        {
            "target": "EGFR",
            "model": "locked official RF",
            "validation": "post-publication novel",
            "n": 425,
            "r2": 0.094375,
            "rmse": 1.240959,
            "spearman_rho": 0.347284,
            "status": "limited",
        },
        {
            "target": "EGFR",
            "model": "locked official RF",
            "validation": "strict recent novel",
            "n": 115,
            "r2": -0.528040,
            "rmse": 1.498235,
            "spearman_rho": -0.049183,
            "status": "fail",
        },
        {
            "target": "EGFR",
            "model": "round-2 multitask D-MPNN",
            "validation": "two rolling historical folds",
            "n": np.nan,
            "r2": np.nan,
            "rmse": 1.119,
            "spearman_rho": 0.297,
            "status": "fail formal-oracle threshold",
        },
        {
            "target": "VEGFR2",
            "model": "locked official RF",
            "validation": "post-publication novel",
            "n": 285,
            "r2": 0.176221,
            "rmse": 0.954835,
            "spearman_rho": 0.391429,
            "status": "moderate",
        },
        {
            "target": "VEGFR2",
            "model": "locked official RF",
            "validation": "strict recent novel",
            "n": 61,
            "r2": 0.411947,
            "rmse": 0.945099,
            "spearman_rho": 0.656182,
            "status": "pass with small external n",
        },
        {
            "target": "VEGFR2",
            "model": "round-2 recent-weighted ExtraTrees",
            "validation": "two rolling historical folds",
            "n": np.nan,
            "r2": np.nan,
            "rmse": 0.951105,
            "spearman_rho": 0.619075,
            "status": "moderate",
        },
    ]
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(
        args.output_dir / "table_s_predictor_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tracked_files = [
        args.scores,
        args.docking,
        root / "models" / "oracles" / "target_EGFR_model.pkl",
        root / "models" / "oracles" / "target_VEGFR2_model.pkl",
        root
        / "results"
        / "predictor_validation_20260729"
        / "locked_model_regression_metrics.csv",
        root
        / "results"
        / "predictor_retraining_round2_20260730"
        / "classical_benchmark"
        / "model_ranking.csv",
    ]
    manifest = {
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "random_seed": RNG_SEED,
        "files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in tracked_files
        ],
    }
    with (args.output_dir / "evidence_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    corr_all = correlations[correlations["subset"] == "all"].copy()
    corr_lines = []
    for row in corr_all.itertuples(index=False):
        corr_lines.append(
            f"| {row.endpoint} | {row.model} | {row.n} | {row.spearman_rho:.3f} "
            f"| [{row.bootstrap_ci_low:.3f}, {row.bootstrap_ci_high:.3f}] "
            f"| {row.permutation_p_two_sided:.4f} |"
        )
    report = f"""# Predictor evidence package (2026-07-30)

## Decision

The locked RF models remain the common scoring functions for the formal generation comparison. This preserves a single, pre-existing endpoint across methods, but the scores must be described as *predicted pActivity*, not measured IC50. The round-2 models are retained as diagnostic models only and must not replace the locked RF models in the primary comparison.

The key reason is domain shift: none of the 35 uniformly docked generated molecules met the strict joint applicability-domain rule (maximum ECFP4 Tanimoto similarity >= 0.60 to both target development sets). Thus, neither RF nor round-2 predictions constitute evidence of experimentally confirmed dual-target activity for these generated molecules.

## Table S1. Predictor validation

| Target | Model | Validation design | n | R2 | RMSE (pActivity) | Spearman rho | Assessment |
|---|---|---|---:|---:|---:|---:|---|
| EGFR | locked official RF | post-publication novel | 425 | 0.094 | 1.241 | 0.347 | limited |
| EGFR | locked official RF | strict recent novel | 115 | -0.528 | 1.498 | -0.049 | failed temporal extrapolation |
| EGFR | round-2 multitask D-MPNN | two rolling historical folds | - | - | 1.119 | 0.297 | not qualified as formal oracle |
| VEGFR2 | locked official RF | post-publication novel | 285 | 0.176 | 0.955 | 0.391 | moderate |
| VEGFR2 | locked official RF | strict recent novel | 61 | 0.412 | 0.945 | 0.656 | encouraging but small external n |
| VEGFR2 | round-2 recent-weighted ExtraTrees | two rolling historical folds | - | - | 0.951 | 0.619 | moderate |

## Table S2. Predictor-docking consistency on the common 35-molecule panel

| Endpoint | Predictor | n | Spearman rho | Bootstrap 95% CI | Permutation p |
|---|---|---:|---:|---:|---:|
{chr(10).join(corr_lines)}

The only reproducible positive association was observed for VEGFR2: locked RF rho = 0.409 (p = 0.0156) and the round-2 candidate rho = 0.382 (p = 0.0247). EGFR and the dual-worst endpoint were not significant. These analyses are supportive cross-checks only: docking energy is not an experimental affinity measurement, and the locked-RF correlations are selection-biased because the 35 compounds were selected partly using locked-RF scores.

## Ready-to-use Methods text (English)

Target activity was evaluated using fixed, target-specific random-forest scoring functions shared across all generative methods. To assess temporal robustness, the predictors were evaluated on post-publication and strict recent-novel cohorts after exact-structure and scaffold-aware filtering. Additional round-2 models were developed using assay-consistent IC50 records and rolling historical validation; these models were used only for diagnostic sensitivity analyses and did not alter the formal reward or ranking. Applicability was defined using the maximum ECFP4 Tanimoto similarity to each target's development set, with a conservative threshold of 0.60. Candidate molecules were independently docked to EGFR and VEGFR2 using the same receptor preparation, search space, and Vina protocol. Predictor-docking agreement was quantified using Spearman correlation, 10,000 bootstrap resamples for 95% confidence intervals, and 10,000 two-sided label permutations. Docking was treated as an orthogonal structural-plausibility check rather than experimental evidence of binding affinity.

## Ready-to-use Limitations text (English)

The activity predictors showed target-dependent temporal generalization. In particular, EGFR performance deteriorated on the strict recent-novel cohort, and none of the 35 uniformly docked generated molecules fell within the joint applicability domain of both target models. Consequently, predicted pActivity values were used as consistent computational objectives for relative comparison and were not interpreted as experimentally measured potency. The observed VEGFR2 predictor-docking association provides limited orthogonal support, whereas the absence of significant EGFR and dual-endpoint associations constrains claims of dual-target activity. Prospective biochemical assays remain necessary to establish true potency.

## 可直接使用的中文结论

正式比较继续使用所有方法共享且预先锁定的 RF 评分器，以保证评价口径一致；其输出只能表述为“预测 pActivity”，不能写成实验 IC50。第二轮 EGFR/VEGFR2 模型仅用于敏感性分析，不替换正式评分器。35 个统一 docking 分子全部位于严格联合适用域之外，因此当前证据支持“计算评分较优的候选分子”和“部分 VEGFR2 结构一致性”，不支持“已获得经实验验证的双靶点抑制剂”。

## Permitted and prohibited claims

- Permitted: relative predicted-activity comparison under one locked scoring protocol; uniform docking comparison; moderate VEGFR2 predictor-docking concordance; explicit out-of-domain disclosure.
- Prohibited: experimentally validated nanomolar potency; confirmed dual-target inhibition; replacement of biochemical assays by docking; presenting round-2 models as externally qualified formal oracles.

## Reproduction

```powershell
python -X utf8 evaluation/analyze_predictor_docking_consistency.py
```

The exact input hashes are recorded in `evidence_manifest.json`.
"""
    (args.output_dir / "paper_ready_predictor_evidence_zh_en.md").write_text(
        report, encoding="utf-8"
    )

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\nMolecule-level correlations:")
    print(correlations.to_string(index=False))
    print("\nMethod-level summary:")
    print(method.to_string(index=False))


if __name__ == "__main__":
    main()
