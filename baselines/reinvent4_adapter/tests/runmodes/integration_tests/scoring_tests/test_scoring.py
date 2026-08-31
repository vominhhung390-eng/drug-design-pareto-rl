import csv
import os
import pytest

from reinvent.runmodes.scoring.run_scoring import run_scoring

SCORING_CONFIG = {
    "type": "custom_sum",
    "parallel": 1,
    "component": [
        {
            "custom_alerts": {
                "endpoint": [
                    {
                        "name": "Unwanted SMARTS",
                        "weight": 0.79,
                        "params": {
                            "smarts": [
                                "[*;r8]",
                                "[*;r9]",
                                "[*;r10]",
                                "[*;r11]",
                                "[*;r12]",
                                "[*;r13]",
                                "[*;r14]",
                                "[*;r15]",
                                "[*;r16]",
                                "[*;r17]",
                                "[#8][#8]",
                                "[#6;+]",
                                "[#16][#16]",
                                "[#7;!n][S;!$(S(=O)=O)]",
                                "[#7;!n][#7;!n]",
                                "C#C",
                                "C(=[O,S])[O,S]",
                                "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
                                "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
                                "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
                                "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
                                "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
                                "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
                            ]
                        },
                    }
                ]
            }
        },
        {
            "MolecularWeight": {
                "endpoint": [
                    {
                        "name": "Molecular weight",
                        "weight": 0.342,
                        "transform": {
                            "type": "double_sigmoid",
                            "high": 500.0,
                            "low": 200.0,
                            "coef_div": 500.0,
                            "coef_si": 20.0,
                            "coef_se": 20.0,
                        },
                    }
                ]
            }
        },
    ],
}


@pytest.fixture
def setup(json_config, tmp_path):
    output_csv = str(tmp_path / "score_results.csv")
    config = {
        "parameters": {
            "smiles_file": json_config["REINVENT_INCEPTION_SMI"],
            "output_csv": output_csv,
        },
        "scoring": SCORING_CONFIG,
    }
    return config, output_csv


@pytest.mark.integration
def test_run_scoring_produces_output_file(setup):
    config, output_csv = setup
    run_scoring(config)
    assert os.path.isfile(output_csv)


@pytest.mark.integration
def test_run_scoring_row_count(setup):
    config, output_csv = setup
    run_scoring(config)
    with open(output_csv) as f:
        rows = list(csv.reader(f))
    # 100 SMILES in the inception file + 1 header row
    assert len(rows) == 101


@pytest.mark.integration
def test_run_scoring_header_columns(setup):
    config, output_csv = setup
    run_scoring(config)
    with open(output_csv) as f:
        header = next(csv.reader(f))
    # Input .smi file → SMILES + Comment, then RDKit_SMILES (REINVENT), Score, components
    assert "RDKit_SMILES (REINVENT)" in header
    assert "Score" in header
    assert len(header) >= 4  # at least SMILES, Comment, Score, 1 component column


@pytest.mark.integration
def test_run_scoring_scores_in_range(setup):
    config, output_csv = setup
    run_scoring(config)
    with open(output_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        score_col = header.index("Score")
        scores = [float(row[score_col]) for row in reader if row and row[score_col]]
    assert len(scores) > 0, "No scored rows found"
    assert all(0.0 <= s <= 1.0 for s in scores), "All scores must be in [0, 1]"
    assert any(s > 0.0 for s in scores), "At least some scores should be non-zero"
