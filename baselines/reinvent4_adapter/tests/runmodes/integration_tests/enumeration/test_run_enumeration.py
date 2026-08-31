import csv
import os
import pytest
from reinvent.runmodes.enumeration.run_enumeration import run_enumeration, PeptideEnumeration
from reinvent.runmodes.enumeration.validation import AAEnumerationConfig

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_LIBRARY_FILE = os.path.join(TEST_DIR, "mock_aa_library.csv")


@pytest.fixture
def mock_files(tmp_path):
    peptide_smi = tmp_path / "mock_peptide.smi"
    peptide_smi.write_text("N[C@@H](CS)C(=O)|?|N[C@@H](C)C(=O)|?|N[C@@H](C)C(=O)O\n")
    output_csv = tmp_path / "mock_output.csv"
    return str(peptide_smi), str(output_csv)


@pytest.fixture
def setup_test_environment(mock_files):
    peptide_smi, output_csv = mock_files
    cfg_dict = {
        "parameters": {
            "smiles_file": peptide_smi,
            "amino_acid_library_file": MOCK_LIBRARY_FILE,
            "aa_names_column": "NAME",
            "smiles_column": "SMILES",
            "output_csv": output_csv,
            "batch_size": 5,
        },
        "scoring": {
            "type": "geometric_mean",
            "component": [{"QED": {"endpoint": [{"name": "QED", "weight": 1.0}]}}],
        },
    }
    return cfg_dict


@pytest.mark.integration
def test_peptide_enumeration_initialization(setup_test_environment):
    config = AAEnumerationConfig(**setup_test_environment)
    enumerator = PeptideEnumeration(config)

    assert enumerator.configuration == config
    assert hasattr(enumerator, "sampler")
    assert hasattr(enumerator, "scoring_function")


@pytest.mark.integration
def test_peptide_enumeration_run(setup_test_environment):
    configuration = AAEnumerationConfig(**setup_test_environment)
    enumerator = PeptideEnumeration(configuration)
    enumerator.run()

    output_csv = setup_test_environment["parameters"]["output_csv"]
    assert os.path.exists(output_csv)


@pytest.mark.integration
def test_merge_columns(setup_test_environment):
    configuration = AAEnumerationConfig(**setup_test_environment)
    enumerator = PeptideEnumeration(configuration)

    peptides = ["N[C@@H](CS)C(=O)", "N[C@@H](CS)C(=O)"]
    aminoacids = ["R|H", "A|H"]
    results_header = ["Score1", "Score2"]
    results_rows = [[0.8, 0.9], [0.7, 0.6]]

    header, rows = enumerator.merge_columns(peptides, aminoacids, results_header, results_rows)

    assert header == ["SMILES", "Amino_Acids", "Score1", "Score2"]
    assert len(rows) == len(peptides)
    assert rows[0] == ["N[C@@H](CS)C(=O)", "R|H", 0.8, 0.9]


@pytest.mark.integration
def test_write_csv(setup_test_environment, tmp_path):
    configuration = AAEnumerationConfig(**setup_test_environment)
    enumerator = PeptideEnumeration(configuration)

    test_csv_file = str(tmp_path / "test_output.csv")
    header = ["Column1", "Column2"]
    rows = [["Value1", "Value2"], ["Value3", "Value4"]]

    enumerator.write_csv(test_csv_file, rows, header=header)

    with open(test_csv_file) as f:
        reader = csv.reader(f)
        file_header = next(reader)
        file_rows = list(reader)

    assert file_header == header
    assert file_rows == rows


@pytest.mark.integration
def test_run_enumeration(setup_test_environment):
    run_enumeration(setup_test_environment)

    output_csv = setup_test_environment["parameters"]["output_csv"]
    assert os.path.exists(output_csv)
