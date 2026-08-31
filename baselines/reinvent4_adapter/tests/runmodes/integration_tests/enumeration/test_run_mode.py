import os
import csv
import unittest
from types import SimpleNamespace
from pathlib import Path

import pytest

from reinvent.Reinvent import main as reinvent_main

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PEPTIDE_FILE = os.path.join(TEST_DIR, "peptide_single_mask.smi")
MULTI_PEPTIDE_FILE = os.path.join(TEST_DIR, "peptide_multiple_masks.smi")
AA_LIB_FILE = os.path.join(TEST_DIR, "mock_aa_library.csv")
TOML_FILE = os.path.join(TEST_DIR, "integration_enumeration.toml")
OUTPUT_FILE = os.path.join(TEST_DIR, "integration_output.csv")


@pytest.mark.integration
class TestEnumerationRunModeIntegration(unittest.TestCase):
    def setUp(self):
        # create TOML config referencing existing mock files
        toml_content = f"""
run_type = "enumeration"
json_out_config = "{(Path(TEST_DIR) / 'integration_enumeration.json').as_posix()}"

[parameters]
batch_size = 10
smiles_file = "{Path(PEPTIDE_FILE).as_posix()}"
amino_acid_library_file = "{Path(AA_LIB_FILE).as_posix()}"
aa_names_column = "NAME"
smiles_column = "SMILES"
output_csv = "{Path(OUTPUT_FILE).as_posix()}"

[scoring]
type = "geometric_mean"

[[scoring.component]]
[scoring.component.QED]
[[scoring.component.QED.endpoint]]
name = "QED"
weight = 1.0
"""
        with open(TOML_FILE, "w") as f:
            f.write(toml_content)

    def tearDown(self):
        for fp in [TOML_FILE, OUTPUT_FILE, os.path.join(TEST_DIR, "integration_enumeration.json")]:
            if os.path.exists(fp):
                os.remove(fp)

    def test_enumeration_run_mode_end_to_end(self):
        self.assertTrue(os.path.exists(PEPTIDE_FILE), "Peptide .smi file missing")
        self.assertTrue(os.path.exists(AA_LIB_FILE), "AA library .csv file missing")

        args = SimpleNamespace(
            log_level="INFO",
            log_filename=None,
            dotenv_filename=None,
            config_filename=Path(TOML_FILE),
            config_format="toml",
            enable_rdkit_log_levels=None,
            device="cpu",
            seed=42,
        )

        reinvent_main(args)

        self.assertTrue(os.path.exists(OUTPUT_FILE), "Output CSV not created")

        with open(OUTPUT_FILE, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        # library has 4 unique amino acids (duplicate Q collapsed) and 2 masks => 4^2 = 16 combinations
        self.assertIn("SMILES", header)
        self.assertIn("Amino_Acids", header)
        self.assertIn("Score", header)
        self.assertEqual(len(rows), 4, f"Expected 4 rows, got {len(rows)}")
        for r in rows:
            self.assertNotIn("?", r[0], f"Masked token still present in enumerated SMILES: {r[0]}")

    def test_enumeration_run_mode_multiple_masks(self):
        self.assertTrue(os.path.exists(MULTI_PEPTIDE_FILE), "Multi-mask peptide .smi file missing")
        self.assertTrue(os.path.exists(AA_LIB_FILE), "AA library .csv file missing")
        multi_cfg_file = os.path.join(TEST_DIR, "integration_enumeration_multi.toml")
        multi_out_file = os.path.join(TEST_DIR, "integration_output_multi.csv")
        multi_json_file = os.path.join(TEST_DIR, "integration_enumeration_multi.json")
        toml_content = f"""
run_type = "enumeration"
json_out_config = "{Path(multi_json_file).as_posix()}"

[parameters]
batch_size = 20
smiles_file = "{Path(MULTI_PEPTIDE_FILE).as_posix()}"
amino_acid_library_file = "{Path(AA_LIB_FILE).as_posix()}"
aa_names_column = "NAME"
smiles_column = "SMILES"
output_csv = "{Path(multi_out_file).as_posix()}"

[scoring]
type = "geometric_mean"

[[scoring.component]]
[scoring.component.QED]
[[scoring.component.QED.endpoint]]
name = "QED"
weight = 1.0
"""
        with open(multi_cfg_file, "w") as f:
            f.write(toml_content)
        args = SimpleNamespace(
            log_level="INFO",
            log_filename=None,
            dotenv_filename=None,
            config_filename=Path(multi_cfg_file),
            config_format="toml",
            enable_rdkit_log_levels=None,
            device="cpu",
            seed=7,
        )
        reinvent_main(args)
        self.assertTrue(os.path.exists(multi_out_file), "Multi-mask output CSV not created")
        with open(multi_out_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        self.assertIn("SMILES", header)
        self.assertIn("Amino_Acids", header)
        self.assertIn("Score", header)
        self.assertEqual(len(rows), 16, f"Expected 16 rows for 2 masks * 4 AAs, got {len(rows)}")
        for r in rows:
            self.assertNotIn("?", r[0], f"Masked token still present in enumerated SMILES: {r[0]}")
        # cleanup
        for fp in [multi_cfg_file, multi_out_file, multi_json_file]:
            if os.path.exists(fp):
                os.remove(fp)


if __name__ == "__main__":
    unittest.main()
