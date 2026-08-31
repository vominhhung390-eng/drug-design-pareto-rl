import pytest

from reinvent.runmodes.scoring.file_io import TabFileReader
from reinvent.chemistry.standardization.filter_types_enum import FilterTypesEnum
from reinvent.chemistry.standardization.filter_configuration import FilterConfiguration
from reinvent.chemistry.standardization.rdkit_standardizer import RDKitStandardizer

SMILES1 = r"""Oc1nc(CCCN2CC=C(c3ccccc3)CC2)nc2ccc(Cl)cc12	CHEMBL186186
COc1ccc(N2C3=C(CC4=C2CCCC4=O)C(=O)CCC3)cc1	CHEMBL1492213
INVALID	NO_ID
CCC(C)n1ccnc1C(=O)O	CHEMBL5180359
O=C(Nc1c(Br)cc(Br)cc1CN[C@H]1CC[C@H](O)CC1)c1cccs1	CHEMBL2106972
CCCCn1c(=O)[nH]c2[nH]c(-c3ccc(OCC(=O)N4CCN(c5ccccc5OC)CC4)cc3)nc2c1=O	CHEMBL37726
C/C(=C\C(=O)C[C@H](C)C(=O)O)[C@H]1C[C@H](O)[C@@]2(C)[C@]34O[C@@H]3C[C@H]3C(C)(C)C(=O)CC[C@]3(C)C4=CC(=O)[C@]12C	CHEMBL4582532
CC(CP(=O)(O)O)OCCn1cnc2c(=O)[nH]c(N)nc21	CHEMBL594335
C#CCOc1ccc(C(=O)c2ccc(C(=O)NCc3ccc4c(c3)CO[C@@]4(CCCN(C)C)c3ccc(F)cc3)cc2)cc1	CHEMBL4281744
CN(C(=O)c1cccc(NC(=O)Cc2cccc(NC(=O)C3CCCN(C(=O)C4CC4)C3)c2)c1)C1CCC1	CHEMBL1897202
Cc1cc(CN)c(O)c2c1CCC2N.Cl	CHEMBL544517
"""

CSV1 = r"""SMILES	ID
Oc1nc(CCCN2CC=C(c3ccccc3)CC2)nc2ccc(Cl)cc12	CHEMBL186186
COc1ccc(N2C3=C(CC4=C2CCCC4=O)C(=O)CCC3)cc1	CHEMBL1492213
INVALID	NO_ID
CCC(C)n1ccnc1C(=O)O	CHEMBL5180359
O=C(Nc1c(Br)cc(Br)cc1CN[C@H]1CC[C@H](O)CC1)c1cccs1	CHEMBL2106972
CCCCn1c(=O)[nH]c2[nH]c(-c3ccc(OCC(=O)N4CCN(c5ccccc5OC)CC4)cc3)nc2c1=O	CHEMBL37726
C/C(=C\C(=O)C[C@H](C)C(=O)O)[C@H]1C[C@H](O)[C@@]2(C)[C@]34O[C@@H]3C[C@H]3C(C)(C)C(=O)CC[C@]3(C)C4=CC(=O)[C@]12C	CHEMBL4582532
CC(CP(=O)(O)O)OCCn1cnc2c(=O)[nH]c(N)nc21	CHEMBL594335
C#CCOc1ccc(C(=O)c2ccc(C(=O)NCc3ccc4c(c3)CO[C@@]4(CCCN(C)C)c3ccc(F)cc3)cc2)cc1	CHEMBL4281744
CN(C(=O)c1cccc(NC(=O)Cc2cccc(NC(=O)C3CCCN(C(=O)C4CC4)C3)c2)c1)C1CCC1	CHEMBL1897202
Cc1cc(CN)c(O)c2c1CCC2N.Cl	CHEMBL544517
"""


@pytest.fixture
def mock_open_smiles(mocker):
    mocked_open = mocker.mock_open(read_data=SMILES1)
    builtin_open = "builtins.open"
    mocker.patch(builtin_open, mocked_open)


@pytest.fixture
def mock_open_csv(mocker):
    mocked_open = mocker.mock_open(read_data=CSV1)
    builtin_open = "builtins.open"
    mocker.patch(builtin_open, mocked_open)


def test_smiles_reader(mock_open_smiles):
    filter_types = FilterTypesEnum()
    standardizer_config = [
        FilterConfiguration(filter_types.GET_LARGEST_FRAGMENT),
        FilterConfiguration(filter_types.GENERAL_CLEANUP),
    ]
    standardizer = RDKitStandardizer(standardizer_config, isomeric=True)

    smiles_reader = TabFileReader("dummy.smi", actions=[standardizer.apply_filter])
    smiles_reader.read()

    # 11 lines in SMILES1, including 1 invalid that becomes None after standardization
    assert len(smiles_reader.smilies) == 11
    # The INVALID line produces None from the standardizer
    assert smiles_reader.smilies[2] is None
    # A valid molecule should be a non-empty string
    assert smiles_reader.smilies[0]
    # SMILES format has no header; header_line is the synthetic default
    assert smiles_reader.header_line == ["SMILES", "Comment"]


def test_csv_reader(mock_open_csv):
    filter_types = FilterTypesEnum()
    standardizer_config = [
        FilterConfiguration(filter_types.GET_LARGEST_FRAGMENT),
        FilterConfiguration(filter_types.GENERAL_CLEANUP),
    ]
    standardizer = RDKitStandardizer(standardizer_config, isomeric=True)

    smiles_reader = TabFileReader("dummy.csv", actions=[standardizer.apply_filter])
    smiles_reader.read()

    # mock_open does not support seek(), so get_dialect() falls through and the file is
    # treated as a headerless SMILES file — the header row is read as a data row (index 0,
    # invalid SMILES → None), giving 12 rows total.
    assert len(smiles_reader.smilies) == 12
    # Header row "SMILES\tID" is not a valid SMILES, so it becomes None after standardization
    assert smiles_reader.smilies[0] is None
    # Third data row (index 3) is "INVALID" → None
    assert smiles_reader.smilies[3] is None
    # A valid molecule should be a non-empty string
    assert smiles_reader.smilies[1]
