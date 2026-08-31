import pytest

from reinvent_plugins.normalizers.rdkit_smiles import normalize


def test_normalize_strips_isotope_labels():
    result = normalize(["[13CH4]"])
    assert result == ["C"]


def test_normalize_strips_atom_map_numbers():
    result = normalize(["[CH3:1][OH:2]"])
    assert result == ["CO"]


def test_normalize_valid_plain_smiles_passthrough():
    result = normalize(["CC(=O)Oc1ccccc1C(=O)O"])
    assert len(result) == 1
    assert result[0] is not None


def test_normalize_invalid_smiles_excluded_by_default():
    result = normalize(["INVALID_SMILES"])
    assert result == []


def test_normalize_invalid_smiles_kept_when_keep_all():
    result = normalize(["INVALID_SMILES"], keep_all=True)
    assert result == ["INVALID_SMILES"]


def test_normalize_multiple_smiles_mixed():
    result = normalize(["[13CH4]", "INVALID", "c1ccccc1"], keep_all=False)
    assert len(result) == 2  # isotope-cleaned methane + benzene; INVALID dropped
    assert "C" in result


def test_normalize_empty_input():
    result = normalize([])
    assert result == []


def test_normalize_strips_both_isotope_and_atom_map():
    """After removing isotope labels and atom map numbers the molecule is valid."""
    result = normalize(["[13C:1]([2H])([2H])[2H]"])
    assert len(result) == 1
    from rdkit import Chem

    assert Chem.MolFromSmiles(result[0]) is not None
