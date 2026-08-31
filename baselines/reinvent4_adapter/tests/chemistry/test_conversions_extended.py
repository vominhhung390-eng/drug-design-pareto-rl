"""Tests for the untested branches in reinvent/chemistry/conversions.py."""

import pytest
from rdkit import Chem

from reinvent.chemistry import conversions
from tests.chemistry.fixtures.test_data import ASPIRIN, CELECOXIB, INVALID


def test_mol_to_random_smiles_produces_valid_smiles():
    mol = Chem.MolFromSmiles(ASPIRIN)
    result = conversions.mol_to_random_smiles(mol)
    assert Chem.MolFromSmiles(result) is not None


def test_mol_to_random_smiles_encodes_same_molecule():
    mol = Chem.MolFromSmiles(ASPIRIN)
    result = conversions.mol_to_random_smiles(mol)
    # canonical SMILES of the randomised result must match the original
    canon_original = Chem.MolToSmiles(mol)
    canon_result = Chem.MolToSmiles(Chem.MolFromSmiles(result))
    assert canon_original == canon_result


def test_convert_to_rdkit_smiles_valid_single():
    """convert_to_rdkit_smiles takes a single SMILES and returns canonical SMILES."""
    result = conversions.convert_to_rdkit_smiles(ASPIRIN)
    assert result is not None
    assert Chem.MolFromSmiles(result) is not None


def test_convert_to_rdkit_smiles_canonical():
    """Two different representations of the same molecule yield the same output."""
    r1 = conversions.convert_to_rdkit_smiles("OC(=O)c1ccccc1OC(C)=O")
    r2 = conversions.convert_to_rdkit_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert r1 == r2


def test_convert_to_standardized_smiles_valid():
    result = conversions.convert_to_standardized_smiles(ASPIRIN)
    assert result is not None
    assert Chem.MolFromSmiles(result) is not None


def test_convert_to_standardized_smiles_invalid_raises():
    """Invalid SMILES raises RuntimeError (does not return None)."""
    with pytest.raises(RuntimeError):
        conversions.convert_to_standardized_smiles(INVALID)


def test_randomize_smiles_roundtrip():
    result = conversions.randomize_smiles(ASPIRIN)
    assert result is not None
    assert Chem.MolFromSmiles(result) is not None


def test_randomize_smiles_invalid_returns_none():
    result = conversions.randomize_smiles(INVALID)
    assert result is None


def test_copy_mol_produces_independent_copy():
    mol = Chem.MolFromSmiles(ASPIRIN)
    copy = conversions.copy_mol(mol)
    assert copy is not mol
    assert Chem.MolToSmiles(copy) == Chem.MolToSmiles(mol)
