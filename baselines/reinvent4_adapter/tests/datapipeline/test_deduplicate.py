import pytest
from rdkit import Chem

from reinvent.datapipeline.filters.deduplicate import inchi_key_deduplicator


def _mols(smiles_list):
    return [Chem.MolFromSmiles(s) if s else None for s in smiles_list]


def test_deduplicator_removes_tautomers():
    """Two SMILES that encode the same molecule collapse to one entry."""
    smilies = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    mols = _mols(smilies)
    result = inchi_key_deduplicator(mols, smilies)
    assert len(result) == 1


def test_deduplicator_keeps_distinct_molecules():
    smilies = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
    mols = _mols(smilies)
    result = inchi_key_deduplicator(mols, smilies)
    assert len(result) == 2


def test_deduplicator_skips_none_smiles():
    smilies = [None, "c1ccccc1"]
    mols = [None, Chem.MolFromSmiles("c1ccccc1")]
    result = inchi_key_deduplicator(mols, smilies)
    assert len(result) == 1
    assert result[0] == "c1ccccc1"


def test_deduplicator_keeps_last_occurrence():
    """When the same InChIKey appears twice the last SMILES string is kept."""
    smilies = ["CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1OC(C)=O"]
    mols = _mols(smilies)
    result = inchi_key_deduplicator(mols, smilies)
    assert len(result) == 1
    assert result[0] == smilies[-1]


def test_deduplicator_empty_input():
    result = inchi_key_deduplicator([], [])
    assert result == []
