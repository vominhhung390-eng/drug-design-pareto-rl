"""Tests for the untested branches in datapipeline/filters/regex.py.

The existing test_unwanted_tokens.py covers the reject-all path (charged/halogen
tokens) and verifies that plain aspirin passes.  These tests target the
isotope/stereo branches and multi-molecule behaviour.
"""

import pytest
from types import SimpleNamespace

from reinvent.datapipeline.filters import RegexFilter
from reinvent.datapipeline.filters import elements as elements_module

_FULL_ELEMENTS = list(elements_module.BASE_ELEMENTS | {"B", "P"})


def _make_filter(**overrides):
    defaults = dict(
        keep_stereo=True,
        keep_isotopes=False,
        keep_isotope_molecules=False,
        max_heavy_atoms=70,
        max_mol_weight=1200,
        min_heavy_atoms=2,
        min_carbons=2,
        elements=_FULL_ELEMENTS,
    )
    defaults.update(overrides)
    return RegexFilter(SimpleNamespace(**defaults))


def test_isotope_molecule_rejected_when_keep_false():
    f = _make_filter(keep_isotopes=False, keep_isotope_molecules=False)
    result = f("C[13CH3]")
    assert result is None


def test_isotope_molecule_kept_when_keep_true():
    f = _make_filter(keep_isotopes=True, keep_isotope_molecules=True)
    result = f("C[13CH3]")
    assert result is not None


def test_stereo_stripped_when_keep_stereo_false():
    f = _make_filter(keep_stereo=False)
    result = f("C[C@@H](N)C(=O)O")  # L-alanine with stereo
    assert result is not None
    assert "@" not in result


def test_stereo_preserved_when_keep_stereo_true():
    f = _make_filter(keep_stereo=True)
    result = f("C[C@@H](N)C(=O)O")
    assert result is not None
    assert "@" in result


def test_max_heavy_atoms_rejects_large_molecule():
    f = _make_filter(max_heavy_atoms=3)
    result = f("c1ccccc1")  # 6 heavy atoms — over limit
    assert result is None


def test_min_heavy_atoms_rejects_tiny_molecule():
    f = _make_filter(min_heavy_atoms=10)
    result = f("CC")  # 2 heavy atoms — under limit
    assert result is None


def test_min_carbons_rejects_no_carbon_molecule():
    f = _make_filter(min_carbons=2)
    result = f("NN")  # no carbons
    assert result is None


def test_element_not_in_allowed_list_rejected():
    # Create filter without Bromine in the allowed list
    elements_no_br = list((elements_module.BASE_ELEMENTS - {"Br"}) | {"P"})
    f = _make_filter(elements=elements_no_br)
    result = f("c1ccc(Br)cc1")  # bromobenzene
    assert result is None


def test_valid_plain_smiles_passes():
    f = _make_filter()
    result = f("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert result is not None
