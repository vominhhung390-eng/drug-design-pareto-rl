import pytest
from types import SimpleNamespace

from reinvent.datapipeline.filters import RegexFilter
from reinvent.datapipeline.filters import elements as elements_module

SMILES = [
    "CC(=O)O[Cl+][O-]",
    "CC(C)(C)[PH+]([BH2-][P+]([BH2-][ClH+2])(C(C)(C)C)C(C)(C)C)C(C)(C)C",
    "[O-][Br+3]([O-])(O)Oc1cccnc1Br",
    "CCC=CCI1C(=O)CCC1([IH])CC(=O)O",
    "CC1=Cc2ccccc2S1(O[I+3]([O-])([O-])O)C(F)(F)F",
    "C[N+]1(C2C[I-]C2)CC1",
]

# Mirror the production behaviour in reinvent/datapipeline/preprocess.py line 90:
# config.filter.elements = list(BASE_ELEMENTS | set(config.filter.elements))
_EXTRA_ELEMENTS = ["B", "P"]
_FULL_ELEMENTS = list(elements_module.BASE_ELEMENTS | set(_EXTRA_ELEMENTS))


@pytest.fixture
def regex_filter():
    config = SimpleNamespace(
        keep_stereo=True,
        keep_isotopes=False,
        keep_isotope_molecules=False,
        max_heavy_atoms=70,
        max_mol_weight=1200,
        min_heavy_atoms=2,
        min_carbons=2,
        elements=_FULL_ELEMENTS,
    )
    yield RegexFilter(config)


def test_unwanted_tokens(regex_filter):
    results = [regex_filter(smiles) for smiles in SMILES]

    assert len(results) == len(SMILES)
    # All test SMILES contain charged/heavy halogen tokens ([Cl+], [Br+3], [IH] etc.)
    # which match the UNWANTED_TOKENS regex, so every entry is rejected
    assert not any(results)


def test_unwanted_tokens_which_pass(regex_filter):
    """Elements B and P are in the excluded list; verify per-SMILES outcomes."""
    results = {smiles: regex_filter(smiles) for smiles in SMILES}

    # Both P-containing and halogen-charged SMILES are rejected by UNWANTED_TOKENS
    assert results["CC(C)(C)[PH+]([BH2-][P+]([BH2-][ClH+2])(C(C)(C)C)C(C)(C)C)C(C)(C)C"] is None
    assert results["CC(=O)O[Cl+][O-]"] is None  # [Cl+] matches UNWANTED_TOKENS


def test_clean_smiles_passes_filter(regex_filter):
    """A plain organic SMILES with no unusual tokens should pass through."""
    result = regex_filter("CC(=O)Oc1ccccc1C(=O)O")  # aspirin, no charged/halogen brackets
    assert result is not None
