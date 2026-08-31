import numpy as np

from reinvent_plugins.components.RDKit.comp_mol_volume import Parameters, MolVolume


def test_comp_mol_volume():
    params = Parameters([0.2, 2.0])
    mol_volume = MolVolume(params)

    smiles = ["c1ccccc1N", "SCc1ccncc1O"]
    results = mol_volume(smiles)

    # EmbedMolecule now uses randomSeed=0xf00d, so output is deterministic
    # regardless of which tests ran before this one.
    expected = np.array([95.144, 123.712])
    assert np.allclose(np.concatenate(results.scores), expected, atol=1e-3)
