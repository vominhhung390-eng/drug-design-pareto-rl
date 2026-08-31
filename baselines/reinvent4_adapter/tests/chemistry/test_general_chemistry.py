import unittest

from rdkit import Chem
from reinvent.chemistry import conversions
from tests.chemistry.fixtures.test_data import (
    INVALID,
    ASPIRIN2,
    METHYL_3_O_TOLYL_PROPYL_AMINE,
    METHYL_3_O_TOLYL_PROPYL_AMINE2,
    CELECOXIB2,
    ASPIRIN_INCHI_KEY,
    CELECOXIB_INCHI_KEY,
)


class Test_general_chemistry(unittest.TestCase):
    def setUp(self):
        self.smiles = [ASPIRIN2, CELECOXIB2, INVALID]
        self.stereo_smiles = METHYL_3_O_TOLYL_PROPYL_AMINE2
        self.non_stereo_smiles = METHYL_3_O_TOLYL_PROPYL_AMINE
        self.mols = [Chem.MolFromSmiles(smile) for smile in [CELECOXIB2, ASPIRIN2]]

    def test_smiles_to_mols_and_indices(self):
        mols, indices = conversions.smiles_to_mols_and_indices(self.smiles)

        self.assertEqual(len(mols), 2)
        self.assertEqual(len(indices), 2)
        # INVALID is filtered; ASPIRIN2 is index 0, CELECOXIB2 is index 1
        self.assertEqual(indices, [0, 1])
        self.assertIsNotNone(mols[0])
        self.assertIsNotNone(mols[1])

    def test_mols_to_fingerprints(self):
        fps = conversions.mols_to_fingerprints(self.mols)

        self.assertEqual(len(fps), 2)
        # UIntSparseIntVect: non-zero elements means the fingerprint captured features
        self.assertGreater(len(fps[0].GetNonzeroElements()), 0)
        self.assertGreater(len(fps[1].GetNonzeroElements()), 0)
        # Celecoxib and aspirin have different fingerprints
        self.assertNotEqual(fps[0], fps[1])

    def test_smiles_to_mols(self):
        mols = conversions.smiles_to_mols(self.smiles)

        self.assertEqual(len(mols), 2)
        # Both returned mols should be valid (non-None)
        self.assertTrue(all(mol is not None for mol in mols))

    def test_smiles_to_fingerprints(self):
        fps = conversions.smiles_to_fingerprints(self.smiles)

        self.assertEqual(len(fps), 2)
        # INVALID was dropped; fingerprints are distinct
        self.assertNotEqual(fps[0], fps[1])

    def test_smile_to_mol_not_none(self):
        mol = conversions.smile_to_mol(ASPIRIN2)

        self.assertIsNotNone(mol)

    def test_smile_to_mol_none(self):
        mol = conversions.smile_to_mol(INVALID)

        self.assertIsNone(mol)

    def test_mols_to_smiles(self):
        mols = conversions.smiles_to_mols(self.smiles)
        smiles = conversions.mols_to_smiles(mols)

        # mols_to_smiles with isomericSmiles=False re-canonicalises; check identity for these inputs
        self.assertEqual(len(smiles), 2)
        self.assertEqual(smiles[0], ASPIRIN2)
        self.assertEqual(smiles[1], CELECOXIB2)

    def test_mols_to_smiles_stereo(self):
        mols = conversions.smile_to_mol(self.stereo_smiles)
        smiles = conversions.mol_to_smiles(mols)

        self.assertEqual(self.non_stereo_smiles, smiles)
