"""Filter and preprocess SMILES with RDKit"""

from __future__ import annotations

__all__ = ["RDKitFilter"]
import logging
from typing import Optional, TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize as Standardizer
from ..logger import setup_mp_logger

if TYPE_CHECKING:
    from ..validation import FilterSection

logger = logging.getLogger(__name__)


class RDKitFilter:
    """Filter SMILES using RDKit"""

    def __init__(self, config: FilterSection, transforms, level="INFO", queue=None):
        self.config = config
        self.transforms = transforms
        self.level = level
        self.queue = queue

        self.instantiated = False
        self.fragment_chooser = None
        self.normalizer = None
        self.uncharger = None
        self.disconnector = None
        self.mol = None

    def __call__(self, smiles: str) -> Optional[tuple]:
        """Filter and preprocess SMILES

        :param smiles: input SMILES
        :returns: processed SMILES
        """

        if not smiles:
            return None

        # FIXME: done here because the instances won't pickle (multiprocessing)
        if not self.instantiated:
            if self.queue:
                setup_mp_logger(logger, self.level, self.queue)

            self.fragment_chooser = Standardizer.LargestFragmentChooser(preferOrganic=True)
            self.uncharger = Standardizer.Uncharger()
            self.disconnector = Standardizer.MetalDisconnector()

            cleanup_params = Standardizer.CleanupParameters()
            self.normalizer = Standardizer.NormalizerFromData(self.transforms, cleanup_params)

            self.tautomer_enumerator = Standardizer.TautomerEnumerator()

            self.instantiated = True

        try:
            new_smiles = self.clean_smiles(smiles)
        except (Chem.KekulizeException, Chem.AtomKekulizeException, Chem.AtomValenceException) as e:
            logger.error(f'"{smiles}": {e}')
            new_smiles = None

        return new_smiles, self.mol

    def clean_smiles(self, smiles: str) -> str | None:
        """Clean SMILES using RDKit's sanitation and normalization functions.

        :param smiles: input SMILES
        :returns: cleaned SMILES or None if SMILES is skipped or invalid
        """

        self.mol = Chem.MolFromSmiles(smiles, sanitize=False)  # alternative sanitation flags

        if not self.mol:
            return None

        if self.config.report_errors:
            problems = Chem.DetectChemistryProblems(self.mol)

            for problem in problems:
                logger.error(f'"{smiles}": {problem.Message()} ({problem.GetType()})')

            if problems:  # will fail further down so bail out here
                return None

        self.mol.UpdatePropertyCache(strict=False)  # should fix implicit valence and ring information...
        self.fragment_chooser.chooseInPlace(self.mol)

        # runs MolOps::sanitizeMol but without
        # MolOps::SANITIZE_CLEANUP (N, P, halogen: high-valent to charge)
        # MolOps::SANITIZE_PROPERTIES (runs updatePropertyCache)
        self.normalizer.normalizeInPlace(self.mol)

        # ring information seems available only after sanitation
        # Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_SETAROMATICITY)
        atom_rings = self.mol.GetRingInfo().AtomRings()

        if len(atom_rings) > self.config.max_num_rings:
            return None

        for ring_idx in atom_rings:  # ring sizes
            if len(ring_idx) > self.config.max_ring_size:
                return None

        if self.config.uncharge:
            self.uncharger.unchargeInPlace(self.mol)

        self.mol = Chem.RemoveHs(self.mol)

        if self.config.uncharge:
            Standardizer.ReionizeInPlace(self.mol)

        # NOTE: this can be very slow, easily by a factor of 10 or more
        if self.config.canonical_tautomer:
            self.mol = self.tautomer_enumerator.Canonicalize(self.mol)

        try:
            new_smiles = Chem.MolToSmiles(
                self.mol,
                canonical=True,
                isomericSmiles=self.config.keep_stereo,
                kekuleSmiles=self.config.kekulize,
                doRandom=self.config.randomize_smiles,
            )
        except RuntimeError as e:
            if "Invariant Violation" in e.args[0]:
                return None
            else:
                raise

        # final check
        mol = Chem.MolFromSmiles(new_smiles)

        if not mol:
            return None

        return new_smiles
