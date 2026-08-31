# call sampler
# call scoring

# TODO: what is missing? Amino acid correct chemistry check (from pepinvent fragments), log out amino acid names, correct input, run mode exposure
from __future__ import annotations

import csv
import logging
import os

from reinvent.models.model_factory import sample_batch

__all__ = ["run_enumeration"]

from typing import List, Optional, Tuple

import numpy as np

from reinvent.chemistry.amino_acids.amino_acids import construct_amino_acids_fragments
#from reinvent.runmodes.enumeration.aa_enumeration_config import AAEnumerationConfig
from reinvent.runmodes.enumeration.aa_enumerator import AAEnumerator
from reinvent.runmodes.enumeration.validation import AAEnumerationConfig
from reinvent.runmodes.scoring.run_scoring import get_result_table, uniquify_header
from reinvent.scoring import Scorer


logger = logging.getLogger(__name__)

def run_enumeration(input_config: dict, write_config: str = None, *args, **kwargs):
    """Enumeration run setup"""
    logger.info("Starting Enumeration")

    config = AAEnumerationConfig(**input_config)

    enumerator = PeptideEnumeration(config)
    enumerator.run()
    logger.info("Enumeration completed")

class PeptideEnumeration:
    def __init__(self, configuration: AAEnumerationConfig):

        self.configuration = configuration
        self.sampler = AAEnumerator(configuration)
        self.library = self.sampler.library

        self.scoring_function = Scorer(configuration.scoring)
        logger.info("Enumeration initialized")

    def run(self) -> None:
        # Sample peptides until the iterator is exhausted
        peptide_set_length = self.configuration.parameters.batch_size
        if_repeat = False
        
        # Sample until exhausted
        num_mask = self.sampler.num_mask
        num_combinations = len(self.library) ** num_mask
        logger.info(f"Maximum number of combinations to be sampled: {num_combinations}")
        
        current_batch = 0        
        while current_batch < num_combinations:
            sampled_peptides = self.sampler.sample(self.configuration.parameters.batch_size)
            peptide_set_length = len(sampled_peptides.items2) 
            logger.info(f"Sampled {peptide_set_length} peptides")
            
            # Score peptides
            if peptide_set_length != 0:
                mask = np.full((peptide_set_length,), True)
                logger.info("Scoring sampled peptides")
                fragmented_amino_acids = construct_amino_acids_fragments(fillers=sampled_peptides.items2, masked_inputs=sampled_peptides.items1,
                                                                         add_O=True, remove_cyclization_numbers=True)
                
                results = self.scoring_function(sampled_peptides.smilies, mask, mask, fragmented_amino_acids)

                # Log peptides
                results_header, results_rows = get_result_table(results)
                header, rows = self.merge_columns(peptides=sampled_peptides.smilies, aminoacids=sampled_peptides.items2, 
                                                  results_header=results_header, results_rows=results_rows)
                logger.info(f"Writing scoring results to {self.configuration.parameters.output_csv}")

                # Write output
                if if_repeat:
                    header = None
                self.write_csv(self.configuration.parameters.output_csv, header=header, rows=rows)
                if_repeat = True      
                
            # Update batch counter
            current_batch += self.configuration.parameters.batch_size

    def merge_columns(self, peptides: List[str], aminoacids: List, results_header: List, results_rows: List
    ) -> Tuple[List[str], List[List]]:
        """Merge the columns from the input file with the columns from the scoring results

        :param reader: the reader objets with the table from the input file
        :param results_header: the header from the scoring results
        :param results_rows: all rows from the scoring results:
        :returns: the constructed table
        """

        # retain original SMILES name and replace if needed
        header = uniquify_header(['SMILES'] + ['Amino_Acids'] + results_header)

        rows = []

        for smi_row, aa_row, results_row in zip(peptides, aminoacids, results_rows):
            rows.append([smi_row] + [aa_row] + results_row)

        return header, rows


    @staticmethod
    def write_csv(filename: str, rows: List[str], header: Optional[List[str]] = None) -> None:
        """Write a CSV file"""
        if header:
            with open(filename, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

        with open(filename, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)