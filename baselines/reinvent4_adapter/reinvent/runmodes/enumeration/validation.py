"""Config Validation"""

from typing import Optional
from pydantic import Field

from reinvent.validation import GlobalConfig


class SectionParameters(GlobalConfig):
    smiles_file: str
    output_csv: str = "score_results.csv"
    amino_acid_library_file: str = "nnaas.csv"
    smiles_column: str = "SMILES"
    aa_names_column: str = "Name"  # default updated
    batch_size: int = 100


class SectionResponder(GlobalConfig):
    endpoint: str
    frequency: Optional[int] = Field(1, ge=1)


class AAEnumerationConfig(GlobalConfig):
    parameters: SectionParameters
    scoring: dict = Field(default_factory=dict)  # validate in Scorer
    responder: Optional[SectionResponder] = None


#class AAEnumerationConfig(GlobalConfig):
    #running_mode: Literal["aa_enumeration"] = "aa_enumeration" #str = Field("aa_enumeration", const=True)
    # peptide_sequence: List[str]
    # aa_library_name: str
    # output_file: str
    # batch_size: int = 100
    # scoring_function: ScoringConfig
