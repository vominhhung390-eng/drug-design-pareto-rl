import importlib.resources
from contextlib import contextmanager

import reinvent.chemistry.library_design.reaction_definitions.data


@contextmanager
def default_reaction_definitions():
    ref = importlib.resources.files(
        reinvent.chemistry.library_design.reaction_definitions.data
    ).joinpath("reaction_definitions.csv")
    with importlib.resources.as_file(ref) as reaction_definitions_path:
        yield reaction_definitions_path
