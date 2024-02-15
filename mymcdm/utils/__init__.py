"Submodule for utilility methods."
from .framing import (
    frame_alternatives,
    frame_criterions,
    make_decision_matrix,
    decompose_decision_matrix,
)

from .types import Result, DecisionMatrix

from .misc import (
    make_ranking,
    replace_fractions,
)

__all__ = [
    "frame_alternatives",
    "frame_criterions",
    "make_decision_matrix",
    "decompose_decision_matrix",
    "make_ranking",
    "replace_fractions",
    "Result",
    "DecisionMatrix",
]
