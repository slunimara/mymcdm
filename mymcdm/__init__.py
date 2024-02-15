"""
.. include:: ../README.md
"""
from .main import decision
from .methods import vikor, vikor_ranking, ahp, ahp_cm, electre, topsis, wpm, wsm
from .inout import load_data

from . import weighting
from . import normalization
from . import inout
from . import methods
from . import utils

__all__ = [
    "decision",
    "load_data",
    "vikor",
    "vikor_ranking",
    "ahp",
    "ahp_cm",
    "electre",
    "topsis",
    "wpm",
    "wsm",
    "inout",
    "normalization",
    "weighting",
    "methods",
    "utils",
]
