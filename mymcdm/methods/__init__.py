"Submodule for MCDM scoring methods."
from .vikor import vikor, vikor_ranking
from .ahp import ahp, ahp_cm
from .electre import electre
from .topsis import topsis
from .wpm import wpm
from .wsm import wsm

__all__ = ["vikor", "vikor_ranking", "ahp", "ahp_cm", "electre", "topsis", "wpm", "wsm"]
