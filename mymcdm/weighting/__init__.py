"Submodule for weighting methods."
from .pairwise import pairwise_comparisons, pairwise_alternatives, is_consistent
from .entropy import entropy_method
from .mean import mean_weight
from .statistical import standard_deviation, svp, critic
from .point_alocation import pam

__all__ = [
    "pairwise_comparisons",
    "pairwise_alternatives",
    "is_consistent",
    "entropy_method",
    "mean_weight",
    "standard_deviation",
    "svp",
    "critic",
    "pam",
]
