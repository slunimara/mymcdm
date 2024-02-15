"Submodule for matrix normalization."
from .max import max
from .linear import linear
from .max_min import max_min
from .vector import vector
from .sum import sum
from .logarithmic import logarithmic

__all__ = ["max", "linear", "max_min", "vector", "sum", "logarithmic"]
