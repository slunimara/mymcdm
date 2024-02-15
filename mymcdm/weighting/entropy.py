"""
Entropy weights method.

References: [3]
"""
from math import log

import numpy as np
from numpy.typing import NDArray

from ..normalization.sum import sum
from ..utils.validation import valid_alternative_matrix


def entropy_method(a_matrix: NDArray, attributes_type: NDArray = None) -> NDArray:
    """Calculates weights using entropy method.

    Args:
        a_matrix (NDArray): Alternative matrix
        attributes_type (NDArray, optional):
            Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    Returns weight vector.
    """
    valid_alternative_matrix(a_matrix)

    row_size = a_matrix.shape[1]

    if attributes_type is None:
        attributes_type = np.full(row_size, True)

    column_size, _ = a_matrix.shape

    normalized, _ = sum(a_matrix, attributes_type)

    log_matrix = np.log(normalized)
    sum_matrix = np.sum(normalized * log_matrix, axis=0)

    e_matrix = -sum_matrix / log(column_size)

    diversity_degree = 1 - e_matrix
    return diversity_degree / np.sum(diversity_degree)
