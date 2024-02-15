"""
Methods for calculating weights using statistics.

References: [3]
"""
import numpy as np
from numpy.typing import NDArray

from ..normalization.max_min import max_min
from ..utils.validation import valid_alternative_matrix


def standard_deviation(a_matrix: NDArray) -> NDArray:
    """Calculates weights using Standard Deviation method.

    Args:
        a_matrix (NDArray): Alternative matrix

    Returns weight vector.
    """
    valid_alternative_matrix(a_matrix)

    sd_vector = np.std(a_matrix, axis=0)
    return sd_vector / np.sum(sd_vector)


def svp(a_matrix: NDArray) -> NDArray:
    """Calculates weights using Statistical Variance Procedure.

    Args:
        a_matrix (NDArray): Alternative matrix

    Returns weight vector.
    """
    valid_alternative_matrix(a_matrix)

    sv_vector = np.var(a_matrix, axis=0)
    return sv_vector / np.sum(sv_vector)


def critic(a_matrix: NDArray, attributes_type: NDArray = None) -> NDArray:
    """Calculates weights using Criteria importance through inter-criteria method.

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

    normalized, _ = max_min(a_matrix, attributes_type)

    correlation_coef = np.corrcoef(normalized)
    correlation_coef = correlation_coef[0:row_size, 0:row_size]

    sd = np.std(normalized, axis=0)
    beta_vector = sd * np.sum(1 - correlation_coef, axis=1)

    return beta_vector / np.sum(beta_vector)
