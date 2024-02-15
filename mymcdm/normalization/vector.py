"""Implementation of vector normalization method

References: [1]
"""
import numpy as np
from numpy.typing import NDArray

from ..utils.validation import validate_normalization_input


@validate_normalization_input
def vector(
    matrix: NDArray,
    attributes_type: NDArray = None
) -> tuple[NDArray, NDArray]:
    """Applies vector normalization on input matrix.

    Args:
        matrix (NDArray): Input matrix
        attributes_type (NDArray, optional):
            Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    Return normalized matrix and boolean matrix
    that indicates new type of attributes.
    """
    # Denominator variable for code clarity
    amplified = np.power(matrix, 2)
    sum = np.sum(amplified, axis=0)

    denominator = np.sqrt(sum)

    # Calculate for benefitial criteria
    new_matrix = matrix / denominator

    # Modify for cost criteria
    for idx, is_beneficial in enumerate(attributes_type):
        if not is_beneficial:
            new_matrix[:, idx] = 1 - new_matrix[:, idx]

    row_size = matrix.shape[1]

    return new_matrix, np.full(row_size, True)
