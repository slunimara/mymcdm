"""Implementation of max normalization method.

References: [1]
"""
import numpy as np
from numpy.typing import NDArray

from ..utils.validation import validate_normalization_input


@validate_normalization_input
def max(
    matrix: NDArray,
    attributes_type: NDArray = None
) -> tuple[NDArray, NDArray]:
    """Applies max normalization on input matrix.

    Args:
        matrix (NDArray): Input matrix
        attributes_type (NDArray, optional):
            Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    Raises:
        ValueError: If maximum value in the colum is zero.

    Return normalized matrix and boolean matrix
    that indicates new type of attributes.
    """
    max_values = matrix.max(axis=0)

    # Calculate for benefitial criteria
    new_matrix = matrix / max_values

    # Modify for cost criteria
    for idx, is_beneficial in enumerate(attributes_type):
        if not is_beneficial:
            new_matrix[:, idx] = 1 - new_matrix[:, idx]

    row_size = matrix.shape[1]
    return new_matrix, np.full(row_size, True)
