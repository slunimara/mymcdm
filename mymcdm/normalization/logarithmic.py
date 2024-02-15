"""Implementation of logarithmic normalization method

References: [1]
"""
import numpy as np
from numpy.typing import NDArray

from ..utils.validation import validate_normalization_input


@validate_normalization_input
def logarithmic(
    matrix: NDArray,
    attributes_type: NDArray = None
) -> tuple[NDArray, NDArray]:
    """Applies vector logarithmic on input matrix.

    Args:
        matrix (NDArray): Input matrix.
        attributes_type (NDArray, optional):
            Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    Return normalized matrix and boolean matrix
    that indicates new type of attributes.
    """
    row_size = matrix.shape[1]

    # Denominator variable for code clarity
    prod = np.prod(matrix, axis=0)
    column_log = np.log(prod)
    log = np.log(matrix)

    # Calculate for benefitial criteria
    new_matrix = log / column_log

    # Modify for cost criteria
    for idx, is_beneficial in enumerate(attributes_type):
        if not is_beneficial:
            column = new_matrix[:, idx]

            new_matrix[:, idx] = (1 - column) / row_size

    return new_matrix, np.full(row_size, True)
