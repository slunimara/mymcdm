"""Implementation of max-min normalization method

References: [1]
"""
import numpy as np
from numpy.typing import NDArray

from ..utils.validation import validate_normalization_input


@validate_normalization_input
def max_min(
    matrix: NDArray,
    attributes_type: NDArray = None
) -> tuple[NDArray, NDArray]:
    """Applies min-max normalization on input matrix.

    Args:
        matrix (NDArray): Input matrix
        attributes_type (NDArray, optional):
            Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    Return normalized matrix and boolean matrix
    that indicates new type of attributes.
    """
    row_size = matrix.shape[1]

    max_values = matrix.max(axis=0)
    min_values = matrix.min(axis=0)

    for idx, is_beneficial in enumerate(attributes_type):
        column = matrix[:, idx]
        max = max_values[idx]
        min = min_values[idx]

        if not (max - min):
            matrix[:, idx] = np.full(row_size, 1)
            continue

        if is_beneficial:
            column -= min
        else:
            column = max - column

        matrix[:, idx] = column / (max - min)

    return matrix, np.full(row_size, True)
