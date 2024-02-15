"""Implementation of linear normalization method.

References: [2]
"""
import numpy as np
from numpy.typing import NDArray

from ..utils.validation import validate_normalization_input


@validate_normalization_input
def linear(
    matrix: NDArray,
    attributes_type: NDArray = None
) -> tuple[NDArray, NDArray]:
    """Applies linear normalization on input matrix.

    Args:
        matrix (NDArray): Input matrix
        attributes_type (NDArray, optional):
            Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    Raises:
        ValueError: If maximum value in the colum that is benefitial is zero.
        ValueError: If minimum value in the colum that is cost is zero.

    Return normalized matrix and boolean matrix
    that indicates new type of attributes.
    """
    for idx, is_beneficial in enumerate(attributes_type):
        column = matrix[:, idx]

        if is_beneficial:
            max = column.max()

            if not max:
                raise ValueError(
                    "The maximum value in the colum that is benefitial "
                    "must not be zero."
                )

            matrix[:, idx] = column / max
        else:
            min = column.min()

            if not min:
                raise ValueError(
                    "The minimum value in the colum that is cost "
                    "must not be zero."
                )

            matrix[:, idx] = min / column

    row_size = matrix.shape[1]
    return matrix, np.full(row_size, True)
