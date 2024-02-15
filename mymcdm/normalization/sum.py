"""Implementation of sum normalization method.

References: [1]
"""
import numpy as np
from numpy.typing import NDArray

from ..utils.validation import validate_normalization_input


@validate_normalization_input
def sum(
    matrix: NDArray,
    attributes_type: NDArray = None
) -> tuple[NDArray, NDArray]:
    """Applies sum normalization on input matrix.

    Args:
        matrix (NDArray): Input matrix
        attributes_type (NDArray, optional):
            Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    Raises:
        ValueError: If sum of column is zero.
        ValueError: If sum of inverted values on column is zero.

    Return normalized matrix and boolean matrix
    that indicates new type of attributes.
    """
    # Benefitial and cost denominator
    b_denominator = np.sum(matrix, axis=0)
    c_denominator = np.sum(1 / matrix, axis=0)

    for idx, is_beneficial in enumerate(attributes_type):
        column = matrix[:, idx]

        if is_beneficial:
            if not b_denominator[idx]:
                raise ValueError(f"The sum of column {idx} must not be zero.")

            matrix[:, idx] = column / b_denominator[idx]
        else:
            if not c_denominator[idx]:
                raise ValueError(
                    f"The sum of inverted values on row {idx} "
                    "must not be zero."
                )

            inverted = 1 / column
            matrix[:, idx] = inverted / c_denominator[idx]

    row_size = matrix.shape[1]
    return matrix, np.full(row_size, True)
