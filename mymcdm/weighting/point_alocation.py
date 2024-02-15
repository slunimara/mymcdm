"""
Point allocation method.

References: [3]
"""
import numpy as np
from numpy.typing import NDArray

from ..utils.validation import validate_pam


@validate_pam
def pam(matrix: NDArray) -> NDArray:
    """Point allocation method. The arithmetic mean
    is used to calculate the weights.

    Args:
        matrix (NDArray): Points matrix.
            Each row represents one expert's opinion.

    Raises:
        ValueError: If number in matrix is not positive.
        ValueError: If element in the matrix is not integer.

    Returns weight vector.
    """
    w_vector = []
    n, _ = matrix.shape

    for row in matrix:
        sum = np.sum(row)
        w_vector.append(row / sum)

    return np.sum(w_vector, axis=0) / n
