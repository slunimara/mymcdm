"""
Pairwise comparison method.

References: [3] [7] [8]
"""
import numpy as np
from numpy.typing import NDArray

from ..utils.validation import valid_alternative_matrix as valid_comparsion_matrix

RANDOM_INDEX = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
    11: 1.51,
    12: 1.54,
    13: 1.56,
    14: 1.57,
    15: 1.59,
}
"Saaty's random index estimates."


def pairwise_comparisons(matrix: NDArray) -> tuple[NDArray, int | None]:
    """Compute priority of comparsion matrix.

    Args:
        matrix (NDArray): Comparsion matrix.

    Returns priority vector and consistency ratio (CR). If size of the matrix
    exceeds random index size then returns Null instead CR.
    """
    valid_comparsion_matrix(matrix)

    priority, eigenvalue = eigenvector_method(matrix)
    return priority, cr(matrix, eigenvalue)


def eigenvector_method(matrix: NDArray) -> NDArray:
    """Eigenvector Method that calculates principal eigenvector.
    Returns normalized principal eigenvector and largest eigenvalue.

    For more information see [7], [8].
    """
    eig_val, eig_vec = np.linalg.eig(matrix)
    max_index = np.argmax(eig_val)

    max_eig_val = eig_val[max_index]
    max_eig_vec = eig_vec[:, max_index]

    norm_eig_vec = max_eig_vec / np.sum(max_eig_vec)

    return norm_eig_vec.real, max_eig_val.real


def pairwise_alternatives(
    comparsion_matrices: list[NDArray] | NDArray,
) -> tuple[NDArray, list[int | None]]:
    """Takes list of comparsion matrices and compute
    alternative matrix using parwise comparsion.

    Args:
        comparsion_matrices (list[NDArray] | NDArray): List of comparsion matrices.

    Returns alternative matrix and consistency ratio of comparsions.
    """
    cr_vector = []
    a_matrix = []

    for matrix in comparsion_matrices:
        priority, cr = pairwise_comparisons(matrix)

        a_matrix.append(priority)
        cr_vector.append(cr)

    a_matrix = np.array(a_matrix).T

    return a_matrix, cr_vector


def is_consistent(ratio: int | None, threshold: int = 0.1) -> bool:
    """Returns boolean value if decision was decided consistently.

    Args:
        ratio (int | None): Consistency ratio.
        threshold (int, optional): Defines treshold for consistency ratio.
            Defaults to 0.1.
    """
    if ratio is not None and ratio <= threshold:
        return True
    return False


def cr(matrix: NDArray, eigenvalue: float) -> float | None:
    """Compute consistency ratio of the comparsion matrix.

    Returns consistency ratio or Null if size of the matrix
    exceeds random index size.
    """
    n, _ = matrix.shape

    if n > len(RANDOM_INDEX):
        return None

    ci = (eigenvalue - n) / (n - 1)
    ri = RANDOM_INDEX[n]

    return ci / ri
