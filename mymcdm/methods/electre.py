"""ELECTRE

References: [4] [5] [6]
"""
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from ..utils.validation import valid_scoring_args_extended


def electre(
    a_dataframe: DataFrame,
    w_vector: NDArray,
    criteria_type: NDArray,
    c_threshold: int = None,
    d_threshold: int = None,
) -> Series:
    """The ELECTRE method.

    Args:
        a_dataframe (pd.DataFrame): Alternative matrix.
        w_vector (NDArray): Weight vector.
        criteria_type (NDArray): Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.
        c_threshold (int, optional): Concordance threshold.
            Defaults to None.
        d_threshold (int, optional): Discordance threshold.
            Defaults to None.

    If c_threshold or d_threshold is set to None
    then concordance or discordance threshold is calculated
    as arithmetic mean of corcondance or discordance matrix.


    Returns vector that indicates order of alternatives.
    The best alternative have the biggest value in the vector.
    """
    valid_scoring_args_extended(a_dataframe, w_vector, criteria_type)

    # Construct the weighted normalized matrix
    wn_matrix = a_dataframe * w_vector

    # Determine the concordance and discordance matrices
    column_size = a_dataframe.shape[0]
    c_matrix, d_matrix = concordance_discordance_matrices(
        wn_matrix, criteria_type, w_vector
    )

    # Determine the concordance and discordance dominance matrices
    fraction = 1 / (column_size * (column_size - 1))

    if c_threshold is None:
        c_threshold = fraction * np.sum(c_matrix)

    if d_threshold is None:
        d_threshold = fraction * np.sum(d_matrix)

    f_condlist = [c_matrix >= c_threshold, c_matrix < c_threshold]
    f_dominance = np.piecewise(c_matrix, f_condlist, [1, 0])

    g_condlist = [d_matrix >= d_threshold, d_matrix < d_threshold]
    g_dominance = np.piecewise(d_matrix, g_condlist, [1, 0])

    # Determine the total dominance matrix
    dominance_matrix = f_dominance * g_dominance

    return dominance_matrix.astype(int)


def concordance_discordance_matrices(
    wn_matrix: DataFrame,
    criteria_type: NDArray,
    w_vector: NDArray,
):
    """Determine the concordance and discordance matrices.

    Args:
        wn_matrix (pd.DataFrame): Weighted normalized matrix.
        criteria_type (NDArray): Binary vector that indicates whether
        the attribute is beneficial (True) or cost (False).
        Defaults sets all attributes as benefitial.
        w_vector (NDArray): Weight vector.
    """
    column_size, row_size = wn_matrix.shape

    if criteria_type is None:
        criteria_type = np.full(row_size, True)

    c_matrix = np.full((column_size, column_size), 0.0)
    d_matrix = np.full((column_size, column_size), 0.0)

    for k in range(column_size):
        for l in range(k + 1, column_size):
            k_row = wn_matrix.iloc[k]
            l_row = wn_matrix.iloc[l]

            c_kl, d_kl = concordance_set(k_row, l_row, criteria_type)
            c_matrix[k][l] = concordance_index(c_kl, w_vector)
            d_matrix[k][l] = discordance_index(d_kl, k_row, l_row)

            c_lk, d_lk = concordance_set(l_row, k_row, criteria_type)
            c_matrix[l][k] = concordance_index(c_lk, w_vector)
            d_matrix[l][k] = discordance_index(d_lk, l_row, k_row)

    return c_matrix, d_matrix


def concordance_set(
    k_row: NDArray, l_row: NDArray, criteria_type: NDArray
) -> tuple[set, set]:
    "Calculates discordance index. For more information see [5] (2.15)"
    c_kl = set()
    d_kl = set()

    for j, val in enumerate(criteria_type):
        # For benefitial criteria first part of condition must be true
        # For cost criteria the other part
        if (val and k_row[j] >= l_row[j]) or (not val and k_row[j] < l_row[j]):
            c_kl.add(j)
        else:
            d_kl.add(j)

    return c_kl, d_kl


def concordance_index(concordance_set: set, w_vector: NDArray) -> float:
    "Calculates corcordnace index."
    sum = 0

    for j in concordance_set:
        sum += w_vector[j]

    return sum


def discordance_index(discordance__set: set, k_row: NDArray, l_row: NDArray) -> float:
    "Calculates discordance index. For more information see [5] (2.16)"
    rows_difference = np.absolute(k_row - l_row)
    restriction = list(discordance__set)

    if not len(restriction):
        return 0

    denominator = np.max(rows_difference)
    numerator = np.max(rows_difference[restriction])

    return numerator / denominator
