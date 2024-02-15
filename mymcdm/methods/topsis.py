"""TOPSIS

References: [4] [5]
"""
from pandas import DataFrame, Series
from numpy.linalg import norm as euclidean_distance
from numpy.typing import NDArray

from ..utils.misc import determine_ideals
from ..utils.validation import valid_scoring_args_extended


def topsis(
    a_dataframe: DataFrame,
    w_vector: NDArray,
    criteria_type: NDArray
) -> Series:
    """The TOPSIS method.

    Args:
        a_dataframe (pd.DataFrame): Alternative matrix.
        w_vector (NDArray): Weight vector.
        criteria_type (NDArray): Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    Raises:
        ValueError: Diference of positive distance and negative distance is zero.

    Returns score vector. The best alternative
    (in the maximalization case) have the biggest
    value in the vector.
    """
    valid_scoring_args_extended(a_dataframe, w_vector, criteria_type)

    # Construct the weighted normalized matrix
    wn_matrix = a_dataframe * w_vector

    # Determine the positive-ideal and the negative-ideal solutions
    positive_ideal, negative_ideal = determine_ideals(wn_matrix, criteria_type)

    # Calculate separation measure using Euclidean distance method
    positive_distances = euclidean_distance(wn_matrix - positive_ideal, axis=1)
    negative_distances = euclidean_distance(wn_matrix - negative_ideal, axis=1)

    # Calculate the relative closeness to the positive-ideal solution
    denominator = positive_distances + negative_distances

    if not all(denominator):
        raise ValueError(
            """Diference of positive distance and negative distance
            must not be zero."""
        )

    result = negative_distances / denominator

    return Series(result, name="score", index=a_dataframe.index)
