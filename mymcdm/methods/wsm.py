"""The weighted sum model

References: [4]
"""
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from ..utils.validation import valid_scoring_args


def wsm(a_dataframe: DataFrame, w_vector: NDArray) -> Series:
    """The weighted sum model method.

    Args:
        a_dataframe (pd.DataFrame): Alternative matrix.
        w_vector (NDArray): Weight vector.

    Returns WSM score vector. The best alternative
    (in the maximalization case) have the biggest
    value in the vector.
    """
    valid_scoring_args(a_dataframe, w_vector)

    w_matrix = np.multiply(a_dataframe, w_vector)
    score = np.sum(w_matrix, axis=1)

    return Series(score, name="score")
