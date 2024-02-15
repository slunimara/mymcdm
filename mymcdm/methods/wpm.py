"""The weighted product model

References: [4]
"""
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from ..utils.validation import valid_scoring_args


def wpm(a_dataframe: DataFrame, w_vector: NDArray) -> Series:
    """The weighted product model method.

    Args:
        a_dataframe (pd.DataFrame): Alternative matrix.
        w_vector (NDArray): Weight vector.

    Returns WPM score vector. The best alternative
    (in the maximalization case) have the biggest
    value in the vector.
    """
    valid_scoring_args(a_dataframe, w_vector)

    amplified = np.power(a_dataframe, w_vector)
    score = np.prod(amplified, axis=1)

    return Series(score, name="score")
