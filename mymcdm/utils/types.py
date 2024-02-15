"Custom dictionary types."

from typing import TypedDict
from pathlib import Path

from pandas import DataFrame, Series
from numpy.typing import NDArray


class Result(TypedDict):
    """Result typed dictionary from decision method.

    Attributes:
        decision (DataFrame): Decision result.
        alternatives (DataFrame): Alternative Dataframe.
            weights (Series): Weight Series.
        criteria_type (NDArray): Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
        n_method (NDArray | None): Normalization method code name that
            represents normalization method which is used to normalize alternatives.
        d_method (str | None): Scoring method code name that represents
            decision method which is used get decision result.
        path (Path | str | None): Path to the output file.
    """

    decision: DataFrame
    alternatives: DataFrame
    weights: Series
    criteria_type: NDArray
    n_method: str | None
    d_method: str
    path: Path | None


class DecisionMatrix(TypedDict):
    """Decision matrix typed dictionary.

    Attributes:
        a_matrix (NDArray): Alternative matrix.
        w_vector (NDArray): Weight vector.
        types (NDArray): Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
    """

    alternatives: DataFrame
    weights: Series
    types: NDArray
