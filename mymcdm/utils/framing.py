"Functions for framing matrices."
import numpy as np
from pandas import DataFrame, Series, Index
from numpy.typing import NDArray


def frame_alternatives(
    a_matrix: NDArray, row_names: NDArray = None, a_types: NDArray = None
) -> DataFrame:
    """Takes alternative matrix and returns Dataframe with heading.

    Args:
        a_matrix (NDArray): Alternative matrix.
        row_names (NDArray, optional): Name for the row indices.
            Defaults set indices as A1, A2,...
        a_types (NDArray, optional): Attributes types.
            Defaults set as beneficial.

    Symbol * indicates if criteria is beneficial or cost.
    """
    rows, columns = a_matrix.shape

    if a_types is None:
        a_types = [f"C{i + 1}+" for i in range(columns)]
    else:
        a_types = [
            f"C{i + 1}+" if val else f"C{i + 1}-" for i, val in enumerate(a_types)
        ]

    if row_names is None:
        row_names = [f"A{i + 1}" for i in range(rows)]

    row_index = Index(row_names, name="Alts.")

    return DataFrame(a_matrix, row_index, a_types)


def frame_criterions(
    w_vector: NDArray, column_names: NDArray = None, c_types: NDArray = None
) -> Series:
    """Takes weight vector and returns Series with heading.

    Args:
        w_vector (NDArray): Weight vector.
        column_names (NDArray, optional): Name for the column indices.
            Defaults set indices as C1, C2,...
        c_types (NDArray, optional): Criterion types.
            Defaults set as beneficial.

    Symbol * indicates if criteria is beneficial or cost.
    """
    columns = w_vector.shape[0]

    if column_names is None:
        column_names = [f"C{i + 1}" for i in range(columns)]

    if c_types is None:
        c_types = np.full(len(column_names), True)

    column_names = [
        f"{header}+" if val else f"{header}-"
        for header, val in zip(column_names, c_types)
    ]

    index = Index(column_names, name="Crits.")

    return Series(w_vector, index, name="weights")


def make_decision_matrix(a_dataframe: DataFrame, w_series: Series) -> DataFrame:
    """Takes alternatives dataframe and weight series and returns decision matrix.

    Args:
        a_dataframe (DataFrame): Alternative DataFrame.
        w_series (Series): Weight Series.
    """
    head = [w_series.index, w_series.values]

    return a_dataframe.set_axis(head, axis=1)


def decompose_decision_matrix(
    decision_matrix: DataFrame,
) -> tuple[DataFrame, NDArray, NDArray]:
    """Takes decision matrix DataFrame from make_decision_matrix method
    and returns alternative matrix, weights, criteria type.

    Args:
        decision_matrix (DataFrame): DataFrame from
            make_decision_matrix method.

    Returns:
        - a_dataframe (DataFrame): Alternative DataFrame.
        - w_vector (NDArray): Weight vector.
        - c_types (NDArray): Criterion types.
    """
    frame = decision_matrix.columns.to_frame()

    weights_index = frame[0].values
    w_vector = frame[1].values

    a_matrix = decision_matrix.copy().set_axis(weights_index, axis=1)
    criteria_type = [index[-1] == "+" for index in weights_index]

    return a_matrix, w_vector, criteria_type
