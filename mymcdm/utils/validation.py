"Functions for validating data."
from math import isclose
from functools import wraps

import numpy as np
from pandas import DataFrame
from numpy.typing import NDArray


def valid_normalized_matrix(matrix: NDArray) -> NDArray:
    """Checks if matrix is normalized in range [0, 1].

    Raises:
        ValueError: If number in matrix is not in range [0,1].
    """
    is_numer_in_range = np.logical_and(matrix >= 0, matrix <= 1)
    if not is_numer_in_range.all():
        raise ValueError(
            "Data must be normalized (into the range [0, 1]) "
            "in order to apply scoring methods."
        )

    return matrix


def valid_scoring_args(a_dataframe: DataFrame, w_vector: NDArray):
    """Checks scoring arguments.

    Raises:
        ValueError: If shapes of the alternative dataframe and
            weight vector are not correct.
        ValueError: If sum of the weights is not 1.
    """
    valid_alternative_matrix(a_dataframe.to_numpy())

    if a_dataframe.shape[1] != len(w_vector):
        raise ValueError(
            "Alternative matrix must have "
            "number of columns equal to size of weight vector."
        )

    weights_sum = np.sum(w_vector)

    if not isclose(weights_sum, 1):
        raise ValueError(f"Sum of the weight vector is {weights_sum} and must be 1.")


def valid_scoring_args_extended(
    a_dataframe: DataFrame, w_vector: NDArray, criteria_type: NDArray
):
    """Checks scoring arguments.

    Raises:
        ValueError: If shapes of the alternative dataframe,
            weight vector and criteria type vector are not correct.
        ValueError: If sum of the weights is not 1.
    """
    valid_scoring_args(a_dataframe, w_vector)

    if criteria_type is not None and len(criteria_type) != len(w_vector):
        raise ValueError(
            "Criteria type and weight vector must have same size."
        )


def valid_alternative_matrix(input: any):
    """Checks if input value is valid alternative matrix.

    Raises:
        ValueError: If matrix do not contains only int or float
        ValueError: If matrix is not ndarray and do not have more than one row.
    """
    # Check if matrix is ndarray and have more than one row
    if isinstance(input, np.ndarray) and np.atleast_2d(input).shape[0] <= 1:
        raise ValueError(
            "Matrix must be numpy array with more than one row."
        )

    # Check if matrix contains only int or float
    if not (input.dtype == np.dtype(float) or input.dtype == np.dtype(int)):
        raise ValueError(
            "Matrix must contain integer of float."
        )


def validate_pam(fun):
    """Decorator that checks point alocation method input.

    Raises:
        ValueError: If number in matrix is not positive.
        ValueError: If element in the matrix is not integer.
    """
    @wraps(fun)
    def wrapper(matrix: NDArray):
        is_number_positive = matrix < 0

        if is_number_positive.all():
            raise ValueError("Input matrix must be positive.")

        if not np.issubdtype(matrix.dtype, np.integer):
            raise ValueError(
                "Input matrix must containt only integers. "
                f"Current matrix dtype is {matrix.dtype}"
            )

        matrix = np.atleast_2d(matrix)

        return fun(matrix)

    return wrapper


def validate_normalization_input(fun):
    """Decorator that checks normalization input.

    Raises:
        ValueError: If shapes of the alternative matrix and
            attributes type vector are not correct.
    """
    @wraps(fun)
    def wrapper(matrix: NDArray, attributes_type: NDArray = None):
        valid_alternative_matrix(matrix)

        # Set matrix type to float
        matrix = matrix.astype(float)

        row_size = matrix.shape[1]

        # Default type of attributes is benefitial
        if attributes_type is None:
            attributes_type = np.full(row_size, True)

            return fun(matrix, attributes_type)

        types = np.atleast_1d(attributes_type)
        types_size = types.shape[0]

        # Checks if number of attributes of row match size of attributes_type
        if row_size != types_size:
            raise ValueError(
                "Wrong size of attributes_type argument. "
                f"Expected {row_size} got {types_size}."
            )

        return fun(matrix, attributes_type)

    return wrapper
