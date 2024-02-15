"Miscellaneous auxiliary functions."
import numpy as np
from pandas import Series
from numpy.typing import NDArray


def determine_ideals(
    matrix: NDArray, criteria_type: NDArray
) -> tuple[NDArray, NDArray]:
    """Determine positive and negative ideals.
    Tuple of positive ideal and negative ideal.

    Args:
        matrix (NDArray): Input matrix.
        criteria_type (NDArray): Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    """
    row_size = matrix.shape[1]

    if criteria_type is None:
        criteria_type = np.full(row_size, True)

    positive_ideal = []
    negative_ideal = []

    max_vector = matrix.max(axis=0)
    min_vector = matrix.min(axis=0)

    for max, min, val in zip(max_vector, min_vector, criteria_type):
        if val:
            positive_ideal.append(max)
            negative_ideal.append(min)
        else:
            positive_ideal.append(min)
            negative_ideal.append(max)

    return np.array(positive_ideal), np.array(negative_ideal)


def make_ranking(score: Series) -> Series:
    "From given alternative score creates ranking."
    ranking = []
    previous = score[0]
    rank = 1

    for alt in score:
        if alt != previous:
            rank += 1

        ranking.append(rank)
        previous = alt

    return Series(ranking, score.index)


def replace_fractions(matrix: NDArray | list) -> NDArray:
    """Replace string fractions in numpy matrix and returns
    matrix with floats. Accuracy is to 16 decimal places.
    Works faster with a duplicate occurrence of a fraction.
    Data must be number of fraction in form \"x/y\"
    where x, y is numbers and y must not be zero.

    Example:
    ```
    # Input
    ["12", "1/3", "1/5", "1/7", "1/8"]

    # Output
    [12, 0.3333333333333333, 0.2, 0.14285714285714285, 0.125]
    ```
    """
    matrix = np.array(matrix, dtype="<U32")

    uniq = np.unique(matrix)
    mask = np.frompyfunc(lambda x: "/" in x, 1, 1)(uniq)
    mask = np.array(mask, dtype=bool)

    fractions = uniq[mask]

    for fraction in fractions:
        mask = matrix == fraction
        try:
            numerator, denominator = fraction.split("/")
            matrix[mask] = int(numerator) / int(denominator)
        except Exception:
            raise ValueError(
                "Data must be number of fraction in form \"x/y\" "
                "where x, y is numbers and y must not be zero. "
                f"Value {fraction} was provided."
            )

    return matrix.astype("f8")
