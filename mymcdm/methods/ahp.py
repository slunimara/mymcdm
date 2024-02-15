"""Analytic hierarchy process (AHP)

References: [4] [5]
"""
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from ..methods.wsm import wsm
from ..weighting.pairwise import (
    pairwise_comparisons,
    pairwise_alternatives,
    is_consistent,
)
from ..utils.validation import valid_scoring_args
from ..utils.framing import frame_alternatives


def ahp(a_dataframe: DataFrame, w_vector: NDArray) -> Series:
    """The final step of Analytic hierarchy process (AHP).

    Args:
        a_dataframe (pd.DataFrame): Alternative matrix.
        w_vector (NDArray): Weight vector.

    Raises:
        ValueError: If alternative matrix row sum isn't approximately
            equal to 1.

    Returns AHP score vector. The best alternative
    (in the maximalization case) have the biggest
    value in the vector.
    """
    if not alternatives_validation(a_dataframe):
        raise ValueError(
            "Alternative matrix row sum must be approximately equal to 1"
        )

    valid_scoring_args(a_dataframe, w_vector)

    return wsm(a_dataframe, w_vector)


def ahp_cm(
    alternatives_cm: list[NDArray] | NDArray,
    criteria_cm: NDArray
) -> tuple[Series, bool]:
    """Compute AHP-score from given alternatives comparsion matrices and
      criteria comparsion matrix.

    Args:
        alternatives_cm (list[NDArray] | NDArray): List of comparsion matrices.
        criteria_cm (NDArray): Comparsion matrix.

    Raises:
        ValueError: If number of columns of alternatives isn't
            equal to number of criteria.
        ValueError: If alternative matrix row sum isn't
            approximately equal to 1.

    Returns AHP score vector and boolean value is pairwise comparsions
    was consistent. The best alternative (in the maximalization case)
    have the biggest value in the vector.
    """
    criteria_count = criteria_cm.shape[0]

    if len(alternatives_cm) != criteria_count:
        raise ValueError(
            f"""Alternative columns count {len(alternatives_cm)}
        isn't equal to number of criteria {criteria_count}."""
        )

    a_matrix, a_cr = pairwise_alternatives(alternatives_cm)
    w_vector, c_cr = pairwise_comparisons(criteria_cm)

    if not alternatives_validation(a_matrix):
        raise ValueError(
            "Alternative matrix row sum must be approximately equal to 1"
        )

    a_dataframe = frame_alternatives(a_matrix)

    cr = np.append(a_cr, c_cr)
    consistent = [is_consistent(val) for val in cr]

    return ahp(a_dataframe, w_vector), all(consistent)


def alternatives_validation(a_matrix: NDArray) -> bool:
    "Returns True if row sum is approximately equal to 1."

    sum = np.sum(a_matrix, axis=0)
    return np.allclose(sum, 1)
