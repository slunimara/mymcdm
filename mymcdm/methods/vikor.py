"""VIKOR

References: [5]
"""
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from ..utils.misc import determine_ideals
from ..utils.validation import valid_scoring_args_extended


def vikor(
    a_dataframe: DataFrame,
    w_vector: NDArray,
    criteria_type: NDArray,
    v_value: int = 0.5,
) -> list[str]:
    """The VIKOR method.

    Args:
        a_dataframe (pd.DataFrame): Alternative matrix.
        w_vector (NDArray): Weight vector.
        criteria_type (NDArray): Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.
        v_value (int, optional): Maximum group utility value.
            Defaults sets to 0.5.

    Returns one or more best solutions as alternative row index
    together with utility, regret and q order of alternatives.
    """
    valid_scoring_args_extended(a_dataframe, w_vector, criteria_type)

    # Determine the positive-ideal and the negative-ideal solutions
    positive_ideal, negative_ideal = determine_ideals(a_dataframe, criteria_type)

    # Calculate the utility and regret measures
    formula = (positive_ideal - a_dataframe) / (positive_ideal - negative_ideal)
    weighted = w_vector * formula

    utility = np.sum(weighted, axis=1)
    regret = np.max(weighted, axis=1)

    # Calculating the Q vector
    nominator_u = v_value * (utility - min(utility))
    nominator_r = (1 - v_value) * (regret - min(regret))

    denominator_u = max(utility) - min(utility)
    denominator_r = max(regret) - min(regret)

    u_group = nominator_u / denominator_u
    r_group = nominator_r / denominator_r

    q_vector = u_group + r_group

    # Index the alternatives
    index = a_dataframe.index

    utility = Series(utility, index, name="utility")
    regret = Series(regret, index, name="regret")
    q_vector = Series(q_vector, index, name="q")

    # Rank the alternatives based on utility, regret and Q
    u_order = utility.sort_values(ascending=True)
    r_order = regret.sort_values(ascending=True)
    q_order = q_vector.sort_values(ascending=True)

    # Determine the best solution that satisfies conditions
    alternatives = q_order.index

    row_size = a_dataframe.shape[0]
    dq = 1 / (row_size - 1)

    acceptable_advantage = (q_order[1] - q_order[0]) >= dq
    acceptable_stability = max(u_order) == u_order[0] and max(r_order) == r_order[0]

    if not acceptable_advantage:
        solutions = [alternatives[0], alternatives[1]]

        for i in range(2, row_size):
            if not q_order[i] - q_order[0] < dq:
                break

            solutions.append(alternatives[i])
    elif not acceptable_stability:
        solutions = [alternatives[0], alternatives[1]]
    else:
        solutions = alternatives[0]

    return solutions, u_order, r_order, q_order


def vikor_ranking(
    a_dataframe: DataFrame,
    w_vector: NDArray,
    criteria_type: NDArray,
    v_value: int = 0.5,
) -> Series:
    """Applying the VIKOR method repeatedly to obtain a ranking of alternatives.

    Args:
        a_dataframe (pd.DataFrame): Alternative matrix.
        w_vector (NDArray): Weight vector.
        criteria_type (NDArray): Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.
        v_value (int, optional): Maximum group utility value. Defaults sets to 0.5.

    Returns rank of the alternatives in Series.
    """
    a_dataframe = a_dataframe.copy()
    rank = 1
    result = {}

    while a_dataframe.shape[0] >= 2:
        solutions, _, _, _ = vikor(a_dataframe, w_vector, criteria_type, v_value)
        a_dataframe.drop(solutions, inplace=True)

        for alt in solutions:
            result.update({alt: rank})

        rank += 1

    if a_dataframe.shape[0] == 1:
        alt = a_dataframe.iloc[0].name
        result.update({alt: rank})

    result = Series(result, name="rank")
    return result.sort_index()
