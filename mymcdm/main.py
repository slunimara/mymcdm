"The main module containing an auxiliary method for decision making."
from pathlib import Path

from numpy.typing import NDArray
from pandas import DataFrame

from . import methods
from . import normalization
from .inout import save_result
from .utils.validation import valid_normalized_matrix
from .utils.misc import make_ranking
from .utils.framing import frame_alternatives, frame_criterions
from .utils.types import Result


def decision(
    a_matrix: NDArray,
    w_vector: NDArray,
    criteria_type: NDArray = None,
    n_method: str | None = None,
    d_method: str = "WSM",
    save: bool = False,
    folder: Path | str = None,
) -> Result:
    """Method for making decision.
    That includes normalization, scoring and saving result.

    Args:
        a_matrix (NDArray): Alternative matrix.
        w_vector (NDArray): Weight vector.
        criteria_type (NDArray): Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.
        n_method (NDArray | None): Normalization method code name that
            represents normalization method which is used to normalize alternatives.
        d_method (str | None): Scoring method code name that represents
            decision method which is used get decision result.
            Defaults to "WSM".
        save (bool, optional): Saves scoring result to the file. Defaults to False.
        folder (pathlib.Path | str, optional): Path to the output folder.
            If None then file will be saved in current folder.

    Code names for normalization and scoring could be found in README.md file.
    """
    # Matrix normalization
    normalized_matrix, criteria_type = normalize(n_method, a_matrix, criteria_type)

    # Framing alternatives
    a_dataframe = frame_alternatives(normalized_matrix, a_types=criteria_type)
    w_series = frame_criterions(w_vector, c_types=criteria_type)

    # Score alternatives
    decision_result = method_decision(d_method, a_dataframe, w_vector, criteria_type)

    path = None

    result: Result = {
        "decision": decision_result,
        "alternatives": a_dataframe,
        "weights": w_series,
        "criteria_type": criteria_type,
        "n_method": n_method,
        "d_method": d_method,
        "path": path,
    }

    if save:
        desc = f"{d_method}_{n_method}"
        path = save_result(result, folder, desc)
        result["path"] = path

    return result


def normalize(code: str | None, a_matrix: NDArray, criteria_type: NDArray):
    """An auxiliary method for selecting the method and
    then normalizing alternative matrix.

    Args:
        code (str): Method code name.
        a_matrix (NDArray): Alternative matrix.
        criteria_type (NDArray): Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    | Code name  | Method name  |
    |-------------|--------------|
    | LOG | Logarithmic normalization |
    | MAX | Max normalization |
    | LINEAR | Linear normalization |
    | MAXMIN | Max-mix normalization |
    | SUM | Sum normalization |
    | VECTOR | Vector normalization |

    Raises:
        ValueError: If method name does not exist.

    Returns normalized alternative matrix.
    """
    match code:
        case None:
            return valid_normalized_matrix(a_matrix), criteria_type
        case "MAXMIN":
            return normalization.max_min(a_matrix, criteria_type)
        case "MAX":
            return normalization.max(a_matrix, criteria_type)
        case "LINEAR":
            return normalization.linear(a_matrix, criteria_type)
        case "VECTOR":
            return normalization.vector(a_matrix, criteria_type)
        case "LOG":
            return normalization.logarithmic(a_matrix, criteria_type)
        case "SUM":
            return normalization.sum(a_matrix, criteria_type)
        case _:
            raise ValueError(
                f'Error: Entered normalization method "{code}" doesn`t exist!'
            )


def method_decision(
    code: str,
    a_dataframe: DataFrame,
    w_vector: NDArray,
    criteria_type: NDArray,
) -> DataFrame:
    """An auxiliary method for selecting the method and
    then deciding the result.

    Args:
        code (str): Method code name.
        a_dataframe (DataFrame): Alternative dataframe.
        w_vector (NDArray): Weight vector.
        criteria_type (NDArray): Binary vector that indicates whether
            the attribute is beneficial (True) or cost (False).
            Defaults sets all attributes as benefitial.

    Raises:
        ValueError: If method name does not exist.

    | Code name | Method name  |
    |-------------|--------------|
    | WSM | Weighted Sum Model |
    | WPM | Weighted Product Model |
    | TOPSIS | Technique for Order of Preference by Similarity to Ideal Solution |
    | VIKOR | VIKOR |
    | ELECTRE | Elimination and Choice Translating Reality |
    | AHP | Analytic hierarchy process |

    Returns decision result as dataframe.
    VIKOR is used repeatedly to obtain a ranking of variants.
    When you enter the ELECTRE method, you get the dominance matrix.
    """
    match code.upper():
        case "WPM":
            result = methods.wpm(a_dataframe, w_vector)
        case "WSM":
            result = methods.wsm(a_dataframe, w_vector)
        case "TOPSIS":
            result = methods.topsis(a_dataframe, w_vector, criteria_type)
        case "AHP":
            result = methods.ahp(a_dataframe, w_vector)
        case "VIKOR":
            result = methods.vikor_ranking(a_dataframe, w_vector, criteria_type)
        case "ELECTRE":
            result = methods.electre(a_dataframe, w_vector, criteria_type)
        case _:
            raise ValueError(f'Error: Entered decision method "{code}" doesn`t exist!')

    # Rank alternatives
    if code.upper() == "ELECTRE":
        size = result.shape[0]
        index = [f"A{i + 1}" for i in range(size)]

        return DataFrame(result, index, index)
    elif result.name == "score":
        score = result.sort_values(ascending=False)
        rank = make_ranking(score)

        decision_dataframe = DataFrame({"score": score, "rank": rank})
        decision_dataframe.sort_index()

        return decision_dataframe
    elif result.name == "rank":
        return DataFrame(result, dtype=int)
