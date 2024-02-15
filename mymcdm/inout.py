"Methods for saving and loading data."
import json
import pathlib
from datetime import datetime

from typing import Final

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Series

from .weighting.pairwise import pairwise_comparisons, pairwise_alternatives
from .utils.misc import replace_fractions
from .utils.types import Result, DecisionMatrix

ORIENT_TYPE: Final = "tight"


def load_data(path: pathlib.Path | str) -> tuple[DecisionMatrix | Result, dict]:
    """The method is designed to load the decision matrix data.
    Supported format is only JSON. Each key is optional.
    Key "format" must be set on one of these values "matrix", "pairwise", "result".
    If none of these values is provided then "format" is set on the value "matrix".
    You can find examples in the README.md file or package documentation.

    Fractions in comparsion matrices can be entered in
    string format (e.g. "1/3", "1/5", "1/7") because
    they are then converted to decimal format.

    Args:
        path (pathlib.Path | str): Path of the file.

    Raises:
        ValueError: If Alternative matrix do not have number of columns equal to
            size of weight vector.
        ValueError: If Criteria type and weight vector do not have same size.
        ValueError: If file type is not JSON or CSV.

    Returns:
        For "matrix" format returns DecisionMatrix type dictionary and None.
        For "pairwise" format returns DecisionMatrix type dictionary and dictionary
            that contains "alternative_cr" and "criteria_cr" keys.
        For "result" format returns Result type dictionary and None.
    
    Because all of the objects are optional, there could be None values.
    """
    if path.endswith('.json'):
        data = read_JSON(path)
    else:
        raise ValueError(f'Error: Entered unknown file type!')

    format = None
    if "format" not in data or data["format"] is None:
        format = "matrix"
    else:
        format = data["format"]

    match format.upper():
        case "MATRIX":
            return parse_matrix_format(data), None
        case "PAIRWISE":
            return parse_pairwise_format(data)
        case "RESULT":
            return parse_result_format(data, path), None
        case _:
            raise ValueError(f'Error: Entered unknown format "{format}"!')


def read_JSON(path: pathlib.Path) -> dict:
    """Reads file in the path and returns parsed JSON to dictionary.

    Args:
        path (pathlib.Path): Path of the file.

    Raises:
        FileNotFoundError: If path does not exist or path is not file.
        OSError: When there is an error while reading a file.
        ValueError: When there is an error while parsing a file.
    """
    path = pathlib.Path(path).absolute()

    if not path.exists():
        raise FileNotFoundError(f'Path "{ path }" does not exist.')

    if path.is_dir():
        raise FileNotFoundError(f'File in the path "{ path }" is directory.')

    try:
        with open(path, "r") as data_file:
            raw_data = data_file.read()
            return json.loads(raw_data)
    except OSError as e:
        raise OSError(f"There was an error while reading a file. {type(e)}: {e}")
    except ValueError:
        raise ValueError("There was an error while parsing the file to JSON.")


def parse_matrix_format(
    data: dict,
) -> DecisionMatrix:
    """Auxiulary method for parsing data from dictionary
    that have matrix format.

    Args:
        data (dict): Data in dictionary format
            that has been obtained from the file.

    Raises:
        ValueError: If Alternative matrix do not have number of columns equal to
            size of weight vector.
        ValueError: If Criteria type and weight vector do not have same size.

    Returns DecisionMatrix type dictionary.
    """
    a_matrix = None
    w_vector = None
    types = None

    if "alternatives" in data:
        a_matrix = replace_fractions(data["alternatives"])
        a_matrix = np.array(a_matrix)

    if "weights" in data:
        w_vector = replace_fractions(data["weights"])
        w_vector = np.array(w_vector)

    if "types" in data:
        types = np.array(data["types"])

    is_present = a_matrix is not None and w_vector is not None
    if is_present and a_matrix.shape[1] != len(w_vector):
        raise ValueError(
            "Alternative matrix must have "
            "number of columns equal to size of weight vector."
        )

    is_present = types is not None and w_vector is not None
    if is_present and len(types) != len(w_vector):
        raise ValueError(f"Criteria type and weight vector must have same size.")

    decision_matrix: DecisionMatrix = {
        "alternatives": a_matrix,
        "weights": w_vector,
        "types": types,
    }

    return decision_matrix


def parse_pairwise_format(
    data: dict,
) -> tuple[DecisionMatrix, dict]:
    """Auxiulary for parsing and handling data from dictionary
    that have pairwise format.

    Args:
        data (dict): Data in dictionary format
            that has been obtained from the file.

    Raises:
        ValueError: If criteria comparison matrix does not have
            same number of rows as alternative comparsion matrices.
        ValueError: If Criteria type and weight vector do not have same size.

    Returns DecisionMatrix type dictionary and dictionary
    that contains "alternative_cr" and "criteria_cr" keys.
    """
    a_matrix = None
    w_vector = None
    types = None

    # Consistency ratios for alternatives and criterion
    a_cr = None
    c_cr = None

    comparsion_matrices = []
    if "alternatives" in data:
        for matrix in data["alternatives"]:
            matrix = replace_fractions(matrix)
            comparsion_matrices.append(matrix)
        comparsion_matrices = np.array(comparsion_matrices)

    if "criteria" in data:
        criteria = replace_fractions(data["criteria"])
    
    if "types" in data:
        types = np.array(data["types"])

    if types.dtype == str:
        types = np.char.capitalize(types)
        types = (types == "True")

    is_present = types is not None and w_vector is not None
    if is_present and len(types) != len(w_vector):
        raise ValueError(f"Criteria types and weight vector must have same size.")

    alternatives_count = comparsion_matrices.shape[0]
    is_present = criteria is not None and comparsion_matrices is not None
    if is_present and criteria.shape[0] != alternatives_count:
        raise ValueError(
            "Criteria comparison matrix must have same number of rows as "
            "alternative comparsion matrices."
        )

    a_matrix, a_cr = pairwise_alternatives(np.array(comparsion_matrices))
    w_vector, c_cr = pairwise_comparisons(criteria)

    index = list(range(1, alternatives_count + 1))
    a_cr = Series(a_cr, index, name="CR")

    decision_matrix: DecisionMatrix = {
        "alternatives": a_matrix,
        "weights": w_vector,
        "types": types,
    }

    cr = {"alternative_cr": a_cr, "criteria_cr": c_cr}

    return decision_matrix, cr


def parse_result_format(
    data: dict,
    path: pathlib.Path,
) -> Result:
    """Auxiulary method that parse dictionary that have result format.
    Dataframes saved as dictionary with orient type tight and
    Series as dictionary.

    Args:
        data (dict): Data in dictionary format
            that has been obtained from the file.

    Returns Result type dictionary.
    """
    result: Result = {
        "decision": DataFrame.from_dict(data["decision"], orient=ORIENT_TYPE),
        "alternatives": DataFrame.from_dict(data["alternatives"], orient=ORIENT_TYPE),
        "weights": Series(data["weights"]),
        "criteria_type": Series(data["criteria_type"]),
        "n_method": data["n_method"],
        "d_method": data["d_method"],
        "path": path,
    }

    return result


def save_result(
    data: Result,
    folder: pathlib.Path | str = None,
    desc: str = "",
) -> pathlib.Path:
    """Saves decision data for later use.

    Args:
        a_dataframe (pd.DataFrame): Alternative dataframe.
        w_vector (NDArray): Weight vector.
        decision_result (pd.DataFrame): Decision result.
        folder (pathlib.Path | str, optional): Path to the output folder.
            If None then file will be saved in current folder.
        desc (str, optional): String that will be in the name of the file.

    Returns path to the result file.
    """
    dictionary = {
        "format": "result",
        "decision": data["decision"].to_dict(orient=ORIENT_TYPE),
        "alternatives": data["alternatives"].to_dict(orient=ORIENT_TYPE),
        "weights": data["weights"].tolist(),
        "criteria_type": data["criteria_type"].tolist(),
        "n_method": data["n_method"],
        "d_method": data["d_method"],
    }

    data = json.dumps(dictionary, ensure_ascii=False, indent=4)

    now = datetime.now()
    current_time = now.strftime("%f_%d-%w-%Y")

    if folder is None:
        folder = pathlib.Path.cwd()

    folder = pathlib.Path(folder)

    if desc:
        desc = f"{desc}_"

    path = folder.joinpath(f"{desc}{current_time}.json")

    with open(path, "w") as writer:
        writer.write(data)

    return path
