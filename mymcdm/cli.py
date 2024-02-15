"Console part of this package."
import sys

import numpy as np

from typing import Final

from . import main
from .utils.framing import make_decision_matrix
from .inout import load_data, save_result

OPTIONS = {
    "verbose": "-v",
    "normalization": "-n",
    "decision": "-d",
}

HELP_TEXT: Final = f"""Tool for handling Multiple Attribute Decision Making problems.

Usage:
\tmymcdm <command> [options]

Commands:
\tdecision\tMakes a decision based on the data entered from the file and saves it.
\thelp\t\tShows help.
\tlist\t\tShows list of normalization and decision methods.

General Options for decision command:
\t{OPTIONS['verbose']}\t\tGive more output.
\t{OPTIONS['normalization']} <code_name>\tNormalization method name or "NONE".
\t{OPTIONS['decision']} <code_name>\tDecision method name.
"""

METHODS_TEXT: Final = f"""
Decision making methods
| Code name  | Method name  |
|-------------|--------------|
| WSM | Weighted Sum Model |
| WPM | Weighted Product Model |
| TOPSIS | Technique for Order of Preference by Similarity to Ideal Solution |
| VIKOR | VIKOR |
| ELECTRE | Elimination and Choice Translating Reality |
| AHP | Analytic hierarchy process |

Normalization methods
| Code name  | Method name  |
|-------------|--------------|
| LOG | Logarithmic normalization |
| MAX | Max normalization |
| LINEAR | Linear normalization  |
| MAXMIN | Max-min normalization |
| SUM | Sum normalization |
| VECTOR | Vector normalization |

For more information, please refer to the relevant documentation or funtion code."""

NORMALIZATION_METHODS: Final = [
    "MAX",
    "LINEAR",
    "MAXMIN",
    "VECTOR",
    "SUM",
    "LOG",
    "NONE",
]

DECISION_METHODS: Final = [
    "VIKOR",
    "VIKOR_RANKING",
    "AHP",
    "AHP_CM",
    "ELECTRE",
    "TOPSIS",
    "WPM",
    "WSM",
]

ERROR_MISSING_ARGUMENT: Final = "Error: There is missing argument after {} option."
ERROR_UNKNOWN_METHOD: Final = "Error: Entered {} method \"{}\" doesn`t exist!"
ERROR_UNKNOWN_COMMAND: Final = "Error: Unknown command \"{}\"."
ERROR_MISSING_DATA: Final = "Error: In the input file you must provide {}."


def cli():
    """Method for package entry point."""
    args = sys.argv[1:]

    command = ""
    if len(args):
        command = args[0]
        options = args[1:]

    match command.lower():
        case "help" | "":
            print(HELP_TEXT)
        case "decision":
            cli_decision(options)
        case "list":
            print(METHODS_TEXT)
        case _:
            print(ERROR_UNKNOWN_COMMAND.format(command))


def cli_decision(options: list[str]):
    "Console decision proces that makes decision bases on options and path."
    verbose, n_method, d_method = get_parameters(options)
    data, cr = cli_load_data()

    if data["alternatives"] is None:
        sys.exit(ERROR_MISSING_DATA.format("alternative matrix"))

    if data["weights"] is None:
        sys.exit(ERROR_MISSING_DATA.format("weighted vector"))

    if data["types"] is None:
        row_size = data["weights"].shape[0]
        data["types"] = np.full(row_size, True)

    result = main.decision(
        data["alternatives"], data["weights"], data["types"], n_method, d_method
    )

    if verbose:
        dm = make_decision_matrix(result["alternatives"], result["weights"])

        print("Decision matrix:")
        print(dm)
        print()

    if cr is not None:
        a_cr = cr["alternative_cr"]
        c_cr = cr["criteria_cr"]

        print("The consistency ratio is:")
        print(f"For alternatives: {a_cr}")
        print(f"For criterion: {c_cr}")
        print()

    print("The result is:")
    print(result["decision"])

    save_data(result)


def get_parameters(options: list[str]):
    "Metod for parsing parameters of command."
    verbose, n_method, d_method = parse_options(options)

    if n_method is None:
        print("Enter normalization method name or \"NONE\".")
        try:
            n_method = input()
            print()
        except Exception as e:
            sys.exit(str(e))

    if n_method.upper() not in NORMALIZATION_METHODS:
        sys.exit(ERROR_UNKNOWN_METHOD.format("normalization", n_method))

    if n_method.upper() == "NONE":
        n_method = None

    if d_method is None:
        print("Enter decision method name.")

        try:
            d_method = input()
            print()
        except Exception as e:
            sys.exit(str(e))

    if d_method.upper() not in DECISION_METHODS:
        sys.exit(ERROR_UNKNOWN_METHOD.format("decision", d_method))

    return verbose, n_method, d_method


def parse_options(options: list[str]):
    "Method for parsing options of command."
    verbose = False
    n_method = None
    d_method = None

    is_present = OPTIONS["verbose"] in options
    if is_present:
        verbose = True

    is_present = OPTIONS["normalization"] in options
    if is_present:
        index = options.index(OPTIONS["normalization"])

        if index + 1 == len(options):
            sys.exit(ERROR_MISSING_ARGUMENT.format(OPTIONS["normalization"]))

        n_method = options[index + 1] or None

    is_present = OPTIONS["decision"] in options
    if is_present:
        index = options.index(OPTIONS["decision"])

        if index + 1 == len(options):
            sys.exit(ERROR_MISSING_ARGUMENT.format(OPTIONS["decision"]))

        d_method = options[index + 1] or None

    return verbose, n_method, d_method


def cli_load_data():
    "Load data from file."
    print("Enter the path to the data file: ")

    try:
        file_path = input()
        print()
        
        return load_data(file_path)
    except Exception as e:
        sys.exit(str(e))


def save_data(result):
    "Asks the user if they want to save the data and saves it if desired."

    print()
    print("Do you want to save result ? [Y / N]: ")
    try:
        save = input()
    except Exception as e:
        sys.exit(str(e))

    if save.upper() != "Y":
        sys.exit()

    print()
    print("Enter the path to the save folder: ")
    try:
        folder_path = input()

        return save_result(result, folder_path)
    except Exception as e:
        sys.exit(str(e))
