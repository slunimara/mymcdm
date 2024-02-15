"""
Example taken from:
Wikipedia contributors. (2021, March 19). Analytic hierarchy process – leader example.
Wikipedia. https://en.wikipedia.org/wiki/Analytic_hierarchy_process_%E2%80%93_leader_example#cite_note-SYNTHESIZING-9
"""
from mymcdm import ahp_cm
from mymcdm.utils import replace_fractions
from mymcdm.inout import read_JSON

data = read_JSON("data/example_cm.json")

comparsion_matrices = []
for matrix in data["alternatives"]:
    matrix = replace_fractions(matrix)
    comparsion_matrices.append(matrix)

criteria = replace_fractions(data["criteria"])
score, cr = ahp_cm(comparsion_matrices, criteria)

print(score, f"Pairwise comparison decisions consistent. - {cr}", sep="\n")
