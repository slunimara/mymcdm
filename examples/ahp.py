"""
Example taken from:
Wikipedia contributors. (2021, March 19). Analytic hierarchy process – leader example.
Wikipedia. https://en.wikipedia.org/wiki/Analytic_hierarchy_process_%E2%80%93_leader_example#cite_note-SYNTHESIZING-9
"""
from mymcdm import decision
from mymcdm.inout import load_data

data, _ = load_data("data/example_cm.json")

a_matrix = data["alternatives"]
w_vector = data["weights"]

data = decision(a_matrix, w_vector, d_method="AHP")

for key in data:
    print(f"{key}:\n{data[key]}")
