import numpy as np

from mymcdm import decision

types = [True, True, False]
w_vector = np.array([1 / 3, 1 / 3, 1 / 3])
a_matrix = np.array([(100, 35, 55), (115, 34, 65), (90, 40, 52), (110, 36, 60)])

data = decision(
    a_matrix, w_vector, n_method="VECTOR", d_method="ELECTRE", criteria_type=types
)

for key in data:
    print(f"{key}:\n{data[key]}")
