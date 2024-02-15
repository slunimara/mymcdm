import numpy as np

from mymcdm import decision

types = [True, True, False]
w_vector = np.array([1 / 3, 1 / 3, 1 / 3])
a_matrix = np.array([(7, 60, 5), (9, 65, 6), (5, 70, 9), (4, 80, 3)])

data = decision(
    a_matrix,
    w_vector,
    n_method="VECTOR",
    d_method="TOPSIS",
    criteria_type=types,
)

for key in data:
    print(f"{key}:\n{data[key]}")
