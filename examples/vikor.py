import numpy as np

from mymcdm import decision

types = [True, True, False]
w_vector = np.array([0.3, 0.4, 0.3])
a_matrix = np.array([(5, 8, 4), (7, 7, 8), (8, 8, 6), (7, 4, 6), (1, 2, 3)])

data = decision(
    a_matrix,
    w_vector,
    n_method="LOG",
    d_method="VIKOR",
    criteria_type=types,
)

for key in data:
    print(f"{key}:\n{data[key]}")
