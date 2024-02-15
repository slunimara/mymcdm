import numpy as np

from mymcdm import decision

types = [True, False, True, True]
w_vector = np.array([0.4, 0.1, 0.25, 0.25])
a_matrix = np.array(
    [
        (9, 9000, 82, 1),
        (8, 8500, 80, 3),
        (8, 9500, 53, 2),
        (8, 15000, 75, 1),
        (10, 11000, 42, 3),
    ]
)

data = decision(
    a_matrix,
    w_vector,
    n_method="MAX",
    d_method="WPM",
    criteria_type=types,
)

for key in data:
    print(f"{key}:\n{data[key]}")
