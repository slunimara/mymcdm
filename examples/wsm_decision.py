import numpy as np
import mymcdm

types = [False, True, True, False]
a_matrix = np.array(
    [
        [6.6, 6, 56, 9499],
        [6.4, 8, 72, 11999],
        [6.67, 8, 56, 11999],
        [6.1, 6, 78, 9990],
        [6.5, 6, 68, 11999],
    ]
)

comparsion_matrix = np.array(
    [[1, 1 / 3, 1 / 5, 1 / 3], [3, 1, 1 / 5, 1 / 3], [5, 5, 1, 3], [3, 3, 1 / 3, 1]]
)

w_vector, cr = mymcdm.weighting.pairwise_comparisons(comparsion_matrix)

# data = mymcdm.decision(
#     a_matrix,
#     w_vector,
#     types,
#     n_method="MAXMIN",
#     d_method="WSM",
# )

data, _ = mymcdm.load_data("data/example_result.json")

for key in data:
    print(f"{key}:\n{data[key]}")
