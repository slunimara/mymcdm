import numpy as np
import mymcdm

type = [False, True, True, False]
a_matrix = np.array(
    [
        [6.6, 6, 56, 9499],
        [6.4, 8, 72, 11999],
        [6.67, 8, 56, 11999],
        [6.1, 6, 78, 9990],
        [6.5, 6, 68, 11999],
    ]
)

normalization_matrix, types = mymcdm.normalization.max_min(a_matrix, type)

print(normalization_matrix, types, sep="\n")

comparsion_matrix = np.array(
    [[1, 1 / 3, 1 / 5, 1 / 3], [3, 1, 1 / 5, 1 / 3], [5, 5, 1, 3], [3, 3, 1 / 3, 1]]
)

w_vector, cr = mymcdm.weighting.pairwise_comparisons(comparsion_matrix)

print(w_vector, cr, sep="\n")

a_dataframe = mymcdm.utils.frame_alternatives(normalization_matrix)

result = mymcdm.wsm(a_dataframe, w_vector)

w_series = mymcdm.utils.frame_criterions(w_vector)
decision_matrix = mymcdm.utils.framing.make_decision_matrix(a_dataframe, w_series)

print(decision_matrix, result, sep="\n")
