"""
Mean Weight.

References: [3]
"""
import numpy as np
from numpy.typing import NDArray


def mean_weight(size: int) -> NDArray:
    """Returns vector of size n that contains 1/n values.

    Args:
        size (int): Size of the required vector.
            Takes positive integer bigger than zero.
    """

    if not isinstance(size, int) or size <= 0:
        raise ValueError(
            f"""Argument n should be positive integer bigger
            than zero. Got {size}"""
        )

    return np.full(size, 1 / size)
