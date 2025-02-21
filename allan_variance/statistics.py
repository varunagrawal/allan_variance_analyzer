"""Various functions for computing statistics on data."""

import numpy as np


def compute_bin_averages(data: np.ndarray, max_bin_size: int, overlap: int):
    """
    Compute the averages over bins of size `max_bin_size`,
    with `overlap` amount of overlap.
    """
    indices = np.arange(0, data.shape[0] - max_bin_size,
                        max_bin_size - overlap)

    # reduceat adds everything after the last index,
    # so we update the end
    data_ = data[:indices[-1] + max_bin_size]

    current_average = np.add.reduceat(data_, indices=indices, axis=0)

    current_average = current_average / max_bin_size

    return current_average
