"""Various functions for computing statistics on data."""

import numpy as np
from tqdm import tqdm


def _compute_cumsum(data):
    """
    Compute cumulative sum with prepended zero row.

    Parameters
    ----------
    data : (N, D)

    Returns
    -------
    cumsum : (N+1, D)
    """

    D = data.shape[1]

    return np.concatenate(
        [
            np.zeros((1, D), dtype=data.dtype),
            np.cumsum(data, axis=0),
        ],
        axis=0,
    )


def compute_bin_averages(cumsum: np.ndarray, bin_size: int, overlap: int):
    """
    Compute the averages over bins of size `bin_size`,
    with `overlap` amount of overlap.
    """
    N = cumsum.shape[0] - 1

    # Compute the stride for the sliding window based on the overlap
    stride = max(1, round(bin_size * (1.0 - overlap)))

    # Compute the starting indices of each bin
    starts = np.arange(0, N - bin_size + 1, stride)

    averages = (cumsum[starts + bin_size] - cumsum[starts]) / bin_size

    return averages


def compute_allan_variances(data, periods, measure_rate=10, overlap=0.5):
    """Compute the Allan Variance given the averages map.

    Args:
        periods (np.ndarray): The time periods between period_min and period_max with step 0.1.
        period_max (float): The maximum period time.
        measure_rate (float): The measurement rate of the IMU.
        overlap (float): The overlap between bins.

    Returns:
        List[np.ndarray]: Allan Variances for various time periods
            from `period_min` to `period_max`.
    """
    # Pre-allocate the Allan Variances
    allan_variances = np.empty(periods.shape + (6,))

    # Precompute the cumulative sum for efficient bin average computation
    data_cumsum = _compute_cumsum(data)

    for idx, period_time in tqdm(enumerate(periods), total=len(periods)):
        max_bin_size = int(period_time * measure_rate)
        bin_overlap = int(np.floor(max_bin_size * overlap))

        # Compute the bin averages in the same loop
        # This saves memory and is faster
        averages = compute_bin_averages(data_cumsum, max_bin_size, bin_overlap)

        n = len(averages)

        d = np.sum(np.power(averages[1:] - averages[:-1], 2), axis=0)

        allan_variance = d / (2 * (n - 1))

        allan_variances[idx] = allan_variance

    return allan_variances
