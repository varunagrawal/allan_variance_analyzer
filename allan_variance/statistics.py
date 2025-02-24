"""Various functions for computing statistics on data."""

import numpy as np
from tqdm import tqdm


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


def compute_allan_variance(data, periods, measure_rate=10, overlap=0.5):
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
    allan_variances = np.empty(periods.shape + (6, ))

    for idx, period_time in tqdm(enumerate(periods), total=len(periods)):
        max_bin_size = int(period_time * measure_rate)
        bin_overlap = int(np.floor(max_bin_size * overlap))

        # Compute the bin averages in the same loop
        # This saves memory and is faster
        averages = compute_bin_averages(data, max_bin_size, bin_overlap)
        n = len(averages)

        d = np.sum(np.power(averages[1:] - averages[:-1], 2), axis=0)
        allan_variance = d / (2 * (n - 1))

        allan_variances[idx] = allan_variance

    return allan_variances
