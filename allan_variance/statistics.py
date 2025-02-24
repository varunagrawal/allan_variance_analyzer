"""Various functions for computing statistics on data."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


@partial(jax.jit, static_argnums=(1, 2))
def compute_bin_averages(data: np.ndarray, max_bin_size: int, overlap: int):
    """
    Compute the averages over bins of size `max_bin_size`,
    with `overlap` amount of overlap.
    """
    def f(data, j):
        current_average = jnp.zeros(6)
        for m in range(max_bin_size):
            current_average += data[j + m]

        current_average /= max_bin_size
        return data, current_average
    
    _, averages = jax.lax.scan(f, data, jnp.arange(0, data.shape[0] - max_bin_size, max_bin_size - overlap))
    
    return averages


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
    allan_variances = jnp.empty(periods.shape + (6, ))

    print("computing bin averages")
    for idx, period_time in tqdm(enumerate(periods), total=len(periods)):
        max_bin_size = int(period_time * measure_rate)
        bin_overlap = int(np.floor(max_bin_size * overlap))

        # Compute the bin averages in the same loop
        # This saves memory and is faster
        averages = compute_bin_averages(data, max_bin_size, bin_overlap)
        n = len(averages)

        d = np.sum(np.power(averages[1:] - averages[:-1], 2), axis=0)
        allan_variance = d / (2 * (n - 1))

        allan_variances.at[idx].set(allan_variance)

    return allan_variances
