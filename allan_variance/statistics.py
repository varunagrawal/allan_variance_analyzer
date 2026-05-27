"""Various functions for computing statistics on data."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


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

    _, averages = jax.lax.scan(
        f, data,
        jnp.arange(0, data.shape[0] - max_bin_size, max_bin_size - overlap))

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

    #TODO: This doesn't seem to work. I will need to rethink the full implementation.

    def f(data, period_time):
        max_bin_size = (period_time * measure_rate).astype(int)
        bin_overlap = (jnp.floor(max_bin_size * overlap)).astype(int)

        averages = compute_bin_averages(data, max_bin_size, bin_overlap)
        n = len(averages)

        d = jnp.sum(jnp.power(averages[1:] - averages[:-1], 2), axis=0)
        allan_variance = d / (2 * (n - 1))

        return allan_variance

    # Vectorize the function over periods
    f_vectorized = jax.vmap(f, in_axes=(None, 0))
    _, allan_variances = f_vectorized(data, periods)

    return allan_variances
