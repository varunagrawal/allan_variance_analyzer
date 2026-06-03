from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# ============================================================================
# Internal utilities
# ============================================================================


def _prepare_signal(data):
    """
    Ensure signal has shape (N, D).
    """

    data = jnp.asarray(data)

    if data.ndim == 1:
        data = data[:, None]

    if data.ndim != 2:
        raise ValueError("Input must have shape (N,) or (N,D).")

    return data


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

    return jnp.concatenate(
        [
            jnp.zeros((1, D), dtype=data.dtype),
            jnp.cumsum(data, axis=0),
        ],
        axis=0,
    )


# ============================================================================
# Single-tau Allan variance kernel
# ============================================================================


@partial(
    jax.jit,
    static_argnames=("bin_size", "overlap"),
)
def _allan_variance_single_tau(
    cumsum,
    bin_size,
    overlap=0.5,
):
    """
    Compute Allan variance for a single tau.

    Parameters
    ----------
    cumsum : (N+1, D)
        Precomputed cumulative sum.

    bin_size : int
        Averaging window size in samples.

    overlap : float
        Overlap fraction in [0,1).

    Returns
    -------
    avar : (D,)
        Allan variance for each channel.
    """

    N = cumsum.shape[0] - 1

    stride = max(
        1,
        round(bin_size * (1.0 - overlap)),
    )

    starts = jnp.arange(
        0,
        N - bin_size + 1,
        stride,
    )

    y = (cumsum[starts + bin_size] - cumsum[starts]) / bin_size

    dy = y[1:] - y[:-1]

    return 0.5 * jnp.mean(dy**2, axis=0)


# ============================================================================
# Public API
# ============================================================================


def compute_allan_deviations(
    data,
    periods,
    sample_rate_hz,
    overlap=0.5,
):
    """
    Compute Allan deviation across specified periods.

    Parameters
    ----------
    data : array-like
        Input signal with shape:
            (N,)
            (N,D)

        Example IMU:
            (N,6)

    periods : array-like
        Averaging times (tau values) in seconds.

    sample_rate_hz : float
        Sensor sampling frequency.

    overlap : float
        Overlap fraction in [0,1).

    Returns
    -------
    adev : jnp.ndarray
        Allan deviations.

        Shape:
            (K,)     for 1D signal
            (K,D)    for multi-axis signal

        where:
            K = len(periods)
    """

    data = _prepare_signal(data)

    squeeze_output = False

    if data.shape[1] == 1:
        squeeze_output = True

    # Precompute cumulative sum ONCE
    cumsum = _compute_cumsum(data)

    # Convert tau values -> bin sizes
    bin_sizes = np.maximum(
        1,
        np.round(np.asarray(periods) * sample_rate_hz).astype(np.int32),
    )

    adevs = []

    for m in bin_sizes:
        avar = _allan_variance_single_tau(
            cumsum,
            bin_size=int(m),
            overlap=overlap,
        )

        adevs.append(jnp.sqrt(avar))

    adevs = jnp.stack(adevs)

    if squeeze_output:
        return adevs[:, 0]

    return adevs
