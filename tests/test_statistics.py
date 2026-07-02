"""Tests for the statistics module."""

import unittest
from pathlib import Path

import numpy as np
from common import SlowAllanVariance, get_config_file

from allan_variance_analyzer.statistics import (
    _compute_cumsum,
    compute_allan_variances,
    compute_bin_averages,
)

current_dir = Path(__file__).parent.absolute()


class TestStatistics(unittest.TestCase):
    """Tests for the statistics module."""

    def setUp(self):
        self.imu_rate = 400
        self.measure_rate = 400
        self.sequence_time = 11000
        self.overlap = 0

        self.config_file = get_config_file()

        data = np.loadtxt(current_dir / "fixtures" / "measurements.csv", delimiter=",")
        self.measurements = data[:, 2:8]
        # Convert gyro measurements from radians to degrees
        # data.at[:, 3:6].set(jnp.rad2deg(data[:, 3:6]))
        self.measurements[:, 3:6] = np.rad2deg(self.measurements[:, 3:6])

        self.slow_av = SlowAllanVariance(self.config_file, ".")

    def test_compute_bin_averages(self):
        """Test the compute_bin_averages method"""

        max_bin_size = int(0.1 * self.measure_rate)
        overlap = int(np.floor(max_bin_size * self.overlap))

        measurements_cumsum = _compute_cumsum(self.measurements)

        actual_averages = compute_bin_averages(
            measurements_cumsum, max_bin_size, overlap
        )

        expected_averages = self.slow_av.compute_bin_averages(self.measurements, 0.1)

        N = expected_averages.shape[0]

        np.testing.assert_allclose(expected_averages, actual_averages[:N])

    def test_compute_averages(self):
        """Test to verify that the computed averages in each bin are correct."""

        measurements_cumsum = _compute_cumsum(self.measurements)
        for period_time in np.arange(0.1, 2, step=0.1):
            max_bin_size = int(period_time * self.measure_rate)
            overlap = int(np.floor(max_bin_size * self.overlap))
            actual_average = compute_bin_averages(
                measurements_cumsum, max_bin_size, overlap
            )
            expected_average = self.slow_av.compute_bin_averages(
                self.measurements, period_time
            )
            # print(f"Period time: {period_time}")
            # print(f"{expected_average.shape=}, {actual_average.shape=}")
            N = expected_average.shape[0]
            np.testing.assert_allclose(expected_average, actual_average[:N])

    def test_compute_allan_variances(self):
        """Test compute_allan_variances method."""

        period_min = 0.1
        period_max = 2
        periods = np.arange(period_min, period_max, step=0.1)

        averages_map = {}
        measurements_cumsum = _compute_cumsum(self.measurements)
        for period_time in periods:
            max_bin_size = int(period_time * self.measure_rate)
            overlap = int(np.floor(max_bin_size * self.overlap))
            current_average = compute_bin_averages(
                measurements_cumsum, max_bin_size, overlap
            )
            averages_map[period_time] = current_average

        actual_allan_variances = compute_allan_variances(
            self.measurements,
            periods=periods,
            measure_rate=self.measure_rate,
            overlap=self.overlap,
        )
        expected_allan_variances = self.slow_av.compute_allan_variances(
            averages_map=averages_map, periods=periods
        )

        actual_allan_variances = np.asarray(actual_allan_variances)
        _, expected_allan_variances = zip(*expected_allan_variances)
        expected_allan_variances = np.asarray(expected_allan_variances)

        np.testing.assert_allclose(expected_allan_variances, actual_allan_variances)
