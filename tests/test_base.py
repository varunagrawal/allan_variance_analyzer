"""Tests for base module."""

import unittest
from pathlib import Path

import numpy as np
import yaml

from allan_variance import AllanVariance


class SlowAllanVariance:
    """A naive implementation of the Allan Variance computations."""

    def __init__(self,
                 config_file,
                 output_path,
                 overlap: int = 0,
                 period_min=0.1,
                 period_max=1000):
        with open(config_file, 'r') as stream:
            self.config_ = yaml.safe_load(stream)

        self.first_msg_ = True
        self.imu_topic_ = self.config_['imu_topic']
        self.imu_rate_ = self.config_['imu_rate']
        self.measure_rate_ = self.config_['measure_rate']
        self.sequence_time_ = self.config_['sequence_time']

        self.imu_skip_ = self.imu_rate_ // self.measure_rate_

        self.imu_output_file_ = Path(output_path) / "allan_variance.csv"

        self.overlap_ = overlap

        # Range we will sample from (0.1s to 1000s)
        self.period_min, self.period_max = period_min, period_max

    def compute_bin_averages(self, data, period_time):
        """Compute the averages over each bin"""
        max_bin_size = int(period_time * self.measure_rate_)
        overlap = int(np.floor(max_bin_size * self.overlap_))

        averages = []

        for j in range(0, data.shape[0] - max_bin_size,
                       max_bin_size - overlap):

            current_average = np.zeros(6)
            for m in range(max_bin_size):
                current_average[0] += data[j + m, 0]
                current_average[1] += data[j + m, 1]
                current_average[2] += data[j + m, 2]

                current_average[3] += data[j + m, 3]
                current_average[4] += data[j + m, 4]
                current_average[5] += data[j + m, 5]

            current_average[0] /= max_bin_size
            current_average[1] /= max_bin_size
            current_average[2] /= max_bin_size
            current_average[3] /= max_bin_size
            current_average[4] /= max_bin_size
            current_average[5] /= max_bin_size

            averages.append(current_average)

        return np.asarray(averages)

    def compute_allan_variance(self, averages_map, periods):
        """Test compute_allan_variance method"""
        allan_variances = []
        for period_time in periods:
            averages = averages_map[period_time]
            num_averages = len(averages)

            allan_variance = np.zeros(6)
            for k in range(num_averages - 1):
                allan_variance[0] += np.power(
                    averages[k + 1, 0] - averages[k, 0], 2)
                allan_variance[1] += np.power(
                    averages[k + 1, 1] - averages[k, 1], 2)
                allan_variance[2] += np.power(
                    averages[k + 1, 2] - averages[k, 2], 2)
                allan_variance[3] += np.power(
                    averages[k + 1, 3] - averages[k, 3], 2)
                allan_variance[4] += np.power(
                    averages[k + 1, 4] - averages[k, 4], 2)
                allan_variance[5] += np.power(
                    averages[k + 1, 5] - averages[k, 5], 2)

            avar = allan_variance / (2 * (num_averages - 1))

            allan_variances.append((period_time, avar))

        return allan_variances

    def run(self, data):
        """Run Allan Variance Analysis"""
        # Dict from period to averages
        averages_map = {}

        periods = np.arange(self.period_min, self.period_max, step=0.1)

        for period_time in periods:
            current_average = self.compute_bin_averages(data, period_time)
            averages_map[period_time] = current_average

        allan_variances = self.compute_allan_variance(
            averages_map=averages_map, periods=periods)

        return allan_variances


current_dir = Path(__file__).parent.absolute()


def get_config_file():
    """Helper to get a config file for testing"""
    return current_dir.parent.absolute() / "config" / "sim.yaml"


class TestAllanVariance(unittest.TestCase):
    """Tests for the AllanVariance class"""

    def setUp(self):
        self.config_file = get_config_file()
        data = np.loadtxt(current_dir / "fixtures" / "measurements.csv",
                          delimiter=",")
        self.measurements = data[:, 2:8]
        # Convert gyro measurements from radians to degrees
        # data.at[:, 3:6].set(jnp.rad2deg(data[:, 3:6]))
        self.measurements[:, 3:6] = np.rad2deg(self.measurements[:, 3:6])

    def test_constructor(self):
        """Test the constructor"""

        av = AllanVariance(self.config_file, ".")

        self.assertEqual(av.config('imu_rate'), 400)
        self.assertEqual(av.config('measure_rate'), 400)

    def test_compute_bin_averages(self):
        """Test the compute_bin_averages method"""
        av = AllanVariance(self.config_file, ".")
        actual_average = av.compute_bin_averages(self.measurements, 0.1)
        slow_av = SlowAllanVariance(self.config_file, ".")
        expected_average = slow_av.compute_bin_averages(self.measurements, 0.1)
        np.testing.assert_allclose(expected_average, actual_average)

        for period_time in np.arange(0.1, 2, step=0.1):
            actual_average = av.compute_bin_averages(self.measurements,
                                                     period_time)
            expected_average = slow_av.compute_bin_averages(
                self.measurements, period_time)
            np.testing.assert_allclose(expected_average, actual_average)

    def test_compute_allan_variance(self):
        """Test compute_allan_variance method."""
        av = AllanVariance(self.config_file, ".")
        slow_av = SlowAllanVariance(self.config_file, ".")

        period_min = 0.1
        period_max = 2
        periods = np.arange(period_min, period_max, step=0.1)

        averages_map = {}
        for period_time in periods:
            current_average = av.compute_bin_averages(self.measurements,
                                                      period_time)
            averages_map[period_time] = current_average

        actual_allan_variances = av.compute_allan_variance(self.measurements,
                                                           periods=periods)
        expected_allan_variances = slow_av.compute_allan_variance(
            averages_map=averages_map, periods=periods)

        actual_allan_variances = np.asarray(actual_allan_variances)
        _, expected_allan_variances = zip(*expected_allan_variances)
        expected_allan_variances = np.asarray(expected_allan_variances)

        np.testing.assert_allclose(expected_allan_variances,
                                   actual_allan_variances)
