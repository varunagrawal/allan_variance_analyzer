import unittest
from pathlib import Path

import numpy as np
import pytest
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

    def run(self, data):
        """Run Allan Variance Analysis"""
        # Dict from period to averages
        averages_map = {}

        periods = np.arange(self.period_min, self.period_max, step=0.1)

        for period_time in periods:
            current_average = self.compute_bin_averages(data, period_time)
            averages_map[period_time] = current_average


current_dir = Path(__file__).parent.parent.absolute()


def get_config_file():
    """Helper to get a config file for testing"""
    return current_dir / "anymal_c.yaml"


class TestAllanVariance(unittest.TestCase):
    """Tests for the AllanVariance class"""

    def setUp(self):
        self.config_file = get_config_file()
        data = np.loadtxt(current_dir / "measurements.csv", delimiter=",")
        self.measurements = data[:, 2:8]
        # Convert gyro measurements from radians to degrees
        # data.at[:, 3:6].set(jnp.rad2deg(data[:, 3:6]))
        self.measurements[:, 3:6] = np.rad2deg(self.measurements[:, 3:6])

    def test_constructor(self):
        """Test the constructor"""

        av = AllanVariance(self.config_file, ".")

        self.assertEqual(av.config('imu_rate'), 400)
        self.assertEqual(av.config('measure_rate'), 100)

    def test_compute_bin_averages(self):
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
