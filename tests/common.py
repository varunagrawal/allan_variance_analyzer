"""Common code used across multiple test files."""

from pathlib import Path

import numpy as np
import yaml

current_dir = Path(__file__).parent.absolute()


def get_config_file():
    """Helper to get a config file for testing"""
    return current_dir.parent.absolute() / "config" / "sim.yaml"


class SlowAllanVariance:
    """A naive implementation of the Allan Variance computations."""

    def __init__(
        self,
        config_file,
        output_path,
        overlap: int = 0,
        period_min=0.1,
        period_max=1000,
    ):
        with open(config_file, "r") as stream:
            self.config_ = yaml.safe_load(stream)

        self.first_msg_ = True
        self.imu_topic_ = self.config_["imu_topic"]
        self.imu_rate_ = self.config_["imu_rate"]
        self.measure_rate_ = self.config_["measure_rate"]
        self.sequence_time_ = self.config_["sequence_time"]

        self.imu_skip_ = self.imu_rate_ // self.measure_rate_

        self.allan_variance_file_ = Path(output_path) / "allan_variance.csv"

        self.overlap_ = overlap

        # Range we will sample from (0.1s to 1000s)
        self.period_min, self.period_max = period_min, period_max

    def compute_bin_averages(self, data, period_time):
        """Compute the averages over each bin"""
        max_bin_size = int(period_time * self.measure_rate_)
        overlap = int(np.floor(max_bin_size * self.overlap_))

        averages = []

        for j in range(0, data.shape[0] - max_bin_size, max_bin_size - overlap):
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

    def compute_allan_variances(self, averages_map, periods):
        """Compute the Allan Variance across different time periods given the averages map."""
        allan_variances = []
        for period_time in periods:
            averages = averages_map[period_time]
            num_averages = len(averages)

            allan_variance = np.zeros(6)
            for k in range(num_averages - 1):
                allan_variance[0] += np.power(averages[k + 1, 0] - averages[k, 0], 2)
                allan_variance[1] += np.power(averages[k + 1, 1] - averages[k, 1], 2)
                allan_variance[2] += np.power(averages[k + 1, 2] - averages[k, 2], 2)
                allan_variance[3] += np.power(averages[k + 1, 3] - averages[k, 3], 2)
                allan_variance[4] += np.power(averages[k + 1, 4] - averages[k, 4], 2)
                allan_variance[5] += np.power(averages[k + 1, 5] - averages[k, 5], 2)

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

        allan_variances = self.compute_allan_variances(
            averages_map=averages_map, periods=periods
        )

        return allan_variances
