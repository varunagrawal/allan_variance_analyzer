"""Base module to Allan Variance Analysis"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

from allan_variance.analysis import analyze

FilePath = Union[str, Path]


class Config:
    """Class for storing IMU configuration info."""

    def __init__(self, config_file: FilePath):
        with open(config_file, 'r') as stream:
            self.config_ = yaml.safe_load(stream)

        self.imu_topic_ = self.config_['imu_topic']
        self.imu_rate_ = self.config_['imu_rate']
        self.measure_rate_ = self.config_['measure_rate']
        self.sequence_time_ = self.config_['sequence_time']
        print(self.config_)

        self.imu_skip_ = self.imu_rate_ // self.measure_rate_

    def config(self, key: str = ""):
        """Getter for the config."""
        if key:
            return self.config_[key]
        else:
            return self.config_

    def imu_topic(self):
        """Get the IMU topic."""
        return self.imu_topic_

    def imu_rate(self):
        """Get the IMU rate."""
        return self.imu_rate_

    def measure_rate(self):
        """Get the IMU measurement rate."""
        return self.measure_rate_

    def sequence_time(self):
        """Get the total sequence time."""
        return self.sequence_time_


class AllanVariance(Config):
    """Main class to perform Allan Variance Analysis"""

    def __init__(self,
                 config_file: FilePath,
                 output_path: FilePath,
                 overlap: int = 0,
                 period_min: float = 0.1,
                 period_max: float = 1000,
                 write_allan_deviations=False):

        super().__init__(config_file=config_file)

        self.imu_output_file_ = Path(output_path) / "allan_variance.csv"

        self.overlap_ = overlap

        # Range we will sample from (e.g. 0.1s to 1000s)
        self.period_min, self.period_max = period_min, period_max

        self.write_allan_deviations_ = write_allan_deviations

    def __call__(self, data):
        """Run Allan Variance"""
        return self.run(data)

    def compute_bin_averages(self, data: np.ndarray, period_time: float):
        """
        Compute the averages over bins of size `period_time`*`measure_rate`.
        """
        data = np.array(data)

        max_bin_size = int(period_time * self.measure_rate_)
        overlap = int(np.floor(max_bin_size * self.overlap_))

        indices = np.arange(0, data.shape[0] - max_bin_size,
                            max_bin_size - overlap)

        # reduceat adds everything after the last index,
        # so we update the end
        data = data[:indices[-1] + max_bin_size]

        current_average = np.add.reduceat(data, indices=indices, axis=0)

        current_average = current_average / max_bin_size

        return current_average

    def compute_allan_variance(self, averages_map: Dict[float, np.ndarray],
                               periods: np.ndarray):
        """Compute the Allan Variance given the averages map.

        Args:
            averages_map (Dict[double, list]): Map from period to averages.
            period_min (float): The minimum period time.
            period_max (float): The maximum period time.

        Returns:
            List[np.ndarray]: Allan Variances for various time periods
                from `period_min` to `period_max`.
        """
        logger.info("Computing Allan Variances")

        # Pre-allocate the Allan Variances
        allan_variances = np.empty(periods.shape + (6, ))

        for idx, period_time in enumerate(periods):
            averages = averages_map[period_time]
            n = len(averages)

            d = np.sum(np.power(averages[1:] - averages[:-1], 2), axis=0)
            allan_variance = d / (2 * (n - 1))

            allan_variances[idx] = allan_variance

        return allan_variances

    def write_deviations(self, periods: np.ndarray,
                         allan_deviations: np.ndarray):
        """Helper method to write the Allan Deviations to file."""
        logger.info("Writing Allan Deviations to allan_variance.csv")
        with open("allan_variance.csv", 'w+') as av_writer:
            for period, allan_deviation in zip(periods, allan_deviations):
                allan_deviation_str = " ".join(allan_deviation.tolist())
                av_writer.write(f"{period} {allan_deviation_str}\n")

    def run(self, data: np.ndarray):
        """Run Allan Variance Analysis"""
        # Assuming gyro data is in radians, convert to degrees
        data[:, 3:6] = np.rad2deg(data[:, 3:6])

        # Dict from period to averages
        averages_map = {}

        periods = np.arange(self.period_min, self.period_max, step=0.1)

        for period_time in tqdm(periods):
            current_average = self.compute_bin_averages(data, period_time)

            averages_map[period_time] = current_average

        allan_variances = self.compute_allan_variance(
            averages_map=averages_map, periods=periods)

        allan_deviations = np.sqrt(allan_variances)

        if self.write_allan_deviations_:
            self.write_deviations(periods, allan_deviations)

        analyze(periods, allan_deviations, self.imu_rate_)
