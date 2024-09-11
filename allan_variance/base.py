"""Base module to Allan Variance Analysis"""

from pathlib import Path
from typing import Union

import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

from allan_variance.analysis import analyze

FilePath = Union[str, Path]


class AllanVariance:
    """Main class to perform Allan Variance Analysis"""

    def __init__(self,
                 config_file: FilePath,
                 output_path: FilePath,
                 overlap: int = 0,
                 period_min: float = 0.1,
                 period_max: float = 1000):
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

        self.dt = 0.1

        # Range we will sample from (e.g. 0.1s to 1000s)
        self.period_min, self.period_max = period_min, period_max

    def imu_topic(self):
        """Get the IMU topic."""
        return self.imu_topic_

    def imu_rate(self):
        """Get the IMU rate."""
        return self.imu_rate_

    def sequence_time(self):
        """Get the total sequence time."""
        return self.sequence_time_

    def config(self, key: str = ""):
        """Getter for the config."""
        if key:
            return self.config_[key]
        else:
            return self.config_

    def __call__(self, data):
        """Run Allan Variance"""
        return self.run(data)

    def compute_bin_averages(self, data, period_time):
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

    def compute_allan_variance(self, averages_map, periods: np.ndarray):
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
        allan_variances = np.empty(periods.shape)

        for idx, period_time in enumerate(periods):
            averages = averages_map[period_time]
            n = len(averages)

            d = np.sum(np.power(averages[1:] - averages[:-1], 2), axis=0)
            allan_variance = d / (2 * (n - 1))

            allan_variances[idx] = allan_variance

        return allan_variances

    def run(self, data):
        """Run Allan Variance Analysis"""
        # Assuming gyro data is in radians, convert to degrees
        data[:, 3:6] = np.rad2deg(data[:, 3:6])

        # Dict from period to averages
        averages_map = {}

        periods = np.arange(self.period_min, self.period_max, step=self.dt)

        for period_time in tqdm(periods):
            current_average = self.compute_bin_averages(data, period_time)

            averages_map[period_time] = current_average

        allan_variances = self.compute_allan_variance(
            averages_map=averages_map, periods=periods)

        logger.info("Writing Allan Deviations to allan_variance.csv")
        with open("allan_variance.csv", 'w+') as av_writer:
            for period, av in zip(periods, allan_variances):
                allan_deviation = np.sqrt(av)
                av_writer.write(
                    f"{period} {allan_deviation[0]} {allan_deviation[1]} {allan_deviation[2]} {allan_deviation[3]} {allan_deviation[4]} {allan_deviation[5]} \n"
                )

        allan_deviations = np.sqrt(np.asarray(allan_variances))

        analyze(periods, allan_deviations)
