"""Base module to Allan Variance Analysis"""

from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from allan_variance_analyzer.analysis import analyze
from allan_variance_analyzer.statistics import compute_allan_variances

FilePath = str | Path


class Config:
    """Class for storing IMU configuration info."""

    def __init__(self, config_file: FilePath):
        with open(config_file, "r") as stream:
            self.config_ = yaml.safe_load(stream)

        self.imu_rate_ = self.config_["imu_rate"]
        self.measure_rate_ = self.config_["measure_rate"]

        self.imu_topic_ = self.config_.get("imu_topic")
        self.sequence_time_ = self.config_.get("sequence_time")
        self.show_plots_ = self.config_.get("show_plots", False)

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

    def imu_skip(self):
        """Get the IMU skip rate."""
        return self.imu_rate_ // self.measure_rate_


class AllanVarianceAnalyzer(Config):
    """Main class to perform Allan Variance Analysis"""

    def __init__(
        self,
        config_file: FilePath,
        output_path: FilePath,
        overlap: int = 0,
        period_min: float = 0.1,
        period_max: float = 1000,
        write_allan_deviations=False,
    ):

        super().__init__(config_file=config_file)

        self.output_path_ = Path(output_path)

        self.overlap_ = overlap

        # Range we will sample from (e.g. 0.1s to 1000s)
        self.period_min, self.period_max = period_min, period_max

        self.write_allan_deviations_ = write_allan_deviations

    def overlap(self):
        """Get the overlap."""
        return self.overlap_

    def __call__(self, data):
        """Run Allan Variance"""
        return self.run(data)

    def write_deviations(self, periods: np.ndarray, allan_deviations: np.ndarray):
        """Helper method to write the Allan Deviations to file."""
        allan_variance_file = self.output_path_ / "allan_variance.csv"

        logger.info(f"Writing Allan Deviations to {allan_variance_file}")
        with open(allan_variance_file, "w+") as av_writer:
            for period, allan_deviation in zip(periods, allan_deviations):
                # Convert to string for writing to file
                allan_deviation_str = " ".join(map(str, allan_deviation.tolist()))
                av_writer.write(f"{period} {allan_deviation_str}\n")

    def run(self, data: np.ndarray):
        """Run Allan Variance Analysis

        Args:
            data (np.ndarray): A Tx6 data array where the first 3
            columns are linear acceleration and the next 3
            are angular velocity.
        """
        # Assuming gyro data is in radians, convert to degrees
        data[:, 3:6] = np.rad2deg(data[:, 3:6])

        periods = np.arange(self.period_min, self.period_max, step=0.1)

        logger.info("Computing Allan Variances")
        allan_variances = compute_allan_variances(
            data, periods, self.measure_rate_, self.overlap_
        )

        allan_deviations = np.sqrt(allan_variances)

        if self.write_allan_deviations_:
            self.write_deviations(periods, allan_deviations)

        analyze(
            periods,
            allan_deviations,
            self.imu_rate_,
            output_path=self.output_path_,
            show_plots=self.show_plots_,
        )
