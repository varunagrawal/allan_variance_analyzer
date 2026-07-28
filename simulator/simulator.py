"""An IMU simulator based on the C++ version in
https://github.com/ori-drs/allan_variance_ros/blob/master/src/ImuSimulator.cpp
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import jax
import jax.numpy as np
import yaml
from jax import random
from loguru import logger
from tqdm import tqdm

from allan_variance_analyzer.base import FilePath


def RandomNormalDistributionVector(key: jax.Array, sigma: float):
    """Sample a normally distributed vector with variance `sigma`."""
    return random.normal(key, shape=(3,)) * sigma


class ImuSimulator:
    """An IMU simulator for testing and debugging."""

    def __init__(self, config_file: FilePath, output_path: FilePath):
        with open(config_file, "r") as stream:
            self.config_ = yaml.safe_load(stream)

        self.accelerometer_noise_density_: float = self.config_[
            "accelerometer_noise_density"
        ]
        self.accelerometer_random_walk_: float = self.config_[
            "accelerometer_random_walk"
        ]
        self.accelerometer_bias_init_ = self.config_["accelerometer_bias_init"]

        self.gyroscope_noise_density_ = self.config_["gyroscope_noise_density"]
        self.gyroscope_random_walk_ = self.config_["gyroscope_random_walk"]
        self.gyroscope_bias_init_ = self.config_["gyroscope_bias_init"]

        self.rostopic_ = self.config_["rostopic"]

        self.update_rate_ = self.config_["update_rate"]

        self.sequence_time_ = self.config_["sequence_time"]

        self.output_path_ = Path(output_path)

    def run(self):
        """Run simulation"""
        key = random.key(42)

        logger.info("Generating IMU data...")
        dt = 1 / self.update_rate_

        accelerometer_bias = np.full((3,), self.accelerometer_bias_init_)
        gyroscope_bias = np.full((3,), self.gyroscope_bias_init_)
        accelerometer_real = np.zeros(3)
        gyroscope_real = np.zeros(3)

        measurements_file = self.output_path_ / "measurements.csv"

        with open(measurements_file, "w+") as writer:
            start_time = datetime.now(tz=timezone.utc)

            for i in tqdm(range(int(self.sequence_time_ * self.update_rate_))):
                # Reference: https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
                accelerometer_bias += RandomNormalDistributionVector(
                    key, self.accelerometer_random_walk_
                ) * np.sqrt(dt)
                gyroscope_bias += RandomNormalDistributionVector(
                    key, self.gyroscope_random_walk_
                ) * np.sqrt(dt)

                acc_measure = (
                    accelerometer_real
                    + accelerometer_bias
                    + RandomNormalDistributionVector(
                        key, self.accelerometer_noise_density_
                    )
                    / np.sqrt(dt)
                )
                gyro_measure = (
                    gyroscope_real
                    + gyroscope_bias
                    + RandomNormalDistributionVector(key, self.gyroscope_noise_density_)
                    / np.sqrt(dt)
                )

                timestamp = start_time + timedelta(0, i / self.update_rate_)
                ts = timestamp.strftime("%S.%f")

                writer.write(
                    f"{i},{ts},{acc_measure[0]},{acc_measure[1]},{acc_measure[2]},{gyro_measure[0]},{gyro_measure[1]},{gyro_measure[2]}\n"
                )

        logger.info("Finished generating data. ")
        writer.close()
        logger.info(f"Measurements saved to {measurements_file}")


if __name__ == "__main__":
    simulator = ImuSimulator("imu_simulator_config.yaml", ".")
    simulator.run()
