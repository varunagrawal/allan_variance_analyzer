"""Module for performing Allan Variance Analysis with ROS1/ROS2 bags."""

from pathlib import Path

from allan_variance_analyzer import AllanVarianceAnalyzer

FilePath = str | Path


class AllanVarianceAnalyzerROS(AllanVarianceAnalyzer):
    """Main class to perform Allan Variance Analysis"""

    def __init__(self, config_file: FilePath, output_path: FilePath) -> None:
        super().__init__(config_file=config_file, output_path=output_path)

    def load_imu_buffer(self, data):
        """Load IMU buffer from the ROS bag."""
        imu_counter = 0

        for tNanoSecs in data:
            imu_counter += 1

            # Subsample IMU measurements
            if (imu_counter % self.imu_skip_ != 0) or (
                imu_counter / self.imu_rate_ > self.sequence_time_
            ):
                continue
