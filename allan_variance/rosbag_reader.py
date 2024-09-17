"""Module for reading ROS bags"""

from pathlib import Path

import numpy as np
from loguru import logger
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm

from allan_variance import FilePath, ImuMeasurement


class ROSBagReader:
    """Class to read ROS bag data without the need to install ROS."""

    def __init__(self,
                 rosbag_path: FilePath,
                 topic: str,
                 imu_rate: float,
                 sequence_time: float,
                 imu_skip: float,
                 typestore=Stores.ROS1_NOETIC):
        self.rosbag_path = Path(rosbag_path)
        self.topic = topic  #  "/sensors/imu"
        self.imu_rate_ = imu_rate
        self.sequence_time_ = sequence_time
        self.imu_skip_ = imu_skip

        # Create a type store to use if the bag has no message definitions.
        self.typestore = get_typestore(typestore)

    def read(self):
        """Read the ROS bag and get the measurements."""
        imu_buffer = []

        logger.info(f"Loading bag from path: {self.rosbag_path}")

        # Create reader instance and open for reading.
        with AnyReader([self.rosbag_path],
                       default_typestore=self.typestore) as reader:

            messages = reader.messages(connections=reader.connections)

            for counter, message in tqdm(enumerate(messages),
                                         total=reader.message_count):
                connection, timestamp, rawdata = message

                if connection.topic == self.topic:
                    # Subsample IMU measurements
                    if ((counter + 1) % self.imu_skip_ != 0) \
                                or (counter / self.imu_rate_ > self.sequence_time_):
                        continue

                    msg = reader.deserialize(rawdata, connection.msgtype)

                    timestamp = msg.header.stamp
                    ts_ns = timestamp.sec * 1000000000 + timestamp.nanosec

                    w = np.asarray([
                        msg.angular_velocity.x, msg.angular_velocity.y,
                        msg.angular_velocity.z
                    ])
                    a = np.asarray([
                        msg.linear_acceleration.x, msg.linear_acceleration.y,
                        msg.linear_acceleration.z
                    ])
                    imu_buffer.append(ImuMeasurement(ts_ns, a, w).asarray())

        logger.info("Loaded all the data")
        return np.asarray(imu_buffer)
