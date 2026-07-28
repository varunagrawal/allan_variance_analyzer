#!/usr/bin/env python
"""
Script to parse a ROS bag and compute IMU calibration parameters.
"""

import argparse
from typing import Iterable

import numpy as np
import ros_numpy
import rosbag
import rospy
from tqdm import tqdm

from allan_variance_analyzer import AllanVarianceAnalyzer
from allan_variance_analyzer.imu_data import ImuMeasurement


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_file", help="Path to the rosbag file.")
    parser.add_argument("config_file", help="Path to the IMU config file.")
    return parser.parse_args()


class BagReader:

    def __init__(self, bag_file: str, config_file: str):
        rospy.init_node('bag_reader', anonymous=True)

        av = AllanVarianceAnalyzer(config_file=config_file, output_path=".")
        topics = (av.imu_topic(), )

        bag = rosbag.Bag(bag_file)
        rospy.loginfo("Reading the bag!")

        num_of_messages = bag.get_message_count(topics)

        imu_measurements = []

        count = 0
        print(bag.get_type_and_topic_info(topic_filters=topics).topics)
        for i, (topic, msg,
                _) in tqdm(enumerate(bag.read_messages(topics=topics))):
            # print(topic, msg)
            # imu_measurement = ImuMeasurement(
            #     msg.header.stamp.nsecs,
            #     linear_acceleration=msg.linear_acceleration,
            #     angular_velocity=msg.angular_velocity)
            # imu_measurements.append(imu_measurement.asarray()[1:])
            # print(msg.angular_velocity)
            print(msg)
            if count > 3:
                break
            count += 1
        bag.close()

        # imu_measurements = np.asarray(imu_measurements)
        # print(f"Collected {imu_measurements.shape[0]} measurements")
        # av(imu_measurements)

        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        """Post-processing"""
        print("SHUTTING DOWN!")


if __name__ == "__main__":
    args = parse_args()
    # BagReader(args.bag_file, args.config_file, topics=('/state_estimator/imu'))
    BagReader(args.bag_file, args.config_file)
