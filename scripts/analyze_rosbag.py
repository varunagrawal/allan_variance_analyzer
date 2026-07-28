#!/usr/bin/env python
"""
python scripts/analyze_rosbag.py config/sim.yaml imu_simulation.bag
"""

import argparse

from allan_variance_analyzer.ros.bag_reader import ROSBagReader

from allan_variance_analyzer import AllanVarianceAnalyzer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config .yml file")
    parser.add_argument("rosbag", help="path to ROS bag to analyze")
    return parser.parse_args()


def main():
    """Main runner"""
    args = parse_args()
    av = AllanVarianceAnalyzer(args.config, "scripts")
    bag_reader = ROSBagReader(
        args.rosbag, av.imu_topic(), av.imu_rate(), av.sequence_time(), av.imu_skip()
    )
    # Data is Tx7 with the first column being the timestamp and the next 6 columns being the IMU data.
    data = bag_reader.read()

    av.run(data[:, 1:])


if __name__ == "__main__":
    main()
