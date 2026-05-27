#!/usr/bin/env python
"""
python scripts/analyze_rosbag.py config/sim.yaml imu_simulation.bag
"""

import argparse

from allan_variance import AllanVariance
from allan_variance.rosbag_reader import ROSBagReader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config .yml file")
    parser.add_argument("rosbag", help="path to ROS bag to analyze")
    return parser.parse_args()


def main():
    """Main runner"""
    args = parse_args()
    av = AllanVariance(args.config, ".")
    bag_reader = ROSBagReader(
        args.rosbag, av.imu_topic(), av.imu_rate(), av.sequence_time(), av.imu_skip_
    )
    data = bag_reader.read()
    av.run(data[:, 1:])


if __name__ == "__main__":
    main()
