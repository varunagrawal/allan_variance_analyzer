"""
Script to help analyze allan_variances.csv file.

python analyze_ava.py allan_variance_ros.csv
"""

import argparse

import numpy as np

from allan_variance.analysis import analyze


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--imu_rate', default=400, type=int)
    return parser.parse_args()


args = parse_arguments()
data = np.loadtxt(args.csv,
                  delimiter=' ',
                  dtype=float,
                  usecols=(0, 1, 2, 3, 4, 5, 6))

periods = data[:, 0].astype(float)
allan_deviations = data[:, 1:7].astype(float)

analyze(periods, allan_deviations, args.imu_rate)
