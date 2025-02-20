"""
Script to help analyze allan_variances.csv file.

python analyze_ava.py allan_variance.csv

E.g. python analyze_ava.py ../tests/fixtures/allan_variance.csv
"""

import argparse

import numpy as np

from allan_variance.analysis import analyze


def parse_arguments():
    """Parse command line arguments."""
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


expected_results = """
ACCELEROMETER:
X Velocity Random Walk:  0.00253 m/s/sqrt(s)  0.15169 m/s/sqrt(hr)
Y Velocity Random Walk:  0.00256 m/s/sqrt(s)  0.15369 m/s/sqrt(hr)
Z Velocity Random Walk:  0.00247 m/s/sqrt(s)  0.14798 m/s/sqrt(hr)
X Bias Instability:  0.00039 m/s^2  5020.65605 m/hr^2
Y Bias Instability:  0.00041 m/s^2  5357.70418 m/hr^2
Z Bias Instability:  0.00043 m/s^2  5580.34531 m/hr^2
X Accel Random Walk:  0.00009 m/s^2/sqrt(s)
Y Accel Random Walk:  0.00009 m/s^2/sqrt(s)
Z Accel Random Walk:  0.00006 m/s^2/sqrt(s)
GYROSCOPE:
X Angle Random Walk:  0.01074 deg/sqrt(s)  0.64452 deg/sqrt(hr)
Y Angle Random Walk:  0.01094 deg/sqrt(s)  0.65617 deg/sqrt(hr)
Z Angle Random Walk:  0.01078 deg/sqrt(s)  0.64710 deg/sqrt(hr)
X Bias Instability:  0.00115 deg/s  4.13739 deg/hr
Y Bias Instability:  0.00124 deg/s  4.45915 deg/hr
Z Bias Instability:  0.00128 deg/s  4.60338 deg/hr
X Rate Random Walk:  0.00017 deg/s/sqrt(s)
Y Rate Random Walk:  0.00018 deg/s/sqrt(s)
Z Rate Random Walk:  0.00024 deg/s/sqrt(s)
"""