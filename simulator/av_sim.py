"""Run AVA on simulator data"""

import numpy as np

from allan_variance import AllanVariance

if __name__ == "__main__":
    av = AllanVariance(config_file="./anymal_c.yaml", output_path=".")

    data = np.loadtxt("measurements.csv", delimiter=",")
    measurements = data[:, 2:8]
    allan_variances = av(measurements)
