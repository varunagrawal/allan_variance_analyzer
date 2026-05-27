"""Tests for the imu_data module"""

import unittest

import numpy as np

from allan_variance.imu_data import ImuMeasurement


class TestImuMeasurement(unittest.TestCase):
    """Tests for the ImuMeasurement class."""

    def setUp(self):
        self.ts = 123456
        self.a = np.asarray([1, 2, 3])
        self.w = np.asarray([0.9, 0.8, 0.7])

    def test_constructor(self):
        """Test the constructor"""
        measurement = ImuMeasurement(self.ts, self.a, self.w)
        np.testing.assert_allclose(measurement.a, self.a)
        np.testing.assert_allclose(measurement.w, self.w)

    def test_asarray(self):
        """Test the asarray method"""
        measurement = ImuMeasurement(self.ts, self.a, self.w)
        np.testing.assert_allclose(
            measurement.asarray(), np.asarray([123456, 1, 2, 3, 0.9, 0.8, 0.7])
        )
