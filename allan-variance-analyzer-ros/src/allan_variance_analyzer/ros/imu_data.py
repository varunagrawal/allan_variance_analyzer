"""Helpers for IMU measurements and data"""

import numpy as np


class ImuMeasurement:
    """Class representing an IMU measurement."""

    def __init__(
        self,
        timestamp: float,
        linear_acceleration: np.ndarray,
        angular_velocity: np.ndarray,
    ):
        """Constructor

        Args:
            timestamp (float): Timestamp in nanoseconds.
            linear_acceleration (np.ndarray): The linear acceleration measurement.
            angular_velocity (np.ndarray): The angular velocity meausrement.
        """
        self.ts = timestamp
        self.a = linear_acceleration
        self.w = angular_velocity

    def asarray(self):
        """Return the measurement as a single array"""
        return np.concatenate(([self.ts], self.a, self.w))
