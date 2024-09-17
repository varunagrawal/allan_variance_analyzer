"""Tests for analysis module"""

import unittest
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from allan_variance.analysis import (accelerometer_analysis,
                                     compute_bias_instability,
                                     compute_rate_random_walk,
                                     compute_white_noise_params, get_intercept,
                                     gyroscope_analysis, write_imu_yaml)

current_dir = Path(__file__).parent.absolute()


class TestAnalysis(unittest.TestCase):
    """Tests for functions which analyze the Allan Deviations."""

    def setUp(self):
        data = np.loadtxt(current_dir / "fixtures" / "allan_variance.csv",
                          delimiter=' ',
                          dtype=float,
                          usecols=(0, 1, 2, 3, 4, 5, 6))

        self.period = data[:, 0]
        self.allan_deviations = data[:, 1:]

        self.white_noise_break_point = np.where(self.period == 10)[0][0]

        self.acceleration = self.allan_deviations[:, 0:3]
        self.rotation_rate = self.allan_deviations[:, 3:6]

    def test_compute_white_noise_params(self):
        """Test the compute_white_noise_params function."""
        accel_wn_intercept, accel_fit_wn = compute_white_noise_params(
            self.period, self.acceleration, self.white_noise_break_point)

        expected_accel_white_noise_params = np.asarray([
            0.0025282402102088882, 0.0025615284313200383, 0.002466368970143399
        ])

        # regression
        np.testing.assert_allclose(expected_accel_white_noise_params,
                                   accel_wn_intercept)

        self.assertIsInstance(accel_fit_wn, list)
        for fn in accel_fit_wn:
            self.assertIsInstance(fn, Callable)

        gyro_wn_intercept, gyro_fit_wn = compute_white_noise_params(
            self.period, self.rotation_rate, self.white_noise_break_point)

        expected_gyro_white_noise_params = np.asarray(
            [0.010742013458498411, 0.010936249927095876, 0.010784944419707091])

        # regression
        np.testing.assert_allclose(expected_gyro_white_noise_params,
                                   gyro_wn_intercept)

        self.assertIsInstance(gyro_fit_wn, list)
        for fn in gyro_fit_wn:
            self.assertIsInstance(fn, Callable)

    def test_compute_rate_random_walk(self):
        """Test compute_rate_random_walk"""
        accel_rr_intercept_x, xfit_rr = get_intercept(self.period,
                                                      self.acceleration[:, 0],
                                                      0.5, 3.0)
        accel_rr_intercept_y, yfit_rr = get_intercept(self.period,
                                                      self.acceleration[:, 1],
                                                      0.5, 3.0)
        accel_rr_intercept_z, zfit_rr = get_intercept(self.period,
                                                      self.acceleration[:, 2],
                                                      0.5, 3.0)

        accel_rr_intercept, fit_rr = compute_rate_random_walk(
            self.period, self.acceleration)

        np.testing.assert_allclose(
            np.asarray([
                accel_rr_intercept_x, accel_rr_intercept_y,
                accel_rr_intercept_z
            ]), accel_rr_intercept)

        for idx, fit_fn in enumerate((xfit_rr, yfit_rr, zfit_rr)):
            x = np.arange(1, 100, 0.1)
            np.testing.assert_allclose(fit_fn(x), fit_rr[idx](x))

    def test_compute_bias_instability(self):
        """Test compute_bias_instability"""
        accel_min_x = np.amin(self.acceleration[:, 0])
        accel_min_y = np.amin(self.acceleration[:, 1])
        accel_min_z = np.amin(self.acceleration[:, 2])

        accel_min_x_index = np.argmin(self.acceleration[:, 0])
        accel_min_y_index = np.argmin(self.acceleration[:, 1])
        accel_min_z_index = np.argmin(self.acceleration[:, 2])

        accel_min, accel_argmin = compute_bias_instability(self.acceleration)

        np.testing.assert_allclose(
            np.asarray([accel_min_x, accel_min_y, accel_min_z]), accel_min)
        np.testing.assert_allclose(
            np.asarray(
                [accel_min_x_index, accel_min_y_index, accel_min_z_index]),
            accel_argmin)

    def test_accelerometer_analysis(self):
        """Test accelerometer_analysis function."""
        worst_accel_white_noise, worst_accel_random_walk = accelerometer_analysis(
            self.period,
            self.acceleration,
            self.white_noise_break_point,
            show_plots=False)

        # regression
        self.assertEqual(0.0025615284313200383, worst_accel_white_noise)
        self.assertEqual(9.079971439106243e-05, worst_accel_random_walk)

    def test_gyroscope_analysis(self):
        """Test gyroscope_analysis function."""
        worst_gyro_white_noise, worst_gyro_random_walk = gyroscope_analysis(
            self.period,
            self.rotation_rate,
            self.white_noise_break_point,
            show_plots=False)

        # regression
        self.assertEqual(0.010936249927095876, worst_gyro_white_noise)
        self.assertEqual(0.0002359578060906847, worst_gyro_random_walk)

    def test_write_imu_yaml(self):
        """Test the write_imu_yaml function."""
        worst_accel_white_noise, worst_accel_random_walk = accelerometer_analysis(
            self.period,
            self.acceleration,
            self.white_noise_break_point,
            show_plots=False)
        worst_gyro_white_noise, worst_gyro_random_walk = gyroscope_analysis(
            self.period,
            self.rotation_rate,
            self.white_noise_break_point,
            show_plots=False)

        write_imu_yaml(worst_accel_white_noise, worst_accel_random_walk,
                       worst_gyro_white_noise, worst_gyro_random_walk, 400)

        with open("imu.yaml", 'r') as stream:
            actual_config = yaml.safe_load(stream)

        with open(current_dir / 'fixtures' / "imu.yaml", 'r') as stream:
            expected_config = yaml.safe_load(stream)

        self.assertDictEqual(expected_config, actual_config)
