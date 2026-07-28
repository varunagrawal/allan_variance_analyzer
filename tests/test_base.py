"""Tests for base module."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from allan_variance_analyzer import AllanVarianceAnalyzer
from common import get_config_file

current_dir = Path(__file__).parent.absolute()


class TestAllanVariance(unittest.TestCase):
    """Tests for the AllanVariance class"""

    def setUp(self):
        self.config_file = get_config_file()
        data = np.loadtxt(current_dir / "fixtures" / "measurements.csv", delimiter=",")
        self.measurements = data[:, 2:8]
        # Convert gyro measurements from radians to degrees
        # data.at[:, 3:6].set(jnp.rad2deg(data[:, 3:6]))
        self.measurements[:, 3:6] = np.rad2deg(self.measurements[:, 3:6])

    def test_constructor(self):
        """Test the constructor"""

        av = AllanVarianceAnalyzer(self.config_file, ".")

        self.assertEqual(av.config("imu_rate"), 400)
        self.assertEqual(av.config("measure_rate"), 400)

    def test_run(self):
        """Test the run method."""
        #  Create a secure temporary directory
        tmp_dir_object = tempfile.TemporaryDirectory()

        # Convert the string path to a Path object for clean syntax
        tmp_path = Path(tmp_dir_object.name)

        av = AllanVarianceAnalyzer(
            self.config_file,
            tmp_path,
            write_allan_deviations=True,
            period_min=0.1,
            period_max=15,
        )

        av.run(self.measurements)

        # Assert that Allan Deviations were written to file
        self.assertTrue((tmp_path / "allan_variance.csv").exists())
