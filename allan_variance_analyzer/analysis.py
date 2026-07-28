"""
Functions to perform analysis of the Allan Deviations to get the IMU parameters.
"""

from collections.abc import Callable

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def line_func(x, m, b):
    """The line function y = mx + b"""
    return m * x + b


def get_intercept(x, y, m, b):
    """Get the x and y intercepts of the line"""
    logx = np.log(x)
    logy = np.log(y)
    # pylint: disable=unbalanced-tuple-unpacking
    coeffs, _ = curve_fit(
        line_func, logx, logy, bounds=([m, -np.inf], [m + 0.001, np.inf])
    )
    poly = np.poly1d(coeffs)

    def yfit(x):
        return np.exp(poly(np.log(x)))

    return yfit(b), yfit


def generate_prediction(
    tau, q_quantization=0, q_white=0, q_bias_instability=0, q_walk=0, q_ramp=0
):
    """
    Given fitted IMU noise model, generate measurement prediction at `tau`.
    """
    n = len(tau)

    A = np.empty((n, 5))
    A[:, 0] = 3 / tau**2
    A[:, 1] = 1 / tau
    A[:, 2] = 2 * np.log(2) / np.pi
    A[:, 3] = tau / 3
    A[:, 4] = tau**2 / 2

    params = np.array(
        [q_quantization**2, q_white**2, q_bias_instability**2, q_walk**2, q_ramp**2]
    )

    return np.sqrt(A.dot(params))


def compute_white_noise_params(period, measurements, white_noise_break_point):
    """
    Compute the ARW/VRW generated from the white noise component.
    Found by fitting a line with gradient=-0.5 at intercept t=1
    """
    wn_intercept = np.zeros(3)
    wn_fit_fn = [None] * 3
    for idx, _ in enumerate("xyz"):
        wn_intercept[idx], wn_fit_fn[idx] = get_intercept(
            period[0:white_noise_break_point],
            measurements[0:white_noise_break_point, idx],
            -0.5,
            1.0,
        )

    return wn_intercept, wn_fit_fn


def compute_rate_random_walk(period, measurements):
    """
    Compute the Rate Random Walk for the given measurements.
    This corresponds to the y-intercept of the line with gradient=0.5 and x-intercept=3.0.
    """
    rr_intercept = np.zeros(3)
    rr_fit_fn = [None] * 3
    for idx, _ in enumerate("xyz"):
        axis_measurements = measurements[:, idx]
        rr_intercept[idx], rr_fit_fn[idx] = get_intercept(
            period, axis_measurements, 0.5, 3.0
        )

    return rr_intercept, rr_fit_fn


def compute_bias_instability(measurement: np.ndarray):
    """
    Compute the bias instability values.
    These are the bias drift standard deviations.
    """
    measurement_min = np.amin(measurement, axis=0)
    measurement_argmin = np.argmin(measurement, axis=0)
    return measurement_min, measurement_argmin


def plot_loglog(
    period: np.ndarray,
    measurements: np.ndarray,
    fit_wn: list[Callable],
    fit_rr: list[Callable],
    wn_intercept: list[float],
    rr_intercept: list[float],
    measurement_min: list[float],
    measurement_min_index: list[int],
    average_white_noise: float,
    average_bias_instability: float,
    average_random_walk: float,
    measurement_type: str,
    sensor_type: str,
    units: str,
    dpi=90,
    figsize=(16, 9),
):
    """Plot Allan Deviations on LogLog scale.

    Args:
        period (np.ndarray): _description_
        measurements (np.ndarray): _description_
        fit_wn (list[Callable]): _description_
        fit_rr (list[Callable]): _description_
        wn_intercept (list[float]): _description_
        rr_intercept (list[float]): _description_
        measurement_min (list[float]): _description_
        measurement_min_index (list[int]): _description_
        average_white_noise (float): _description_
        average_bias_instability (float): _description_
        average_random_walk (float): _description_
        measurement_type (str): _description_
        sensor_type (str): _description_
        units (str): _description_
        dpi (int, optional): _description_. Defaults to 90.
        figsize (tuple, optional): _description_. Defaults to (16, 9).
    """

    fig = plt.figure(num=measurement_type, dpi=dpi, figsize=figsize)

    plt.loglog(period, measurements[:, 0], "r--", label="X")
    plt.loglog(period, measurements[:, 1], "g--", label="Y")
    plt.loglog(period, measurements[:, 2], "b--", label="Z")

    for idx, c in enumerate("rgb"):
        if idx == 2:
            wn_label = "White noise fit line"
            rr_label = "Random Rate fit line"
        else:
            wn_label = ""
            rr_label = ""

        plt.loglog(period, fit_wn[idx](period), "m-", label=wn_label)
        plt.loglog(period, fit_rr[idx](period), "y-", label=rr_label)

        plt.loglog(1.0, wn_intercept[idx], f"{c}o", markersize=20)
        plt.loglog(3.0, rr_intercept[idx], f"{c}*", markersize=20)

        plt.loglog(
            period[measurement_min_index[idx]],
            measurement_min[idx],
            f"{c}^",
            markersize=20,
        )

    fitted_model = generate_prediction(
        period,
        q_white=average_white_noise,
        q_bias_instability=average_bias_instability,
        q_walk=average_random_walk,
    )
    plt.loglog(period, fitted_model, "-k", label="fitted model")

    plt.title(sensor_type, fontsize=30)
    plt.ylabel(f"Allan Deviation {units}", fontsize=30)
    plt.legend(fontsize=25)
    plt.grid(True)
    plt.xlabel("Period (s)", fontsize=30)
    plt.tight_layout()

    plt.draw()
    plt.close()

    fig.savefig(f"{measurement_type.lower()}.png", dpi=600, bbox_inches="tight")


def accelerometer_analysis(
    period, acceleration, white_noise_break_point, show_plots: bool = True
):
    """Analyze the accelerometer measurements to get accelerometer parameters."""
    # Compute VRW from the white noise
    # gradient=-0.5, intercept at t=1
    accel_wn_intercept, accel_fit_wn = compute_white_noise_params(
        period, acceleration, white_noise_break_point
    )

    # Compute rate random walk
    # gradient=0.5, intercept at t=3
    accel_rr_intercept, accel_fit_rr = compute_rate_random_walk(period, acceleration)

    accel_min, accel_min_index = compute_bias_instability(acceleration)

    print("ACCELEROMETER:")
    for idx, axis in enumerate("XYZ"):
        print(
            f"{axis} Velocity Random Walk: {accel_wn_intercept[idx]: .5f} m/s/sqrt(s)",
            f"{accel_wn_intercept[idx] * 60: .5f} m/s/sqrt(hr)",
        )

    for idx, axis in enumerate("XYZ"):
        print(
            f"{axis} Bias Instability: {accel_min[idx]: .5f} m/s^2",
            f"{accel_min[idx] * 3600 * 3600: .5f} m/hr^2",
        )

    for idx, axis in enumerate("XYZ"):
        print(f"{axis} Accel Random Walk: {accel_rr_intercept[idx]: .5f} m/s^2/sqrt(s)")

    average_acc_white_noise = accel_wn_intercept.mean()
    average_acc_bias_instability = accel_min.mean()
    average_acc_random_walk = accel_rr_intercept.mean()

    # Use worst value
    worst_accel_white_noise = np.amax(accel_wn_intercept)
    worst_accel_random_walk = np.amax(accel_rr_intercept)

    if show_plots:
        plot_loglog(
            period,
            acceleration,
            accel_fit_wn,
            accel_fit_rr,
            accel_wn_intercept,
            accel_rr_intercept,
            accel_min,
            accel_min_index,
            average_acc_white_noise,
            average_acc_bias_instability,
            average_acc_random_walk,
            "Acceleration",
            "Accelerometer",
            "m/s^2",
        )

    return worst_accel_white_noise, worst_accel_random_walk


def gyroscope_analysis(
    period, rotation_rate, white_noise_break_point, show_plots: bool = True
):
    """Analyze the gyroscope measurements to get gyroscope parameters."""
    # Compute ARW from the white noise
    # gradient=-0.5, intercept at t=1
    gyro_wn_intercept, gyro_fit_wn = compute_white_noise_params(
        period, rotation_rate, white_noise_break_point
    )

    # Compute rate random walk
    # gradient=0.5, intercept at t=3
    gyro_rr_intercept, gyro_fit_rr = compute_rate_random_walk(period, rotation_rate)

    gyro_min, gyro_min_index = compute_bias_instability(rotation_rate)

    print("GYROSCOPE:")
    for idx, axis in enumerate("XYZ"):
        print(
            f"{axis} Angle Random Walk: {gyro_wn_intercept[idx]: .5f} deg/sqrt(s)",
            f"{gyro_wn_intercept[idx] * 60: .5f} deg/sqrt(hr)",
        )

    for idx, axis in enumerate("XYZ"):
        print(
            f"{axis} Bias Instability: {gyro_min[idx]: .5f} deg/s {gyro_min[idx] * 60 * 60: .5f} deg/hr"
        )

    for idx, axis in enumerate("XYZ"):
        print(f"{axis} Rate Random Walk: {gyro_rr_intercept[idx]: .5f} deg/s/sqrt(s)")

    average_gyro_white_noise = gyro_wn_intercept.mean()
    average_gyro_bias_instability = gyro_min.mean()
    average_gyro_random_walk = gyro_rr_intercept.mean()

    # use worst value
    worst_gyro_white_noise = np.amax(gyro_wn_intercept)
    worst_gyro_random_walk = np.amax(gyro_rr_intercept)

    if show_plots:
        plot_loglog(
            period,
            rotation_rate,
            gyro_fit_wn,
            gyro_fit_rr,
            gyro_wn_intercept,
            gyro_rr_intercept,
            gyro_min,
            gyro_min_index,
            average_gyro_white_noise,
            average_gyro_bias_instability,
            average_gyro_random_walk,
            measurement_type="Gyro",
            sensor_type="Gyroscope",
            units="deg/s",
        )

    return worst_gyro_white_noise, worst_gyro_random_walk


def write_imu_yaml(
    worst_accel_white_noise: float,
    worst_accel_random_walk: float,
    worst_gyro_white_noise: float,
    worst_gyro_random_walk: float,
    update_rate: int,
):
    """
    Write IMU calibration parameters to YAML file.

    Args:
        worst_accel_white_noise (float): Accelerometer white noise
        worst_accel_random_walk (float): Accelerometer bias random walk
        worst_gyro_white_noise (float): Gyroscope white noise
        worst_gyro_random_walk (float): Gyroscope bias random walk
        update_rate (int): The IMU update rate.
    """
    print("Writing Kalibr imu.yaml file.")
    with open("imu.yaml", "w") as yaml_file:
        yaml_file.write("# Accelerometer\n")
        yaml_file.write(f"accelerometer_noise_density: {worst_accel_white_noise}\n")
        yaml_file.write(f"accelerometer_random_walk: {worst_accel_random_walk}\n")
        yaml_file.write("\n")

        yaml_file.write("# Gyroscope\n")
        # Convert back to radians here
        yaml_file.write(
            f"gyroscope_noise_density: {worst_gyro_white_noise * np.pi / 180}\n"
        )
        yaml_file.write(
            f"gyroscope_random_walk: {worst_gyro_random_walk * np.pi / 180}\n"
        )
        yaml_file.write("\n")

        yaml_file.write(f"update_rate: {update_rate} # Make sure this is correct\n")

    print("Make sure to update rostopic and rate.")


def analyze(period, allan_deviations, update_rate: int, show_plots: bool = True):
    """Analyze the Allan Deviations to get IMU parameters"""
    acceleration = allan_deviations[:, 0:3]
    rotation_rate = allan_deviations[:, 3:6]

    white_noise_break_point = np.where(period == 10)[0][0]

    worst_accel_white_noise, worst_accel_random_walk = accelerometer_analysis(
        period, acceleration, white_noise_break_point, show_plots=show_plots
    )

    worst_gyro_white_noise, worst_gyro_random_walk = gyroscope_analysis(
        period, rotation_rate, white_noise_break_point, show_plots=show_plots
    )

    write_imu_yaml(
        worst_accel_white_noise,
        worst_accel_random_walk,
        worst_gyro_white_noise,
        worst_gyro_random_walk,
        update_rate=update_rate,
    )
