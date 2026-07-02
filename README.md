# Allan Variance Analyzer

Efficient Tool for Allan Variance Analysis

This library provides tools for performing Allan Variance Analysis on IMU data in order to calibrate the IMU parameters.
This is done by computing the Angle Random Walk (ARW), Bias Instability and Gyro Random Walk for the gyroscope, and Velocity Random Walk (VRW), Bias Instability and Accel Random Walk for the accelerometer.

While plenty of libraries exist for this purpose, this library stands out for its immense efficiency, use of pure Python (no bindings or compilation needed), and ease of use, while being fully unit tested for ease-of-mind when adopting it for your use case.

## Faster

The popular [allan_variance_ros](https://github.com/ori-drs/allan_variance_ros) package (written in C++), when run on the example `imu_simulation.bag` takes:

`611.90s user 204.06s system 96% cpu 14:06.21 total`

Our library takes:

`40.64s user 2.04s system 91% cpu 46.558 total`

which is a 15x speed improvement!

## Easier

- There is no ROS dependency! But still useable within ROS.
- Since this library is written completely in Python, you can simply do `import allan_variance_analyzer` to use it.
- `pip` installable.
- Outputs an `imu.yaml` file following the [Kalibr](https://github.com/ethz-asl/kalibr) format.

## Collecting IMU Data

Please follow the below steps to collect the data needed to calibrate the IMU.

1. Place your IMU on some damped surface and record your IMU data to the file format of your choice. Please be sure to record **at least** 3 hours of data. The longer the sequence, the more accurate the results.

2. **Recommended** Reorganize messages by timestamp.

3. Define an interface for reading the data file and loading it as a `Tx6` `numpy` array where `T` is the number of data samples and we have 3-dimensional gyroscrope and 3-dimensional accelerometer data.

4. Create a configuration YAML file which specifies the IMU rate (Hz) and the measure rate (Hz) which is the rate at which data is measured/subsampled. You can add additional parameters for your particular interface. An example config file is availabe in [config/sim.yaml](config/sim.yaml).

5. Run Allan Variance computation tool. Please see [analyze_rosbag.py](scripts/analyze_rosbag.py) for an example script using ROS bag data.

6. The result is a generated `imu.yaml` file in the Kalibr format.

## Example Data

We use the 3 hour log of a [Realsense D435i IMU](https://drive.google.com/file/d/1ovI2NvYR52Axt-KuRs5HjVk7-57ky72H/view?usp=sharing) with timestamps already re-arranged, as provided by [raabuchanan](https://github.com/raabuchanan).

Our library automatically generates a Kalibr compatible file as `imu.yaml`:

```yaml
# Accelerometer
accelerometer_noise_density: 0.006308226052016165 
accelerometer_random_walk: 0.00011673723527962174 

# Gyroscope
gyroscope_noise_density: 0.00015198973532354657 
gyroscope_random_walk: 2.664506559330434e-06 

update_rate: 400.0 # Make sure this is correct

```

## Author

[Varun Agrawal](varunagrawal.github.io)

If you use this package for academic work, please consider using the citation below:

```bib
@software{AllanVarianceAnalyzer,
  author       = {Varun Agrawal},
  title        = {Fast Allan Variance Analysis},
  year         = {2026},
  version      = {1.0.1},
  url          = {https://github.com/varunagrawal/allan_variance}
}
```

## References

- [Indirect Kalman Filter for 3D Attitude Estimation, Trawny & Roumeliotis](http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf)
- [An introduction to inertial navigation, Oliver Woodman](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf)
- [Characterization of Errors and Noises in MEMS Inertial Sensors Using Allan Variance Method, Leslie Barreda Pupo](https://upcommons.upc.edu/bitstream/handle/2117/103849/MScLeslieB.pdf?sequence=1&isAllowed=y)
- [Kalibr IMU Noise Documentation](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model)

## Local Development

We use `uv` to manage the project.

To run unit tests:

```sh
uv run pytest
```

To run an example script:

```sh
uv run python scripts/analyze_rosbag.py config/sim.yaml imu_simulation.bag
```
