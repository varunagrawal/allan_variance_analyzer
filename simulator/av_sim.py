"""Run AVA on simulator data"""

from allan_variance_analyzer import AllanVarianceAnalyzer, ROSBagReader

if __name__ == "__main__":
    av = AllanVarianceAnalyzer(
        config_file="config/sim.yaml", output_path=".", write_allan_deviations=True
    )

    reader = ROSBagReader(
        "catkin_ws/imu_simulation.bag",
        av.imu_topic(),
        av.imu_rate(),
        av.sequence_time(),
        av.imu_skip_,
    )
    data = reader.read()

    measurements = data[:, 1:7]
    allan_variances = av(measurements)
