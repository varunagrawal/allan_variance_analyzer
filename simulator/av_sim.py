"""Run AVA on simulator data"""

from allan_variance import AllanVariance, ROSBagReader

if __name__ == "__main__":
    av = AllanVariance(config_file="config/sim.yaml", output_path=".")

    reader = ROSBagReader("catkin_ws/imu_simulation.bag", av.imu_topic(),
                          av.imu_rate(), av.sequence_time(), av.imu_skip_)
    data = reader.read()

    measurements = data[:, 1:7]
    allan_variances = av(measurements)
