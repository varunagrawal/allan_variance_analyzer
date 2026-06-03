# allan_variance

Tools for Allan Variance Analysis

## TL;DR

This is a faster and easier-to-use library for Allan Variance Analysis.

### Faster

The popular [allan_variance_ros](https://github.com/ori-drs/allan_variance_ros) package (written in C++), when run on the example `imu_simulation.bag` takes:

`611.90s user 204.06s system 96% cpu 14:06.21 total`

Our library takes:

`40.64s user 2.04s system 91% cpu 46.558 total`

which is an almost 2x speed improvement!

### Easier

- There is no ROS dependency!
- Since this library is written completely in Python, you can simple `import allan_variance` and use the tools.
- `pip` installable.

## README

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
