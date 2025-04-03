# allan_variance
Tools for Allan Variance Analysis

## TL;DR

This is a faster and easier to use library for Allan Variance Analysis.

### Faster

The popular [allan_variance_ros](https://github.com/ori-drs/allan_variance_ros) package (written in C++), when run on the example `imu_simulation.bag` takes:

`611.90s user 204.06s system 96% cpu 14:06.21 total`

Our library takes:

`357.53s user 3.79s system 96% cpu 6:13.18 total`

which is an almost 2x speed improvement!

### Easier

- There is no ROS dependency!
- Since this library is written completely in Python, you can simple `import allan_variance` and use the tools.
- Future updates may include accelerator support to run on parallel devices.

## README



## Local Development

We use `poetry` to manage the project.

First, tell `poetry` to use the system python:
```sh
poetry env use system
```

After this, you can run the following and it should be accessible in your python environment
```sh
poetry install
```
