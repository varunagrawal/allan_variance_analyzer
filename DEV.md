# Developer Notes

## Code Structure

We use namespace packages with the src-layout best practice.
The namespace packages are:

1. `allan-variance-analyzer` with module name `allan_variance_analyzer`.
2. `allan-variance-analyzer-ros` with module name `allan_variance_analyzer.ros`.

The `pyproject.toml` at the top-level defines both the main package build details and the [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/).

## ROS1

We use RoboStack to run ROS1 commands.
To setup the robostack env:

```shell
pyenv shell anaconda3-2023.09-0
micromamba activate ros_env

roscore
```

Now build the ROS packages. We assume you have the `examples/ros` subdirectory copied to your `catkin_ws/src` directory.

```shell
catkin_make
source devel/setup.zsh  # Change based on your shell
rosrun allan_variance bag_reader.py "/Users/varunagrawal/Dropbox (GaTech)/Data/oxford/anymal_2018-12-12-15-43-18-001.bag"
```

## ROS2

## Packaging

### Build

In the root directory, run:

```sh
uv build --all
```

### Upload

To upload the package to PyPI, run

```sh
python -m twine upload dist/* --verbose
```
