from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[],  # directory inside `src`
    package_dir={'': 'src'},
    scripts=['src/bag_reader.py'])
setup(**d)