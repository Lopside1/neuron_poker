"""Distribution to pipy"""

from setuptools import setup, find_packages, Extension
import pybind11

with open("readme.rst") as readme:
    long_description = readme.read()

ext_modules = [
    Extension(
        "tools.montecarlo_cpp.pymontecarlo",
        ["tools/montecarlo_cpp/.rendered.pymontecarlo.cpp"],
        include_dirs=[
            pybind11.get_include(),
            # add any additional include directories here
        ],
        extra_compile_args=["-std=c++14"],  # <--- Use C++14 (or "-std=c++17")
    ),
]

setup(
    name='neuron_poker',
    version='1.0.0',
    long_description=long_description,
    url='https://github.com/dickreuter/neuron_poker',
    author='Nicolas Dickreuter',
    author_email='dickreuter@gmail.com',
    license='MIT',
    description=('OpenAi gym for textas holdem poker with graphical rendering and montecarlo.'),
    packages=find_packages(exclude=['tests', 'gym_env', 'tools']),
    install_requires=['pyglet', 'pytest', 'pandas', 'pylint', 'gym', 'numpy', 'matplotlib'],
    platforms='any',
)
