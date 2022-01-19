import sys
from setuptools import setup, find_namespace_packages, find_packages


setup(
    name="gym_po",
    version="0.0.1",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
    ],
    packages=find_packages(include=['gym_po', 'gym_po.*']),
    install_requires=["gym", "numpy", "numba"],
)