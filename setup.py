import sys
from setuptools import setup, find_namespace_packages



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
    packages=find_namespace_packages('gym-po'),
    install_requires=["gym", "numpy"],
)