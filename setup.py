import sys
from setuptools import setup, find_namespace_packages, find_packages


setup(
    name="gym_po",
    version="0.0.2",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
    ],
    packages=[pkg for pkg in find_packages() if pkg.startswith('gym_po')],
    install_requires=["gym>=0.21.0", "numpy<1.22,>=1.21", "numba>=0.55.1", "pygame", "dotsi"],
    python_requires=">=3.8"
)