import os
import fnmatch
from setuptools import setup, find_namespace_packages, find_packages, fin

def find_data_files(package_dir, patterns, excludes=()):
  """Recursively finds files whose names match the given shell patterns."""
  paths = set()

  def is_excluded(s):
    for exclude in excludes:
      if fnmatch.fnmatch(s, exclude):
        return True
    return False

  for directory, _, filenames in os.walk(package_dir):
    if is_excluded(directory):
      continue
    for pattern in patterns:
      for filename in fnmatch.filter(filenames, pattern):
        # NB: paths must be relative to the package directory.
        relative_dirpath = os.path.relpath(directory, package_dir)
        full_path = os.path.join(relative_dirpath, filename)
        if not is_excluded(full_path):
          paths.add(full_path)
  return list(paths)

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
    python_requires=">=3.8",
    package_data={
        'dm_control':
            find_data_files(
                package_dir='gym_po',
                patterns=['*.amc', '*.msh', '*.png', '*.skn', '*.stl', '*.xml',
                          '*.textproto', '*.h5'],
                excludes=[
                    '*/dog_assets/extras/*',
                    '*/kinova/meshes/*',  # Exclude non-decimated meshes.
                ]),
    },
)