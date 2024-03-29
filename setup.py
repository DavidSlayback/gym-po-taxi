import os
import fnmatch
from setuptools import setup, find_namespace_packages, find_packages

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
    version="0.0.4",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
    ],
    packages=[pkg for pkg in find_packages() if pkg.startswith('gym_po')],
    install_requires=["gymnasium>=0.26.0", "numpy>=1.23", "numba>=0.56.3", "pygame", "dotsi"],
    python_requires=">=3.8",
    package_data={
        'gym_po':
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