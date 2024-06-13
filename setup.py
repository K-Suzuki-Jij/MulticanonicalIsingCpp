import sys

try:
    from skbuild import setup
except ImportError:
    from setuptools import setup

setup_requires = [
    "numpy",
    "mpmath",
    "matplotlib",
    "pybind11 >=2.12.0, < 2.13.0",
    "cmake > 3.20",
    "scikit-build > 0.16.0",
]

if any(arg in sys.argv for arg in ("pytest", "test")):
    setup_requires.append("pytest-runner")

setup(
    setup_requires=setup_requires,
    use_scm_version=True,
    packages=[  
        'muca',
        'muca.algorithm',
        'muca.model',
        'muca.results',
        ],
    cmake_install_dir="muca",
    include_package_data=False,
    zip_safe=False,
)
