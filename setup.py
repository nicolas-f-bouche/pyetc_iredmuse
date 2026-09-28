"""
Legacy setup.py for backward compatibility.
Version is read from pyetc_iredmuse/__init__.py — edit only that file to bump the version.
"""
from setuptools import setup
import re
import os

# Single source of truth for version
_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "pyetc_iredmuse", "__init__.py")) as _f:
    _version = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', _f.read(), re.M).group(1)

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pyetc_iredmuse",
    version=_version,
    description="Exposure Time Calculator for the iredMUSE instrument (VLT)",
    author="Nicolas Bouché, Matteo Ferro, Roland Bacon",
    author_email="nicolas.bouche@cnrs.fr",
    url="https://github.com/nfbouche/pyetc_iredmuse",
    packages=["pyetc_iredmuse"],
    package_data={
        "pyetc_iredmuse": [
            "data/**/*",
            "Band_Filters/**/*.txt",
            "ESO_original_spectra/**/*",
        ]
    },
    use_scm_version=False,
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    keywords="astronomy spectroscopy exposure-time-calculator WST",
)
