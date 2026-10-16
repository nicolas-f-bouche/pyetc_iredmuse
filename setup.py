"""Legacy setup.py for backward compatibility."""
from setuptools import setup
import os
import runpy
from pathlib import Path
from setuptools.command.build_py import build_py as _build_py

_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "pyetc_iredmuse", "VERSION"), encoding="ascii") as _f:
    _version = _f.read().strip()


def _get_release_date():
    release_info = os.path.join(_here, "pyetc_iredmuse", "_release_info.py")
    return runpy.run_path(release_info)["__releasedate__"]


class build_py(_build_py):
    def run(self):
        super().run()
        release_info = Path(self.build_lib) / "pyetc_iredmuse" / "_release_info.py"
        release_info.write_text(
            f'__releasedate__ = "{_get_release_date()}"\n',
            encoding="ascii",
        )

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pyetc_iredmuse",
    version=_version,
    description="Exposure Time Calculator for the iredMUSE instrument (VLT)",
    author="Nicolas Bouché, Matteo Ferro, Roland Bacon",
    author_email="nicolas.bouche@cnrs.fr",
    url="https://github.com/nicolas-f-bouche/pyetc_iredmuse",
    packages=["pyetc_iredmuse"],
    package_data={
        "pyetc_iredmuse": [
            "VERSION",
            "data/**/*",
            "Band_Filters/*.txt",
            "ESO_original_spectra/**/*",
        ]
    },
    use_scm_version=False,
    include_package_data=True,
    install_requires=requirements,
    cmdclass={"build_py": build_py},
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
