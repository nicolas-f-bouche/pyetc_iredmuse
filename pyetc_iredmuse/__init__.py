"""
pyetc_iredmuse - Exposure Time Calculator for IFS

A Python package for exposure time calculation and signal-to-noise ratio estimation
for IFS instruments MUSE BlueMUSE iredMUSE, WST and Harmoni
"""

from pathlib import Path

__version__ = Path(__file__).with_name("VERSION").read_text(encoding="ascii").strip()
from ._release_info import __releasedate__

__author__ = "Nicolas Bouché & Matteo Ferro & Roland Bacon"

# Import main classes and functions
from .iredmuse import iredMUSE
from .muse import MUSE
from .etc import (
    ETC,
    sersic,
    moffat,
    get_seeing_fwhm,
    lsf_resolution,
    compute_sky,
    mask_spectrum_edges,
    mask_line_region,
    mask_spectra_in_dict,   
    convolve_and_center,
    plot_noise_components,
)
from .specalib import (
    PhotometricSystem,
    SEDModels,
    FilterManager,
    plot_spectra_comparison,
)

# Define what gets imported with "from pyetc_iredmuse import *"
__all__ = [
    # Main classes
    "ETC",
    "iredMUSE",
    "MUSE",
    "PhotometricSystem",
    "SEDModels",
    "FilterManager",
    # Functions
    "sersic",
    "moffat",
    "get_seeing_fwhm",
    "lsf_resolution",
    "compute_sky",
    "mask_spectrum_edges",
    "mask_line_region",
    "mask_spectra_in_dict",
    "convolve_and_center",
    "plot_noise_components",
    "plot_spectra_comparison",
]
