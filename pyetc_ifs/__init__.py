"""
pyetc_ifs - Exposure Time Calculator for IFS

A Python package for exposure time calculation and signal-to-noise ratio estimation
for IFS instruments MUSE BlueMUSE iredMUSE, WST and Harmoni
"""

__version__ = "0.3"
__releasedate__ = "24 Sep 2026"
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

# Define what gets imported with "from pyetc_ifs import *"
__all__ = [
    # Main classes
    "WST",
    "ETC",
    "iredMUSE",
    "MUSE",
    "blueMUSE",
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
