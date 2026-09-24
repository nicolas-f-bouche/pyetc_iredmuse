# pyetc_ifs

Exposure Time Calculator (ETC) for the iredMUSE instrument.

## Credits

forked from Project Link: https://github.com/ferromatteo/pyetc_wst

thanks to Matteo Ferro - [matteo.ferro@inaf.it]

Cite: Ferro, Genoni et al. 2026, SPIE


## Description

**pyetc_ifs** is a Python package for exposure time calculation and signal-to-noise ratio (SNR) estimation for IFU instruments, including:

- muse

- wst

- bluemuse

- iredmuse

- harmoni

## Requirements

[ All of them can be installed via `pip` ]
- Python >= 3.9
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- astropy >= 5.0.0
- mpdaf >= 3.5.0
- skycalc_cli

```
pip install "numpy>=1.20.0" "scipy>=1.7.0" "matplotlib>=3.3.0" "astropy>=5.0.0" "mpdaf>=3.5.0" "skycalc_cli"
```

## Installation

### From GitHub

You can install directly from GitHub using pip:

```bash
pip install git+https://github.com/nfbouche/pyetc_ifs.git
```

If you already have it installed via pip, you can upgrade it with:

#### Option 1: forced (recommended)
```bash
pip install --force-reinstall git+https://github.com/nfbouche/pyetc_ifs.git
```

#### Option 2: normal upgrade
```bash
pip install --upgrade git+https://github.com/nfbouche/pyetc_ifs.git
```

#### Option 3: uninstall and reinstall (cleanest option)
```bash
pip uninstall pyetc_ifs
pip install git+https://github.com/nfbouche/pyetc_ifs.git
```

## Quick Start

```python
from pyetc_iredmuse import iredMUSE

# skip_dataload = False will load the static sky configurations + general transmissions
redmuse = iredMUSE(log='DEBUG', skip_dataload=False)

# Display instrument information
redmuse.info()

# Access specific configuration
ifs = redmuse.ifs['zband']

# Build the full dictionaries needed for computation (full_obs), which will include observing conditions, source properties, computation requests, and instrument configuration
full_obs = {...}
con, ob, spe, im, spe_input = redmuse.build_obs_full(full_obs)

# Compute time or snr given the full dictionary results

# for SNR:
res_snr = redmuse.snr_from_source(con, im, spe, debug=True / False)

# for SNR at a specific wavelength:
res_snr_at_wave = redmuse.snr_at_wave(con, im, spe, debug=True / False)

# for time/exposures/best combination
res_time = redmuse.time_from_source(con, im, spe, compute='dit' / 'ndit' / 'best', debug=True / False)

# --- SNR in a spectral window ---
from pyetc_ifs.etc import snr_in_window

# Get median SNR in [5000, 6000] Å from a snr_from_source result
med = snr_in_window(res_snr, lam1=5000, lam2=6000)

# --- Find NDIT for target median SNR in a window ---
result = redmuse.time_from_source_window(con, im, spe,
                                         lam1=5000, lam2=6000,
                                         target_snr=10,
                                         compute='ndit')
print(f"Required NDIT: {result['ndit']}, achieved median SNR: {result['median_snr']:.2f}")
# The full snr_from_source result is in result['res']
```
`debug=True/False` allows to print detailed info of the current run.

**NOTE**: *`time_from_source()` basically update the 'dit', 'ndit' or both values in the obs. dictionary to the value/values needed to reach a specific SNR at a specific wavelength, after it you could run a `res_snr = wst.snr_from_source(con, im, spe)` and plot the SNR to check the results.*

A full_obs dictionary should look like this (detailed information are given in the file **encoding.txt**):
```python
full_obs = {
    "INS": "moslr",
    "CH": "red",
    
    "NDIT": 1,
    "DIT": 600, 
    
    "SNR": 5,
    "Lam_Ref": 5000,
    
    "OBJ_FIB_DISP": None,  # omit or set None for the default 90% MOS centering efficiency
    
    "PWV": 10,
    "FLI": 0.5,
    "MOON_SEP": 45,
    "SEE": 0.8,
    "AM": 1.2,
    "SKYCALC": False,
    "GLAO": False,   # set True to enable Ground Layer AO mode
    
    "Obj_SED": 'template',
    "SED_Name": 'MARCS_8000K_lg+45',
    "UPLOAD_FILE": "path/to/spec.txt"
    
    "OBJ_MAG": 15, #can be None for loaded spectrum
    "MAG_SYS": 'Vega',
    "MAG_FIL": 'V',
    
    "Z": 0,
    "BB_Temp": 9000.,
    "PL_Index": None,
    
    "SEL_FLUX": 50e-16,
    "SEL_CWAV": 8000,
    "SEL_FWHM":20,
    
    "Obj_Spat_Dis": 'resolved',
    
    "IMA": 'moffat',
    
    "IMA_FWHM": 0.5,
    "IMA_BETA": 2.5,
    
    "Sersic_Reff": 1,
    "Sersic_Ind": 3,
    
    "COADD_WL": 10,
    
    "COADD_XY": 1, #(all integer numbers or 'best')

    # SNR window: target median SNR over a wavelength range instead of at a single reference
    # wavelength. Applies to compute='dit', 'ndit', 'best'. Ignored for line sources.
    "SNR_RANGE": False,  # set True to enable window-based SNR targeting
    "LAM_WIN1": 5000,    # window start in Å
    "LAM_WIN2": 6000,    # window end in Å
}
```

The throughput system is selected when initializing `WST`:

```python
WST(throughput_system='AR')      # default when omitted or set to None
WST(throughput_system='GRINAR')
```

Only `AR` and `GRINAR` are valid values; any other value raises `ValueError`.

For MOS observations, `OBJ_FIB_DISP` is optional and defaults to `None`. When it is omitted or set to `None`, the core applies the 90% mean object-centering efficiency. If a non-negative displacement is provided, the core uses that value in the geometric fiber-aperture calculation without the additional 90% factor.
**NOTE**: *"COADD_XY": 'best' — automatically selects the spatial coadding that maximizes the SNR. Like the compute options in `time_from_source`, it updates "COADD_XY" in the obs dictionary with the chosen value.*

**NOTE (3)**: *`"SNR_RANGE": True` — when set, `time_from_source` targets the median SNR over the wavelength window `[LAM_WIN1, LAM_WIN2]` instead of the SNR at `Lam_Ref`. Works for `compute='dit'`, `'ndit'`, and `'best'` (which internally uses `'ndit'`). Line sources (`Obj_SED='line'`) always use their line-center wavelength and ignore this flag. The window is automatically clipped to the instrument spectral range if it extends beyond it.*

**NOTE (2)**: *When using `"Obj_SED": "upload"`, you must provide `"UPLOAD_FILE": "/path/to/spectrum.dat"` pointing to a two-column ASCII file/FITS table (wavelength, flux). Optional comment headers (or "units" of the FITS columns) set units: `# nm` or `# aa` for wavelength (default: Å - `aa`), `# fl` or `# ph` for flux (default: erg/cm²/s/Å - `fl`). Set `"OBJ_MAG": null` to use the spectrum as-is, or set a numeric value (e.g. `18`) to normalize it to that magnitude in the chosen `MAG_FIL`/`MAG_SYS` band.*

After the computation results can be plotted easily accessing the mpdaf `Spectrum` objects in the results dictionary like this:
```python
res_snr['spec']['snr'].plot()
```
![Noise Plot](images/SNR.png)

or
```python
res_snr['spec']['nph_source'].plot()
```

In general the results of the snr computation `res_snr` will have a main dictionary `res_snr['spec']` which contains several sub-dictionaries, all related to the 
integration in the aperture area for the MOS, and the requested `COADD_XY x COADD_XY` for the IFS (which has also another dictionary `res_snr['peak']`) for the central pixel value.

These sub-dictionaries include:
- 'nph_source': photon from source
- 'nph_sky': photon from sky
- 'snr': snr in the aperture/integration area
- 'snr_rebin': snr in the aperture/integration area, rebinned with COADD_WL
- 'simulated_counts': 1D extracted raw spectrum
- 'noise': another dictionary with noise components and their fractions
  - 'tot'
  - 'frac_source'
  - 'frac_sky'
  - 'frac_dark' 
  - 'frac_ron'


Moreover, there is a handy function to plot all the noise components together, and will accept `res_snr['spec']['noise']` (and also `res_snr['peak']['noise']`) for IFS): 

```python
plot_noise_components(res_snr['spec']['noise'])
```
![Noise Plot](images/noise.png)

## Notebook
`iredMUSE_LimMag.ipynb`: computes the limiting magnitude of the WST Integral Field Spectrograph (IFS) as a function of wavelength, for both point sources and extended sources (surface brightness), across the blue and red channels.
For a given target S/N ratio, the notebook sweeps the wavelength range and finds — via Brent's root-finding method — the faintest AB magnitude detectable under three sky background conditions: dark, grey, and bright time.


## Documentation

update in future version

## Citation

This package has been developed from the original `pyetc` package available at https://github.com/RolandBacon/pyetc

update in future version

## Version

### 0.1 — 26 June 2026
- initial version
- Throughput curves not official

### 1.0 — 1 July 2026
- Official release of the iredMUSE Exposure Time Calculator.
- Throughput curves not official
  
## Contact

Nicolas Bouché - [nicolas.bouche@cnrs.fr]

Project Link: [https://github.com/nfbouche/pyetc_iredmuse](https://github.com/nfbouche/pyetc_iredmuse)
