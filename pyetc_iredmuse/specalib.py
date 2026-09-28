import os
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, integrate
from astropy import units as u
from astropy.constants import h, c, k_B
from importlib import resources

class PhotometricSystem:
    """
    Class for managing photometric systems (Vega and AB) and conversions between them.
    """
    def __init__(self):
        # Definition of available systems and filters, should be modified as needed
        self.MAG_SYSTEMS = ["Vega", "AB"]
        
        # From https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
        # 2025-09-26: GAIA filters added from Riello+ 2021, A&A, 649, A3 (Gaia EDR3), for the las one we don't have the reference yet
        self.filters_vega = ["U", "B", "V", "R", "I", "J", "H", "K", "GbpGAIA", "GGAIA", "GrpGAIA", "GrvsGAIA"]
        # Last four values: check Jordi et al. 2010, A&A, 523, A48, assuming Gbp-G ~ 0.16 and Grp-G ~ -0.13 for a G2V star (Vega-like)
        self.ab_vega_diff_vf = [0.79, -0.09, 0.02, 0.21, 0.45, 0.91, 1.39, 1.85, 0.0154, 0.1137, 0.3561, 0.00]
        
        # Should be properly computed, since for now we are using the conversion from https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
        self.filters_AB = ["uSDSS", "gSDSS", "rSDSS", "iSDSS", "zSDSS", "uLSST", "gLSST", "rLSST", "iLSST", "zLSST"]
        self.ab_vega_diff_abf = [0.91, -0.08, 0.16, 0.37, 0.54, 0.91, -0.08, 0.16, 0.37, 0.54] 
        
        # Zero points for Vega from 
        # https://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html#mags
        # Gaia filters from Riello+ 2021, A&A, 649, A3 (Gaia EDR3), with their pivot wave, last one unchaged
        # BP: [0: ll in A, 1: Fv in [erg/cm^2/s/Hz], 2: Fll in [erg/cm^2/s/A], 3: PHll in [Photons/cm^2/s/A]]
        self.VEGA_flux_zeropoints = {
            "U": [3600., None, 4.18023e-9, 757.5], 
            "B": [4380., None, 6.60085e-9, 1455.4],
            "V": [5450., None, 3.60994e-9, 990.4],
            "R": [6410., None, 2.28665e-9, 737.9],
            "I": [7980., None, 1.22603e-9, 492.5],
            "J": [12200., None, 3.12e-10, 191.6],
            "H": [16300., None, 1.14e-10, 93.5],
            "K": [21900., None, 3.94e-11, 43.4],
            "GbpGAIA": [5109.7, None, 4.01188e-9, None],
            "GGAIA": [6217.9, None, 2.40375e-9, None],
            "GrpGAIA": [7769.1, None, 1.58489e-9, None],
            "GrvsGAIA": [8578.16, None, 9.03937e-10, None]
        }

        # Zero points for AB from 
        # http://svo2.cab.inta-csic.es/theory/fps/index.php?id=SLOAN & id=LSST
        # BP: [0: ll in A, 1: Fv in [erg/cm^2/s/Hz], 2: Fll in [erg/cm^2/s/A], 3: PHll in [Photons/cm^2/s/A]]
        self.AB_flux_zeropoints = {
            "uSDSS": [3542.10, 3631., 8.88093e-9, None],
            "gSDSS": [4723.59, 3631., 4.79807e-9, None],
            "rSDSS": [6201.71, 3631., 2.78937e-9, None],
            "iSDSS": [7672.59, 3631., 1.82728e-9, None],
            "zSDSS": [10500.61, 3631., 9.28119e-10, None],
            "uLSST": [3641.17, 3631., 8.57499e-9, None],
            "gLSST": [4704.08, 3631., 4.83202e-9, None],
            "rLSST": [6155.82, 3631., 2.83044e-9, None],
            "iLSST": [7504.64, 3631., 1.91692e-9, None],
            "zLSST": [8695.51, 3631., 1.43756e-9, None]
        }  

        self.band_filters = self._load_filter_profiles()  
    
    def _load_filter_profiles(self):
        """Loads filter transmission profiles from files."""
        filters_folder = str(resources.files("pyetc_iredmuse").joinpath("Band_Filters/"))
        band_filters = {}
        
        for filename in os.listdir(filters_folder):
            if filename.endswith(".txt"):
                with open(os.path.join(filters_folder, filename), 'r', encoding='latin-1') as f:
                    lines = f.readlines()
                    data = np.loadtxt([line for line in lines if not line.strip().startswith(('#', '!'))])
                    band_name = filename[:-4]  # Remove '.txt' extension
                    band_filters[band_name] = data
        return band_filters
                
    def get_flux_zeropoint(self, band, system="Vega", quantity="Fll"):
        """
        Returns the zero point flux for a given filter and system.
        
        Parameters:
        - band: filter name (e.g. 'U', 'g')
        - system: 'Vega' or 'AB'
        - quantity: 'Fv' [erg/cm²/s/Hz], 'Fll' [erg/cm²/s/Å], 'PHll' [photons/cm²/s/Å]
        
        Returns:
        - lambda_eff: effective wavelength [Å]
        - zeropoint: zero point value in requested units
        """
        if system == "Vega":
            if band not in self.filters_vega:
                raise ValueError(f"Invalid Vega filter. Choose from: {self.filters_vega}")
            zp_dict = self.VEGA_flux_zeropoints
        elif system == "AB":
            if band not in self.filters_AB:
                raise ValueError(f"Invalid AB filter. Choose from: {self.filters_AB}")
            zp_dict = self.AB_flux_zeropoints
        else:
            raise ValueError("Invalid photometric system. Choose 'Vega' or 'AB'")
        
        if quantity == "Fv":
            sel = 1
        elif quantity == "Fll":
            sel = 2
        elif quantity == "PHll":
            sel = 3
        else:
            raise ValueError("Invalid quantity. Choose 'Fv', 'Fll' or 'PHll'")
        
        return zp_dict[band][0], zp_dict[band][sel]
    
    def convert_magnitude(self, mag, band, from_system, to_system):
        """
        Converts magnitude between photometric systems.
        
        Parameters:
        - mag: magnitude value
        - band: filter name
        - from_system: source system ('Vega' or 'AB')
        - to_system: target system ('Vega' or 'AB')
        
        Returns:
        - mag_converted: converted magnitude
        """
        if from_system == to_system:
            return mag
        
        if from_system == "Vega" and to_system == "AB":
            if band not in self.filters_AB:
                raise ValueError(f"Invalid AB filter for conversion. Choose from: {self.filters_AB}")
            return mag + self.ab_vega_diff_abf[self.filters_AB.index(band)]
        
        elif from_system == "AB" and to_system == "Vega":
            if band not in self.filters_vega:
                raise ValueError(f"Invalid Vega filter for conversion. Choose from: {self.filters_vega}")
            return mag - self.ab_vega_diff_vf[self.filters_vega.index(band)]
        
        else:
            raise ValueError("Unsupported conversion")

    def auto_conversion(self, mag, band, sys):
        """
        Automatically converts magnitude to the appropriate system for the given band.
        
        Parameters:
        - mag: magnitude
        - band: filter name
        - sys: current system ('Vega' or 'AB')
        
        Returns:
        - new_mag: converted magnitude
        - new_sys: new system
        """
        new_sys = sys
        new_mag = mag
        if sys == "Vega":
            if band not in self.filters_vega:
                new_sys = 'AB'
                new_mag = self.convert_magnitude(mag, band, sys, new_sys)
        elif sys == "AB":
            if band not in self.filters_AB:
                new_sys = 'Vega'
                new_mag = self.convert_magnitude(mag, band, sys, new_sys)
        
        if new_sys != sys:
            print("Wrong coupling of filter and system")
            print("Conversion:")
            print(f"{sys} > {new_sys}")
            print(f"{mag} > {new_mag}")
        return new_mag, new_sys
    
class SEDModels:
    """
    Class for generating spectral energy distribution (SED) models.
    """
    # Class-level cache for loaded templates
    _template_cache = {}
    
    def __init__(self):
        # Save all filenames from ESO_original_spectra/ directory to a dictionary
        self.eso_spectra_files = {}
        eso_spectra_dir = str(resources.files("pyetc_iredmuse").joinpath("ESO_original_spectra/"))
        try:
            for idx, filename in enumerate(os.listdir(eso_spectra_dir)):
                if os.path.isfile(os.path.join(eso_spectra_dir, filename)):
                    # Using filename without extension as the key
                    name = os.path.splitext(filename)[0]
                    self.eso_spectra_files[name] = os.path.join(eso_spectra_dir, filename)
        except FileNotFoundError:
            print(f"Warning: Directory {eso_spectra_dir} not found")
        
    @staticmethod
    def blackbody(wavelength, temperature):
        """
        Generates a blackbody spectrum.
        
        Parameters:
        - wavelength: wavelength array [Å]
        - temperature: temperature [K]
        
        Returns:
        - flux: flux [erg/s/cm²/Å]
        """
        wavelength_m = wavelength * 1e-10  # Convert Å to meters
        exponent = h.value * c.value / (wavelength_m * k_B.value * temperature)
        
        flux = (2 * h.value * c.value**2 / (wavelength_m**5 * (np.exp(exponent) - 1))) * 1e-10
        return flux * 1e7  # Convert W/m²/m to erg/s/cm²/Å
    
    @staticmethod
    def powerlaw(wavelength, slope, norm_wavelength=5500, norm_flux=1.0):
        """
        Generates a power law spectrum.
        
        Parameters:
        - wavelength: wavelength array [Å]
        - slope: power law index
        - norm_wavelength: normalization wavelength [Å]
        - norm_flux: flux at normalization wavelength
        
        Returns:
        - flux: normalized flux
        """
        return norm_flux * (wavelength / norm_wavelength)**slope

    @staticmethod
    def gaussian_line(wavelength, center, flux, fwhm):
        """
        Generates a gaussian spectral line.
        
        Parameters:
        - wavelength: wavelength array [Å]
        - center: line center [Å]
        - flux: integrated line flux [erg/s/cm²]
        - fwhm: full width at half maximum [Å]
        
        Returns:
        - flux: flux [erg/s/cm²/Å]
        """
        sigma = fwhm / 2.355
        exponent = -0.5 * ((wavelength - center) / sigma)**2
        return (flux / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)
    
    @staticmethod
    def interpolate_spectrum(wavelength, target_wavelength, flux):
        """
        Interpolates a spectrum to new wavelengths.
        
        Parameters:
        - wavelength: original wavelengths [Å]
        - target_wavelength: target wavelengths [Å]
        - flux: original flux [any unit]
        
        Returns:
        - flux_interp: interpolated flux
        """
        return np.interp(target_wavelength, wavelength, flux)
    
    @staticmethod
    def _resolve_template_path(filename):
        if os.path.dirname(filename):
            if os.path.exists(filename):
                return filename
            root, ext = os.path.splitext(filename)
            if ext.lower() == '.dat':
                alt = root + '.sed'
                if os.path.exists(alt):
                    return alt
            elif ext.lower() == '.sed':
                alt = root + '.dat'
                if os.path.exists(alt):
                    return alt
            return filename
        else:
            # Takes the directory of the current file (specalib.py)
            base_dir = os.path.dirname(__file__)
            template_dir = os.path.join(base_dir, 'ESO_original_spectra')
            candidate = os.path.join(template_dir, filename)
            if os.path.exists(candidate):
                return candidate

            root, ext = os.path.splitext(candidate)
            if ext.lower() == '.dat':
                alt = root + '.sed'
                if os.path.exists(alt):
                    return alt
            elif ext.lower() == '.sed':
                alt = root + '.dat'
                if os.path.exists(alt):
                    return alt
            elif ext == '':
                for suffix in ('.dat', '.sed'):
                    alt = candidate + suffix
                    if os.path.exists(alt):
                        return alt

            return candidate
        
    @classmethod
    def template(cls, filename, waveunit='AA', unitsf='Fll'):
        """
        Reads a two-column file, skipping lines that start with '#' or '!'.
        Uses a class-level cache to avoid re-reading files from disk.
        
        Parameters:
        - filename: path to the template file
        - waveunit: wavelength unit, 'AA' (default) or 'nm'
        - unitsf: flux unit 'Fll' or 'PHll'

        Returns:
        - tem: template name
        - wave: wavelength array [Å]
        - flux: flux array
        """
        filepath = SEDModels._resolve_template_path(filename)
        
        # Check cache first
        if filepath not in cls._template_cache:
            with open(filepath, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        
            # Filter lines not starting with "#" or "!"
            data = np.loadtxt([line for line in lines if not line.strip().startswith(('#', '!'))])
            # Store in cache (wave and flux as copies to avoid mutation)
            cls._template_cache[filepath] = (data[:, 0].copy(), data[:, 1].copy())
        
        # Get from cache
        wave, flux = cls._template_cache[filepath]
        # Return copies to avoid mutation of cached data
        wave = wave.copy()
        flux = flux.copy()
        
        tem = os.path.basename(filename).split('.')[0]
            
        if waveunit == 'nm':
            wave *= 10
            
        if unitsf == 'PHll':
            flux = 1.98644746e-08 * (flux / wave)
        return tem, wave, flux

    @staticmethod
    def parse_uploaded_spectrum(file_content, waveunit=None, fluxunit=None):
        """Parse an uploaded spectrum from text or FITS content.

        For text files:
          The file must contain two columns: wavelength and flux.
          Lines starting with '#' or '!' are treated as comments.
          Special single-keyword comment lines set units:
            # nm  or  # aa   -> wavelength unit (default: aa = Angstrom)
            # fl  or  # ph   -> flux unit (default: fl = erg/cm^2/s/AA)

        For FITS files:
          Reads the first extension (BINTABLE) or primary HDU.
          Expects at least two columns; the first is wavelength, the second is flux.
          Wavelength is assumed in Angstrom unless a column unit indicates nm.

        Parameters
        ----------
        file_content : str, bytes, or path-like
            Text content, raw bytes (for FITS), or file path of the spectrum file.
        waveunit : str or None
            Override wavelength unit ('nm' or 'aa'). If None, auto-detect.
        fluxunit : str or None
            Override flux unit ('fl' or 'ph'). If None, auto-detect.

        Returns
        -------
        wave : ndarray
            Wavelength array in Angstrom.
        flux : ndarray
            Flux array in erg/cm^2/s/AA.
        """
        # --- Detect FITS content ---
        raw_bytes = None
        if isinstance(file_content, bytes):
            raw_bytes = file_content
        elif isinstance(file_content, str) and not file_content.startswith('SIMPLE'):
            # Could be a file path or text content
            if os.path.isfile(file_content) and file_content.lower().endswith(('.fits', '.fit')):
                with open(file_content, 'rb') as f:
                    raw_bytes = f.read()

        # Check for FITS magic bytes
        is_fits = raw_bytes is not None and raw_bytes[:6] == b'SIMPLE'

        if is_fits:
            return SEDModels._parse_fits_spectrum(raw_bytes, waveunit=waveunit, fluxunit=fluxunit)

        # --- Text parsing (original logic) ---
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf-8', errors='replace')

        lines = file_content.splitlines()

        detected_waveunit = 'aa'
        detected_fluxunit = 'fl'
        data_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('#') or stripped.startswith('!'):
                token = stripped.lstrip('#!').strip().lower()
                if token == 'nm':
                    detected_waveunit = 'nm'
                elif token == 'aa':
                    detected_waveunit = 'aa'
                elif token == 'ph':
                    detected_fluxunit = 'ph'
                elif token == 'fl':
                    detected_fluxunit = 'fl'
                continue
            data_lines.append(stripped)

        if not data_lines:
            raise ValueError("Spectrum file contains no data lines.")

        data = np.loadtxt(io.StringIO('\n'.join(data_lines)))
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Spectrum file must contain at least two columns (wavelength, flux).")

        wave = data[:, 0].copy()
        flux = data[:, 1].copy()

        # Sort by wavelength
        order = np.argsort(wave)
        wave = wave[order]
        flux = flux[order]

        # Apply unit overrides
        wu = waveunit if waveunit is not None else detected_waveunit
        fu = fluxunit if fluxunit is not None else detected_fluxunit

        # Convert wavelength to Angstrom
        if wu == 'nm':
            wave *= 10.0

        # Convert photon flux to energy flux: F_lambda = N_ph * E_ph / AA
        if fu == 'ph':
            h_cgs = 6.62607015e-27   # erg s
            c_cgs = 2.99792458e18    # AA/s
            E_photon = h_cgs * c_cgs / wave  # erg per photon
            flux = flux * E_photon

        return wave, flux

    @staticmethod
    def _parse_fits_spectrum(fits_bytes, waveunit=None, fluxunit=None):
        """Parse a FITS spectrum from raw bytes.

        Supports BINTABLE extensions (as produced by the ETC download) and
        simple IMAGE HDUs with a WCS wavelength axis.

        Parameters
        ----------
        fits_bytes : bytes
            Raw FITS file content.
        waveunit : str or None
            Override wavelength unit ('nm' or 'aa').
        fluxunit : str or None
            Override flux unit ('fl' or 'ph').

        Returns
        -------
        wave : ndarray  [Angstrom]
        flux : ndarray  [erg/cm^2/s/AA]
        """
        from astropy.io import fits as pyfits
        from astropy.table import Table as AstropyTable

        hdulist = pyfits.open(io.BytesIO(fits_bytes))

        wave = None
        flux = None

        # Try BINTABLE extensions first
        for hdu in hdulist[1:]:
            if isinstance(hdu, (pyfits.BinTableHDU, pyfits.TableHDU)):
                tbl = AstropyTable.read(hdulist, hdu=hdulist.index_of(hdu))
                colnames = [c.lower() for c in tbl.colnames]
                if len(tbl.colnames) < 2:
                    continue
                # Identify wavelength column
                wave_col = None
                for candidate in ('wave', 'wavelength', 'lambda', 'lam', 'wav'):
                    for i, cn in enumerate(colnames):
                        if candidate in cn:
                            wave_col = tbl.colnames[i]
                            break
                    if wave_col:
                        break
                if wave_col is None:
                    wave_col = tbl.colnames[0]

                # Identify flux column
                flux_col = None
                for candidate in ('flux', 'flam', 'f_lambda', 'fll', 'counts', 'data'):
                    for i, cn in enumerate(colnames):
                        if candidate in cn:
                            flux_col = tbl.colnames[i]
                            break
                    if flux_col:
                        break
                if flux_col is None:
                    flux_col = tbl.colnames[1]

                wave = np.asarray(tbl[wave_col], dtype=float)
                flux = np.asarray(tbl[flux_col], dtype=float)

                # Check units from column metadata
                if waveunit is None:
                    wcol_obj = tbl[wave_col]
                    unit_str = ''
                    if hasattr(wcol_obj, 'unit') and wcol_obj.unit is not None:
                        unit_str = str(wcol_obj.unit).lower()
                    if 'nm' in unit_str or 'nanometer' in unit_str:
                        waveunit = 'nm'
                break

        # Fallback: IMAGE HDU with WCS
        if wave is None:
            primary = hdulist[0]
            if primary.data is not None and primary.data.ndim == 1:
                flux = np.asarray(primary.data, dtype=float)
                hdr = primary.header
                crval = hdr.get('CRVAL1', 1)
                cdelt = hdr.get('CDELT1', hdr.get('CD1_1', 1))
                crpix = hdr.get('CRPIX1', 1)
                n = len(flux)
                wave = crval + (np.arange(n) - (crpix - 1)) * cdelt
                if waveunit is None:
                    cunit = str(hdr.get('CUNIT1', 'Angstrom')).lower()
                    if 'nm' in cunit or 'nanometer' in cunit:
                        waveunit = 'nm'

        hdulist.close()

        if wave is None or flux is None:
            raise ValueError("Could not find wavelength/flux data in the FITS file. "
                             "Expected a BINTABLE with at least 2 columns or a 1D IMAGE HDU.")

        # Sort by wavelength
        order = np.argsort(wave)
        wave = wave[order]
        flux = flux[order]

        # Convert units
        if waveunit == 'nm':
            wave *= 10.0

        if fluxunit == 'ph':
            h_cgs = 6.62607015e-27
            c_cgs = 2.99792458e18
            E_photon = h_cgs * c_cgs / wave
            flux = flux * E_photon

        return wave, flux


class FilterManager:
    """
    Class for managing filters and normalizations.
    """
    def __init__(self, phot_system):
        self.phot_system = phot_system
    
    def get_filter_profile(self, band):
        """
        Returns the transmission profile of a filter.
        
        Parameters:
        - band: filter name
        
        Returns:
        - wavelength: wavelength array [Å]
        - transmission: transmission [0-1]
        """
        if band not in self.phot_system.band_filters:
            raise ValueError(f"Filter {band} not found")
        return self.phot_system.band_filters[band].T[0], self.phot_system.band_filters[band].T[1]
    
    def apply_filter(self, wavelength, flux, band, mag=None, system="Vega", typeSP="Fll"):
        """
        Applies a filter to a spectrum and normalizes to a magnitude.
        
        Parameters:
        - wavelength: spectrum wavelength array [Å]
        - flux: spectrum flux [erg/s/cm²/Å]
        - band: filter name
        - mag: target magnitude (None for no normalization)
        - system: photometric system ('Vega' or 'AB')
        - typeSP: flux type ('Fll' or 'PHll')
        
        Returns:
        - common_wave: common wavelength grid
        - filtered_flux: filtered and normalized flux
        - K: normalization factor
        """
        # Get filter profile
        filt_wave, filt_trans = self.get_filter_profile(band)
        
        # Interpolate spectrum and filter to a common grid
        min_wave = max(wavelength.min(), filt_wave.min())
        max_wave = min(wavelength.max(), filt_wave.max())
        common_wave = np.linspace(min_wave, max_wave, 1000)
        
        flux_interp = SEDModels.interpolate_spectrum(wavelength, common_wave, flux)
        trans_interp = SEDModels.interpolate_spectrum(filt_wave, common_wave, filt_trans)
        
        # Apply filter
        filtered_flux = flux_interp * trans_interp  # To be corrected, only works with integral 1?
        
        # Normalize if requested
        K = 1.0
        if mag is not None:
            # Get zero point
            _, zp_ph = self.phot_system.get_flux_zeropoint(band, system, typeSP)
            zp_ph_scaled = zp_ph * (2.512**(-mag))
            
            # Calculate normalization factor
            integral_flux = integrate.trapezoid(filtered_flux, common_wave)
            integral_trans = integrate.trapezoid(trans_interp, common_wave)
            
            K = (zp_ph_scaled * integral_trans) / integral_flux
            filtered_flux *= K 
        
        return common_wave, filtered_flux, K

# Plot for comparison
def plot_spectra_comparison(wave_coarse, flux_coarse, wave, flux, 
                           label_coarse, label, title=None, namepng=None):
    """
    Plots the coarse spectrum, interpolated fine spectrum, and original fine spectrum for comparison.
    
    Parameters:
    - wave_coarse: coarse wavelength grid
    - flux_coarse: coarse flux
    - wave: fine wavelength grid
    - flux: fine flux
    - label_coarse: label for coarse spectrum
    - label: label for fine spectrum
    - title: plot title
    - namepng: filename to save the plot (without extension)
    """
    
    flux_interpolated = SEDModels.interpolate_spectrum(wavelength=wave,
                                    target_wavelength=wave_coarse, flux=flux)
    
    # To ensure it works also for zeros
    mask = (flux_interpolated != 0) & (flux_coarse != 0)
    relative_diff = np.zeros_like(flux_interpolated)  # Initialize with zeros
    relative_diff[mask] = (flux_interpolated[mask] - flux_coarse[mask]) / flux_interpolated[mask] * 100
    
    mean_diff = np.mean(relative_diff)
    std_diff = np.std(relative_diff)
    
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Upper plot: Spectra ---
    ax[0].plot(wave_coarse, flux_coarse, 'o-', markersize=0, label=label_coarse, color='blue', alpha=0.5)
    ax[0].plot(wave_coarse, flux_interpolated, 's--', markersize=0, label=label, color='red', alpha=0.5)
    ax[0].set_ylabel("Flux density [erg cm$^{-2}$ s$^{-1}$ \u00c5$^{-1}$]")
    ax[0].legend()
    ax[0].set_title(title if title else "Spectra Comparison")
    ax[0].grid()
    
    # --- Lower plot: Relative Difference ---
    ax[1].plot(wave_coarse, relative_diff, 'o-', markersize=0, color='black',
              label=f"Mean: ({mean_diff:.2f}±{std_diff:.2f})%")
    ax[1].axhline(0, color='gray', linestyle='--', lw=1)
    ax[1].set_ylabel("Relative Diff (%)")
    ax[1].set_xlabel("Wavelength [\u00c5]")
    ax[1].legend()
    ax[1].grid()
    
    if namepng is not None:
        plt.savefig(f"{namepng}.png", dpi=300, bbox_inches='tight')
    plt.show()

# # # # # # # MORE # # # # # #
# ADD y, Y, Z filters
# properly compute magnitude conversions based on the new ZPs (in principle better to use the AB filters in AB and same for Vega), not clear especially for GAIA
# handle also Fnu spectra for the templates, for now we usually work with Fll
# # # # # # # # # # # # # # #