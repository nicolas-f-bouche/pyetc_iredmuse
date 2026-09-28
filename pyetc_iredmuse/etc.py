from astropy.modeling import functional_models
import logging
import os
import glob
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import fftconvolve

from astropy import constants
from astropy.table import Table
import astropy.units as u
from astropy.modeling.models import Sersic2D

from io import BytesIO

try:
    from skycalc_cli.skycalc import SkyModel as _SkyCLIModel
except ImportError:  # pragma: no cover
    _SkyCLIModel = None

from mpdaf.obj import Spectrum, WaveCoord, Image, moffat_image

from .specalib import PhotometricSystem, SEDModels, FilterManager

# Initialize photometric system, SED models and filter manager
phot_system = PhotometricSystem()
sed_models = SEDModels()
filter_manager = FilterManager(phot_system)

# # # # # # # # # # # # # # # 
# # # global parameters # # # 
# # # # # # # # # # # # # # # 

# speed of light and Planck constant in CGS units
C_cgs = constants.c.cgs.value
H_cgs = constants.h.cgs.value

# tolerance to check for wavelength range
tol_wave = 2

# number of fwhm to consider for line profile, all Spectrum object values outside put to zero
n_fwhm = 4

# number of wavelength points to for the PSF images
wave_grid = 5  

# saturation threshold in e-/ph/counts
threshold_sat = 50000  

# Mean MOS fiber-centering loss used when no explicit object displacement is supplied.
MOS_DEFAULT_CENTERING_EFF = 0.9

# default angstrom value for the SNR wave tolerance check
default_angstrom_edge = 2

# # # # # # # # # # # # # # #

__all__ = [
    'ETC',
    'sersic',
    'moffat',
    '_checkline',
    '_checkrange',
    '_checkobs',
    'get_seeing_fwhm',
    'lsf_resolution',
    'compute_sky',
    'mask_spectrum_edges',
    'mask_line_region',
    'mask_spectra_in_dict',
    'convolve_and_center',
    'plot_noise_components',
    'simulate_counts',
    'simulate_counts_vectorized',
    'snr_in_window',
]

class ETC:
    """ Generic class for Exposure Time Computation (ETC) """

    # Class-level PSF cache: key = (seeing, airmass, ins_name, spaxel_size, iq_beta, wave_tuple)
    _psf_cache = {}
    
    # Class-level source image cache: key = (type, spaxel_size, fwhm/reff, beta/n, uneven)
    _source_image_cache = {}
    
    # Class-level skycalc cache: key = (airmass, pwv, mss, ins, ch, lbda1, lbda2)
    _skycalc_cache = {}
    # Broad master sky cache: key = sky conditions only (no channel)
    # Stores (flux_arr, trans_arr, wmin_aa, wstep_aa) over the full WST range.
    _skycalc_broad_cache = {}
    # Broad download parameters (cover all WST channels: 3700–11000 Å)
    _BROAD_WMIN_NM  = 369.0   # nm  — small buffer below IFS blue (3700 Å)
    _BROAD_WMAX_NM  = 1101.0  # nm  — small buffer above MOS LR red (~11000 Å)
    _BROAD_WDELTA_NM = 0.005  # nm  = 0.05 Å/point; needed for MOS-HR LSF convolutions

    _transdata = {}

    WST = dict(effective_area_MOS=93.57,  # mean of median and weighted mean of the ICD document
                    effective_area_IFS=92.03,  # minimum of the ICD document
                    diameter=12.0,  # primary diameter
                    desc='Based on Design doc: WST-00006_3 Telescope optical design report',
                    # Link: https://stfc365.sharepoint.com/sites/Wide-FieldSpectroscopicTelescope/_layouts/15/DocIdRedir.aspx?ID=7ZWYDD3PV4SU-175398458-2018
                    version='03/03/2026',
                    iq_fwhm_ins={
                        'ifs': 0.1,
                        # average FWHM on 95% FOV as also used in the WST_IFS_Tradeoff_Matrix. Link: https://stfc365.sharepoint.com/sites/Wide-FieldSpectroscopicTelescope/_layouts/15/DocIdRedir.aspx?ID=7ZWYDD3PV4SU-175398458-2008
                        'mos': 0.1875,
                    }
                    )
    VLT = dict(effective_area_IFS=48.5,  # VLT effective area in m^2
                    effective_area_MOS=None,  # VLT effective area in m^2
                    diameter=8.0,  # primary diameter
                    desc='Based on Prelim Concept',
                    version='26/06/2026',
                    iq_fwhm_ins={
                        'ifs': 0.1,
                        # assumed in the Prelim Concept, this probably considers also the detector (charge diffusion)
                    }
                    )

    def __init__(self, log=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False

    def set_logging(self, log):
        """ Change logging value

        Parameters
        ----------
        log : str
             desired log mode "DEBUG","INFO","WARNING","ERROR"

        """
        self.logger.setLevel(log)

    def _info(self, ins_names):
        """ print detailed information

        Parameters
        ----------
        ins_names : list of str
               list of instrument names (e.g ['ifs','moslr'])

        """  
        if ('desc' in self.tel) and ('version' in self.tel):
            self.logger.info('Telescope %s version %s', self.tel['desc'],self.tel['version'])      
        if self.tel['effective_area_MOS'] is not None:
            self.logger.info('Diameter: %.2f m Eff. Area MOS: %.1f m2 Eff. Area IFS: %.1f m2', self.tel['diameter'],self.tel['effective_area_MOS'], self.tel['effective_area_IFS'])
        else:
            self.logger.info('Diameter: %.2f m Eff. Area IFS: %.1f m2', self.tel['diameter'], self.tel['effective_area_IFS'])

        for ins_name in ins_names:
            insfam = getattr(self, ins_name)
            for chan in insfam['channels']:
                ins = insfam[chan]
                self.logger.info('%s type %s Channel %s', ins_name.upper(), ins['type'], chan)
                self.logger.info('\t Throughput model: %s', ins['desc'])
                self.logger.info('\t Configuration version: %s', ins['version'])
                self.logger.info('\t Spaxel size: %.2f arcsec Image Quality tel and ins fwhm: %.2f and %.2f arcsec beta: %.2f ', ins['spaxel_size'], ins['iq_fwhm_tel'],ins['iq_fwhm_ins'], ins['iq_beta'])
                if 'aperture' in ins.keys():
                    self.logger.info('\t Fiber aperture: %.1f arcsec', ins['aperture'])
                self.logger.info('\t Wavelength range %s A step %.2f A LSF %.1f pix Npix %d', ins['instrans'].get_range(),
                                  ins['instrans'].get_step(), ins['lsfpix'], ins['instrans'].shape[0])
                self.logger.info('\t Instrument transmission peak %.2f at %.0f - min %.2f at %.0f',
                                  ins['instrans'].data.max(), ins['instrans'].wave.coord(ins['instrans'].data.argmax()),
                                  ins['instrans'].data.min(), ins['instrans'].wave.coord(ins['instrans'].data.argmin()))
                self.logger.info('\t Detector RON %.1f e- Dark %.1f e-/h', ins['ron'],ins['dcurrent'])


    def set_obs(self, obs):
        """save obs dictionary to self

        Parameters
        ----------
        obs : dict
            dictionary of observation parameters

        """

        self.obs = obs
        return

    # method to build the full observation setup from input dictionary
    def build_obs_full(self, fo):
        """Build observation parameters and setup from input dictionary

        Parameters
        ----------
        fo : dict
            Dictionary containing observation parameters

        Returns
        -------
        tuple
            (conf, obs, spec, ima, spec_input)
            - conf: instrument configuration
            - obs: observation parameters
            - spec: processed spectrum
            - ima: source image (if resolved)
            - spec_input: input spectrum
        """
        # Get instrument configuration
        insfam = getattr(self, fo["INS"])
        conf = insfam[fo["CH"]]

        # first check on value of the SED type
        if fo['Obj_SED'] not in ('template', 'pl', 'bb', 'line', 'uniform', 'upload'):
            raise ValueError(f"Invalid SED type: {fo['Obj_SED']} \n Allowed values are 'template', 'pl', 'bb', 'line', 'uniform', 'upload'")
        
        # Determine spectral type
        if fo['Obj_SED'] == 'line':
            dummy_type = 'line'
        elif fo['Obj_SED'] in ('template', 'pl', 'bb', 'uniform', 'upload'):
            dummy_type = 'cont'

        # check coadding for IFS, should be odd integer, now also the even are allowed
        #if fo["CH"] == 'ifs' and fo["COADD_XY"] % 2 == 0:
        #        raise ValueError("the spatial coadding in the IFS must be an odd integer for a symmetric aperture.")

        inter_dict = {}

        obs = dict(
            INS=fo.get("INS", None),
            CH=fo.get("CH", None),

            seeing=fo.get("SEE", None),
            seeing_input=fo.get("SEE", None),   # original user value, never overridden
            ndit=fo.get("NDIT", None),
            dit=fo.get("DIT", None),
            spec_type=dummy_type,
            spec_specific=fo.get('Obj_SED', None),

            ima_type=fo.get('Obj_Spat_Dis', None),

            ima_coadd=fo.get("COADD_XY", None),

            skycalc=fo.get("SKYCALC", None),
            airmass=fo.get("AM", None),
            pwv=fo.get("PWV", None),
            fli=fo.get("FLI", None),
            moon_target_sep=fo.get("MOON_SEP", 45),

            wave_line_center=fo.get('SEL_CWAV', None),
            wave_line_fwhm=fo.get('SEL_FWHM', None),
            wave_line_flux=fo.get('SEL_FLUX', None),

            snr=fo.get("SNR", None),
            snr_wave=fo.get("Lam_Ref", None),
            snr_range=fo.get("SNR_RANGE", False),
            lam_win1=fo.get("LAM_WIN1", None),
            lam_win2=fo.get("LAM_WIN2", None),

            disp=fo.get('OBJ_FIB_DISP', None),

            spbin=fo.get("COADD_WL", None),

            band=fo.get("MAG_FIL", None),
            mag=fo.get("OBJ_MAG", None),
            syst=fo.get("MAG_SYS", None),

            redshift=fo.get("Z", None),

            sed_name=fo.get("SED_Name", None),

            bb_temp=fo.get("BB_Temp", None),
            pl_index=fo.get("PL_Index", None),

            ima=fo.get("IMA", None),

            ima_fwhm=fo.get("IMA_FWHM", None),
            ima_beta=fo.get("IMA_BETA", None),

            sersic_reff=fo.get("Sersic_Reff", None),
            sersic_ind=fo.get("Sersic_Ind", None),

            glao=fo.get("GLAO", False),

            upload_wave=None,
            upload_flux=None
        )

        if obs['disp'] is not None:
            try:
                obs['disp'] = float(obs['disp'])
            except (TypeError, ValueError):
                raise ValueError("OBJ_FIB_DISP must be a non-negative number or None")
            if not np.isfinite(obs['disp']) or obs['disp'] < 0:
                raise ValueError("OBJ_FIB_DISP must be a non-negative number or None")

        # GLAO handling: override seeing and PSF beta
        if obs['glao']:
            ins_name = fo.get("INS", "")
            if ins_name in ('moslr', 'moshr'):
                # MOS + GLAO: use the standard seeing formula (Kolb/VK) but force
                # seeing to the value in self.glao['mos_seeing'] (set in wst.py, default 0.8 arcsec).
                # The user-supplied seeing value is ignored.
                # Moffat beta stays at the instrument default (2.80).
                mos_seeing = getattr(self, 'glao', {}).get('mos_seeing', 0.8)
                obs['seeing'] = mos_seeing
                obs['glao_seeing_note'] = (
                    f"MOS+GLAO: user seeing ({obs['seeing_input']}\") ignored; "
                    f"fixed to {mos_seeing}\" (Paranal median, zenith, 5000 Å); beta=2.80"
                )
                self.logger.info("GLAO active (MOS): user seeing ignored, fixed to %.2f arcsec "
                                 "(Paranal median at zenith, 5000 Å); Moffat beta = 2.80", mos_seeing)
            else:
                # IFS + GLAO: the GLAO polynomial formula is used (see get_seeing_fwhm),
                # which is a function of wavelength and airmass only.
                # The user-supplied seeing value is completely ignored.
                # Moffat beta read from self.glao['ifs_beta'] (set in wst.py), fallback 2.5.
                obs['glao_iq_beta'] = getattr(self, 'glao', {}).get('ifs_beta', 2.5)
                obs['glao_seeing_note'] = (
                    f"IFS+GLAO: user seeing ({obs['seeing_input']}\") ignored; "
                    "IQ from GLAO polynomial IQ(λ,AM); beta=2.5"
                )
                self.logger.info("GLAO active (IFS): user seeing ignored, wavelength-dependent "
                                 "GLAO IQ formula applied; Moffat beta = 2.5")

        # Read spectrum from file if upload SED type
        if fo['Obj_SED'] == 'upload':
            fpath = fo.get('UPLOAD_FILE', None)
            if fpath is None:
                raise ValueError("Upload SED type requires 'UPLOAD_FILE' (path to spectrum file).")
            if not os.path.isfile(fpath):
                raise ValueError(f"Spectrum file not found: {fpath}")
            if fpath.lower().endswith(('.fits', '.fit')):
                with open(fpath, 'rb') as f:
                    content = f.read()
            else:
                with open(fpath, 'r') as f:
                    content = f.read()
            obs['upload_wave'], obs['upload_flux'] = sed_models.parse_uploaded_spectrum(content)

        self.set_obs(obs)

        # we compute the moon keywords here given the the FLI values, allowed values are 
        # 0 (darksky), 0.5 (greysky), 1 (brightsky), for skycalc = False
        if fo['FLI'] == 0:
            moon = 'darksky'
        elif fo['FLI'] == 0.5:
            moon = 'greysky'
        elif fo['FLI'] == 1:
            moon = 'brightsky'
        else:
            moon = None

        # add moon to obs dictionary
        obs['moon'] = moon

        # we put here some other checks on values of the variables
        if fo['COADD_WL'] < 1:
            raise ValueError("the spectral coadding must be a positive integer.")
        if fo['Obj_Spat_Dis'] not in ('sb', 'resolved', 'ps'):
            raise ValueError(f"Invalid spatial distribution: {fo['Obj_Spat_Dis']} \n Allowed values are 'sb', 'resolved', 'ps'")
        if fo['Obj_Spat_Dis'] == 'resolved' and fo['IMA'] not in ('moffat', 'sersic', None):
            raise ValueError(f"Invalid image type (for resolved case): {fo['IMA']} \n Allowed values are 'moffat', 'sersic'")       
        if fo['Obj_SED'] != 'line' and fo.get('OBJ_MAG') is not None and fo['MAG_SYS'] not in ('AB', 'Vega'):
            raise ValueError(f"Invalid MAG_SYS: {fo['MAG_SYS']} \n Allowed values are 'AB', 'Vega'")

        # we compute the sky configuration/or take it from the static files, 
        # the sky spectrum is already convoluted for the LSF
        if not isinstance(obs["skycalc"], bool):
            raise ValueError('SKYCALC must be True or False')
        obs["skyemi"], obs["skyabs"] = self.get_sky()

        # if the spectrum is line, we override the lam_ref with the center of the line in wave_center
        if fo['Obj_SED'] == 'line':
            obs['snr_wave'] = obs['wave_line_center']
            self.logger.debug(f"Override snr_wave with wave_line_center: {obs['snr_wave']}")

        # Get spectrum
        spec_input, spec = self.get_spec()

        # With no explicit fiber displacement, preserve the mean MOS centering
        # loss used by the web interface. An explicit displacement is handled
        # geometrically by mos_fiber_aperture/mos_fiber_aperture_batch.
        if conf['type'] == 'MOS' and obs['disp'] is None:
            spec = spec * MOS_DEFAULT_CENTERING_EFF

        # Handle resolved source image
        ima = None
        if fo['Obj_Spat_Dis'] == 'resolved':
            # add the uneven to use also even coadding for the image
            # when coadd is 'best', we default to uneven=1 (odd); the actual
            # coadd will be resolved later in the SNR/time computation
            coadd_xy = fo.get('COADD_XY')
            if isinstance(coadd_xy, (int, float)) and coadd_xy % 2 == 0:
                uneven = 0
            else:
                uneven = 1
            dima = {
                'type': obs["ima"],
                'fwhm': obs["ima_fwhm"],
                'beta': obs["ima_beta"],
                'n': obs["sersic_ind"],
                'reff': obs["sersic_reff"],
                'uneven': uneven
            }
            ima = self.get_image(conf, dima)

        return conf, obs, spec, ima, spec_input

    def get_data(self, obj, chan, name):
        """ retrieve instrument data from the associated setup files

        Parameters
        ----------
        obj : ETC class
            instrument class (e.g. etc.ifs)
        chan : str
            channel name (eg 'red')
        name : str
            instrument name (eg 'ifs')
        skydir : str
            directory path where the sky fits file can be found
        transdir : str
            directory path where the transmission fits file can be found

        """
        ins = obj[chan]

        # Sky emission and atmospheric transmission
        flist = glob.glob(os.path.join(self.SKYDIR, "*.fits"))
        flist.sort()
        ins['sky'] = []
        moons = []
        for fname in flist:
            f = os.path.basename(fname).split('_')
            moon = f[0]
            moons.append(moon)
            airmass = float(f[1])
            pwv = float(f[2][:-5])
            d = dict(moon=moon, airmass=airmass, pwv=pwv)

            tab = Table.read(fname, unit_parse_strict="silent")

            start = tab['lam'][0] * 10
            step = (tab['lam'][1] - tab['lam'][0]) * 10
            wave = WaveCoord(cdelt=step, crval=start, cunit=u.angstrom)

            d_emi = Spectrum(data=tab['flux'], wave=wave)
            d_abs = Spectrum(data=tab['trans'], wave=wave)

            d['emi'] = d_emi.resample(ins['dlbda'], start=ins['lbda1'],
                                      shape=int((ins['lbda2'] - ins['lbda1']) / ins['dlbda']) + 1)
            d['abs'] = d_abs.resample(ins['dlbda'], start=ins['lbda1'],
                                      shape=int((ins['lbda2'] - ins['lbda1']) / ins['dlbda']) + 1)
            ins['sky'].append(d)

        # all the transmission curves
        self.logger.info(f"Reading {name} {chan} into self._transdata")
        filename = glob.glob(os.path.join(self.TRANSDIR, f'{name}_{chan}_noatm.fits'))[0]
        trans = Table.read(os.path.join(self.TRANSDIR, filename), unit_parse_strict="silent")
        self._transdata[chan] = trans

        # # # Not needed anymore from Olga's throughput files
        # We compute the total transmision (excluded atmosphere)
        # cc = trans.colnames[1:-1]
        # all = np.prod([trans[c] for c in cc], axis=0)
        # trans['trans'] = all

        # We compute the instrument only transmission (exluded CCD and telescope, all the other columns)
        trans['only_inst'] = trans['total'] / (trans['detector_QE'] * trans['telescope'])

        skywave = ins['sky'][0]['emi'].wave
        skycoord = skywave.coord()

        ins['instrans'] = Spectrum(
            data=np.interp(skycoord, trans['wave'] * 10, trans['total']), wave=skywave)
        ins['telescope'] = Spectrum(
            data=np.interp(skycoord, trans['wave'] * 10, trans['telescope']), wave=skywave)
        ins['QE'] = Spectrum(
            data=np.interp(skycoord, trans['wave'] * 10, trans['detector_QE']), wave=skywave)
        ins['total_instrumental'] = Spectrum(
            data=np.interp(skycoord, trans['wave'] * 10, trans['only_inst']), wave=skywave)

        ins['skys'] = list(set(moons))
        ins['wave'] = ins['instrans'].wave
        ins['chan'] = chan
        ins['name'] = name
        ins['advice'] = 'Beware if you change the static sky files and/or transmission curves, even by a little marging, it is good to have in the files: lambda1_sky < lambda1_trans < lambda1_config, same for dlambda and opposite for lambda2 (they are all trimmed and resampled according to the configuration dictionary, this is done in order to avoid edges problems)'
        return

    def get_sky(self, obs=None):
        """
        Return sky emission and transmission spectra.

        Parameters
        ----------
        obs : dict or None
            Observation parameters. If None, uses self.obs.

        Returns
        -------
        tuple of MPDAF spectra
            emission and absorption sky spectra

        """
        if obs is None:
            obs = self.obs
        
        insfam = getattr(self, obs["INS"])
        conf = insfam[obs["CH"]]

        static = not bool(obs.get("skycalc", False))
        airmass = obs.get('airmass')
        moon = obs.get('moon')

        if static:
            # look up in the loaded static sky files
            pwv = obs.get('pwv')
            # Expose effective values for debug/reporting consistency.
            obs['airmass_effective'] = airmass
            obs['pwv_effective'] = pwv
            obs['moon_target_sep_effective'] = obs.get('moon_target_sep')
            obs['moon_alt_effective'] = obs.get('moon_alt')
            obs['incl_moon_effective'] = None
            obs['moon_sun_sep_effective'] = None
            obs['skycalc_adjustments'] = []
            available_airmass = set(sky['airmass'] for sky in conf['sky'])
            available_moon = set(sky['moon'] for sky in conf['sky'])
            available_pwv = set(sky['pwv'] for sky in conf['sky'])
            for sky in conf['sky']:
                if np.isclose(sky['airmass'], airmass) and (sky['moon'] == moon) and np.isclose(sky['pwv'], pwv):
                    return sky['emi'], sky['abs']
            raise ValueError(f"moon {moon} airmass {airmass} pwv {pwv} not found in loaded sky configurations. Available airmass: {sorted(available_airmass)}, available moon: {sorted(available_moon)}, available pwv: {sorted(available_pwv)}")
        else:
            # compute on the fly with skycalc
            adjustments = []

            # SkyCalc expects airmass in a physical range; keep the request valid.
            try:
                airmass = float(airmass)
            except (TypeError, ValueError):
                adjustments.append(f"AM not valid ({airmass}); using default 1.0")
                airmass = 1.0
            airmass_eff = float(np.clip(airmass, 1.0, 3.0))
            if not np.isclose(airmass_eff, airmass):
                adjustments.append(f"AM {airmass:.3f} out of range [1.0, 3.0]; using {airmass_eff:.3f}")
            airmass = airmass_eff

            fli = obs.get('fli')
            if fli is None:
                adjustments.append("FLI missing; using default 0.0")
                fli = 0.0
            try:
                fli = float(fli)
            except (TypeError, ValueError):
                adjustments.append(f"FLI not valid ({fli}); using default 0.0")
                fli = 0.0
            fli_eff = float(np.clip(fli, 0.0, 1.0))
            if not np.isclose(fli_eff, fli):
                adjustments.append(f"FLI {fli:.3f} out of range [0, 1]; using {fli_eff:.3f}")
            fli = fli_eff

            theta_rad = np.arccos(1 - 2 * fli)  # result in radians
            mss = np.degrees(theta_rad)  # convert to degrees
            moon_target_sep = obs.get('moon_target_sep', 45)
            try:
                moon_target_sep = float(moon_target_sep)
            except (TypeError, ValueError):
                adjustments.append(f"MOON_SEP not valid ({moon_target_sep}); using default 45")
                moon_target_sep = 45.0
            moon_target_sep_eff = float(np.clip(moon_target_sep, 0.0, 180.0))
            if not np.isclose(moon_target_sep_eff, moon_target_sep):
                adjustments.append(
                    f"MOON_SEP {moon_target_sep:.3f} out of range [0, 180]; using {moon_target_sep_eff:.3f}"
                )
            moon_target_sep = moon_target_sep_eff

            pwv = obs.get('pwv', 1.0)
            try:
                pwv = float(pwv)
            except (TypeError, ValueError):
                adjustments.append(f"PWV not valid ({pwv}); using default 1.0")
                pwv = 1.0
            allowed_pwv = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0]
            closest_value = min(allowed_pwv, key=lambda v: np.abs(v - pwv))
            if pwv not in allowed_pwv:
                self.logger.warning(f"PWV value not allowed, assigned the closest one: {pwv} → {closest_value}")
                adjustments.append(f"PWV {pwv:.3f} not on allowed grid; using {closest_value}")
                pwv = closest_value

            # Enforce documented moon-geometry constraints:
            # |z-zmoon| <= rho <= z+zmoon, with z=90-alt(target), zmoon=90-altmoon.
            # moon_alt is fixed at 45° (not a user-facing parameter).
            moon_alt = 45.0
            z_target = float(np.degrees(np.arccos(np.clip(1.0 / max(float(airmass), 1.0), -1.0, 1.0))))
            z_moon = 90.0 - moon_alt
            rho_min = abs(z_target - z_moon)
            rho_max = min(180.0, z_target + z_moon)
            clipped_sep = float(np.clip(moon_target_sep, rho_min, rho_max))
            if not np.isclose(clipped_sep, moon_target_sep):
                self.logger.warning(
                    "MOON_SEP %.2f outside valid range [%.3f, %.3f] for AM=%.3f and moon_alt=%.2f; using %.3f",
                    moon_target_sep, rho_min, rho_max, airmass, moon_alt, clipped_sep
                )
                adjustments.append(
                    f"MOON_SEP {moon_target_sep:.3f} not geometrically valid for AM={airmass:.3f}; using {clipped_sep:.3f}"
                )
            moon_target_sep = clipped_sep

            # For (near) new moon, disable scattered moonlight component directly.
            # This follows SkyCalc semantics and avoids inconsistent edge cases.
            include_moon = 'N' if fli <= 1e-3 else 'Y'

            # Keep track of the effective values really used by SkyCalc.
            obs['airmass_effective'] = airmass
            obs['pwv_effective'] = pwv
            obs['moon_target_sep_effective'] = moon_target_sep
            obs['moon_alt_effective'] = moon_alt
            obs['incl_moon_effective'] = include_moon
            obs['moon_sun_sep_effective'] = mss
            obs['skycalc_adjustments'] = adjustments

            skycalc_params = {
                'msolflux': 130.0,
                'observatory': 'paranal',
                'airmass': airmass,
                'pwv': pwv,
                'incl_moon': include_moon,
                'moon_sun_sep': mss,
                'moon_alt': moon_alt,
                'moon_target_sep': moon_target_sep,
            }

            # Build cache key from effective SkyCalc moon params.
            cache_key = (
                round(airmass, 4),
                pwv,
                round(mss, 2),
                round(moon_target_sep, 2),
                round(moon_alt, 2),
                include_moon,
                obs['INS'],
                obs['CH'],
                conf['lbda1'],
                conf['lbda2'],
                conf['dlbda']
            )

            # Check per-channel cache first
            if cache_key in ETC._skycalc_cache:
                return ETC._skycalc_cache[cache_key]

            # Build broad_key (sky conditions only, no channel geometry)
            broad_key = (
                round(airmass, 4),
                pwv,
                round(mss, 2),
                round(moon_target_sep, 2),
                round(moon_alt, 2),
                include_moon,
            )

            if broad_key not in ETC._skycalc_broad_cache:
                # Download broad master sky covering the full WST wavelength range
                broad_params = dict(skycalc_params)
                broad_params['wmin']       = ETC._BROAD_WMIN_NM
                broad_params['wmax']       = ETC._BROAD_WMAX_NM
                broad_params['wdelta']     = ETC._BROAD_WDELTA_NM
                broad_params['wgrid_mode'] = 'fixed_wavelength_step'

                _skm = _SkyCLIModel()
                _skm.stop_on_errors_and_exceptions = False
                _skm.callwith(broad_params)
                if _skm.data is None:
                    # Last-resort fallback: disable moon component and retry.
                    adjustments.append(
                        "SkyCalc rejected parameters; retried with incl_moon='N' for robustness"
                    )
                    obs['skycalc_adjustments'] = adjustments
                    broad_params['incl_moon'] = 'N'
                    broad_params['moon_target_sep'] = 45.0
                    broad_params['moon_alt'] = 45.0
                    _skm = _SkyCLIModel()
                    _skm.stop_on_errors_and_exceptions = False
                    _skm.callwith(broad_params)
                    if _skm.data is None:
                        raise RuntimeError(
                            f"SkyCalc rejected request for AM={airmass}, FLI={fli}, "
                            f"MOON_SEP={moon_target_sep}, moon_alt={moon_alt:.2f}."
                        )
                    obs['incl_moon_effective'] = 'N'
                    obs['moon_target_sep_effective'] = 45.0
                    obs['moon_alt_effective'] = 45.0
                    obs['skycalc_adjustments'] = adjustments

                tab = Table.read(BytesIO(_skm.data), format='fits')
                if tab['lam'].unit is None:
                    tab['lam'].unit = u.nm

                wmin_aa  = float(tab['lam'][0]) * 10
                wstep_aa = float(tab['lam'][1] - tab['lam'][0]) * 10
                ETC._skycalc_broad_cache[broad_key] = (
                    np.array(tab['flux']),
                    np.array(tab['trans']),
                    wmin_aa,
                    wstep_aa,
                )

            flux_arr, trans_arr, wmin_aa, wstep_aa = ETC._skycalc_broad_cache[broad_key]

            # Build master Spectrum objects from the cached broad arrays
            master_wave  = WaveCoord(cdelt=wstep_aa, crval=wmin_aa, cunit=u.angstrom)
            d_emi_master = Spectrum(data=flux_arr.copy(), wave=master_wave)
            d_abs_master = Spectrum(data=trans_arr.copy(), wave=master_wave)

            # Scale lsfpix from channel pixels to master pixels
            lsf_fwhm_aa   = conf['lsfpix'] * conf['dlbda']   # LSF FWHM in Å
            lsfpix_master = max(0.5, lsf_fwhm_aa / wstep_aa)

            d_emi_lsfpix = d_emi_master.filter(width=lsfpix_master)
            shape = int((conf['lbda2'] - conf['lbda1']) / conf['dlbda']) + 1
            d_emi_lsfpix = d_emi_lsfpix.resample(conf['dlbda'], start=conf['lbda1'], shape=shape)

            d_abs = d_abs_master.resample(conf['dlbda'], start=conf['lbda1'], shape=shape)

            # Cache per-channel result
            ETC._skycalc_cache[cache_key] = (d_emi_lsfpix, d_abs)

            return d_emi_lsfpix, d_abs

    def get_spec(self, obs=None):
            """Get and process spectrum based on input parameters
        
            Parameters
            ----------
            obs : dict or None
                Observation parameters. If None, uses self.obs.
            
            Returns
            -------
            tuple
                (spec_raw, spec_cut)
                - spec_raw: Original spectrum
                - spec_cut: Processed and trimmed spectrum
            """

            if obs is None:
                obs = self.obs

            insfam = getattr(self, obs["INS"])
            conf = insfam[obs["CH"]]

            lstep = conf['instrans'].get_step()
            l1, l2 = conf['instrans'].get_start(), conf['instrans'].get_end()

            if obs['spec_specific'] == 'template':
                name, def_wave, flux = sed_models.template(f"{obs['sed_name']}.dat")
                redshift = obs['redshift']
                band = obs['band']
                mag = obs['mag']
                syst = obs['syst']

                mag, syst = phot_system.auto_conversion(mag, band, syst)

                # Redshift correction
                def_wave *= (1 + redshift)

                # Check range
                _checkrange(def_wave, l1, l2)

                _, _, K = filter_manager.apply_filter(def_wave, flux, band, mag, syst)

            elif obs['spec_specific'] == 'bb':
                def_wave = np.linspace(100, 30000, 10000)

                # Redshift correction
                redshift = obs['redshift']
                def_wave *= (1 + redshift)

                tmp = obs['bb_temp']
                band = obs['band']
                mag = obs['mag']
                syst = obs['syst']

                flux = sed_models.blackbody(def_wave, tmp)
                mag, syst = phot_system.auto_conversion(mag, band, syst)
                _, _, K = filter_manager.apply_filter(def_wave, flux, band, mag, syst)

            elif obs['spec_specific'] == 'pl':
                def_wave = np.linspace(100, 30000, 10000)

                # Redshift correction
                redshift = obs['redshift']
                def_wave *= (1 + redshift)

                indpl = obs['pl_index']
                band = obs['band']
                mag = obs['mag']
                syst = obs['syst']

                flux = sed_models.powerlaw(def_wave, indpl)
                mag, syst = phot_system.auto_conversion(mag, band, syst)
                _, _, K = filter_manager.apply_filter(def_wave, flux, band, mag, syst)

            elif obs['spec_specific'] == 'uniform':
                def_wave = np.linspace(100, 30000, 10000)

                # Redshift correction
                redshift = obs['redshift']
                def_wave *= (1 + redshift)

                band = obs['band']
                mag = obs['mag']
                syst = obs['syst']

                # Uniform spectrum = powerlaw with index 0
                flux = sed_models.powerlaw(def_wave, 0.0)
                mag, syst = phot_system.auto_conversion(mag, band, syst)
                _, _, K = filter_manager.apply_filter(def_wave, flux, band, mag, syst)

            elif obs['spec_specific'] == 'line':
                def_wave = np.linspace(100, 30000, 10000)
                center = obs['wave_line_center']
                fwhm = obs['wave_line_fwhm']

                _checkline(center, fwhm, l1, l2)
                tot_flux = obs['wave_line_flux']
                flux = sed_models.gaussian_line(def_wave, center, tot_flux, fwhm)
                K = 1

            elif obs['spec_specific'] == 'upload':
                def_wave = obs['upload_wave']
                flux = obs['upload_flux']
                if def_wave is None or flux is None:
                    raise ValueError("Upload SED type requires 'UPLOAD_FILE' (path to spectrum file).")
                def_wave = np.asarray(def_wave, dtype=float)
                flux = np.asarray(flux, dtype=float)
                # Normalize to magnitude if mag is provided, otherwise use as-is
                if obs['mag'] is not None:
                    band = obs['band']
                    mag = obs['mag']
                    syst = obs['syst']
                    mag, syst = phot_system.auto_conversion(mag, band, syst)
                    _, _, K = filter_manager.apply_filter(def_wave, flux, band, mag, syst)
                else:
                    K = 1
                
            # Put wave and flux*K in a MPDAF object
            spec_raw = Spectrum(data=flux*K, wave=WaveCoord(cdelt=def_wave[1]-def_wave[0], 
                                                    crval=def_wave[0]))

            # Resample
            #rspec = spec_raw.resample(lstep, start=l1)
            #spec_cut = rspec.subspec(lmin=l1, lmax=l2)

            # # # fastest resampling 
            npts = conf['instrans'].shape[0]
            target_wave = np.linspace(l1, l1 + (npts - 1) * lstep, npts)
            resampled_flux = np.interp(target_wave, def_wave, flux * K,
                                       left=0.0, right=0.0)
            spec_cut = Spectrum(data=resampled_flux, wave=WaveCoord(cdelt=lstep, crval=l1))
            
            return spec_raw, spec_cut

    def get_image(self, ins, dima):
        """ compute source image from the model parameters

         Parameters
         ----------
         ins : dict
             instrument (eg self.ifs['blue'] or self.moslr['red'])
         dima : dict
             dictionary of parameters describing the source spectrum

         Returns
         -------
         MPDAF image
             image of the source

         """

        # Build cache key based on image type and parameters
        if dima['type'] == 'moffat':
            cache_key = ('moffat', ins['spaxel_size'], round(dima['fwhm'], 6), 
                        round(dima['beta'], 6), dima.get('uneven', 0))
        elif dima['type'] == 'sersic':
            cache_key = ('sersic', ins['spaxel_size'], round(dima['reff'], 6),
                        round(dima['n'], 6), dima.get('uneven', 0))
        else:
            raise ValueError(f"Unknown image type {dima['type']}")
        
        if cache_key in ETC._source_image_cache:
            return ETC._source_image_cache[cache_key]
        
        if dima['type'] == 'moffat':
            ima = moffat(ins['spaxel_size'], dima['fwhm'], dima['beta'], uneven=dima.get('uneven', 0))
        elif dima['type'] == 'sersic':
            ima = sersic(ins['spaxel_size'], dima['reff'], dima['n'], uneven=dima.get('uneven', 0))
        
        ETC._source_image_cache[cache_key] = ima
        return ima

    # PSF images at a specific wavelengths
    def get_image_psf(self, ins, wave, uneven=1):
        """Compute PSF image(s) for one or more wavelengths.

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        wave : float or array-like
            wavelength(s) in Angstrom
        uneven : int
            if 1 odd-sized image (centered on pixel), 
            if 0 even-sized (centered between pixels) (Default value = 1)

        Returns
        -------
        tuple
            ima : MPDAF image or list of images
                PSF image(s)
            iq : float or np.ndarray
                FWHM(s) used for PSF(s)
        """
        glao = self.obs.get('glao', False)
        use_glao_formula = glao and ins.get('type') == 'IFS'
        iq_beta_eff = self.obs.get('glao_iq_beta', ins['iq_beta']) if use_glao_formula else ins['iq_beta']
        wave_tuple = tuple(np.atleast_1d(wave))
        # seeing is not used by the GLAO formula (IFS) so it may legitimately be None
        seeing_key = round(self.obs['seeing'], 4) if self.obs['seeing'] is not None else None
        cache_key = (
            seeing_key,
            round(self.obs['airmass'], 4),
            ins['name'], ins['spaxel_size'], iq_beta_eff,
            wave_tuple, uneven, use_glao_formula
        )
        
        if cache_key in ETC._psf_cache:
            return ETC._psf_cache[cache_key]
        
        iq, _ = get_seeing_fwhm(
            self.obs['seeing'],
            self.obs['airmass'],
            wave,
            self.tel['diameter'],
            ins['iq_fwhm_tel'],
            ins['iq_fwhm_ins'],
            glao=use_glao_formula
        )

        iq_arr = np.atleast_1d(iq)
        if iq_arr.size == 1:
            ima = moffat(ins['spaxel_size'], float(iq_arr[0]), iq_beta_eff, uneven=uneven)
            ima.data /= ima.data.sum()
            ima.oversamp = 10
            result = ima
        else:
            ima_arr = []
            for val in iq_arr:
                ima = moffat(ins['spaxel_size'], float(val), iq_beta_eff, uneven=uneven)
                ima.data /= ima.data.sum()
                ima.oversamp = 10
                ima_arr.append(ima)
            result = ima_arr
        
        ETC._psf_cache[cache_key] = result
        return result

    @classmethod
    def clear_psf_cache(cls):
        """Clear the PSF cache to free memory."""
        cls._psf_cache.clear()
    
    @classmethod
    def clear_source_image_cache(cls):
        """Clear the source image cache to free memory."""
        cls._source_image_cache.clear()
    
    @classmethod
    def clear_skycalc_cache(cls):
        """Clear the skycalc cache to free memory."""
        cls._skycalc_cache.clear()
    
    @classmethod
    def clear_all_caches(cls):
        """Clear all ETC caches to free memory."""
        cls._psf_cache.clear()
        cls._source_image_cache.clear()
        cls._skycalc_cache.clear()

    # IFS function for the fraction of flux collected in peak spaxel and in NxN region
    # added the difference between odd/even N
    # added fix padding images for large coadding which exceed image boundaries
    def ifs_spaxel_aperture(self, ins, ima, N=3):
        """
        Compute the fraction of flux collected in the central spaxel and in a centered NxN region.

        Parameters
        ----------
        ins : dict
            Instrument dictionary, must contain 'spaxel_size'.
        ima : MPDAF Image
            Normalized image (ima.data.sum() == 1), with .oversamp attribute.
        N : int
            Side length of the NxN region (in spaxel, default 3).

        Returns
        -------
        tuple
            (flux_central_spaxel, flux_NxN)
        """
        oversamp = ima.oversamp
        ny, nx = ima.data.shape
        cy, cx = np.unravel_index(np.argmax(ima.data), ima.data.shape)

        # Calculate NxN aperture extent
        half_N = (N * oversamp) // 2
        
        # Determine aperture boundaries (before any padding)
        if N % 2 == 1:  # Odd
            ymin_N = cy - half_N
            ymax_N = cy + half_N + (N * oversamp % 2)
            xmin_N = cx - half_N
            xmax_N = cx + half_N + (N * oversamp % 2)
        else:  # Even - centered on 4-pixel junction
            ymin_N = cy - half_N
            ymax_N = cy + half_N
            xmin_N = cx - half_N
            xmax_N = cx + half_N

        # Calculate required padding (symmetric on both sides)
        pad_top = max(0, -ymin_N)
        pad_bottom = max(0, ymax_N - ny)
        pad_left = max(0, -xmin_N)
        pad_right = max(0, xmax_N - nx)
        
        # Pad image with zeros if aperture would exceed boundaries
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded_data = np.pad(ima.data, 
                                ((pad_top, pad_bottom), (pad_left, pad_right)), 
                                mode='constant', 
                                constant_values=0)
            # Update center coordinates after padding
            cy += pad_top
            cx += pad_left
            
            # Recalculate aperture boundaries in padded coordinates
            if N % 2 == 1:  # Odd
                ymin_N = cy - half_N
                ymax_N = cy + half_N + (N * oversamp % 2)
                xmin_N = cx - half_N
                xmax_N = cx + half_N + (N * oversamp % 2)
            else:  # Even
                ymin_N = cy - half_N
                ymax_N = cy + half_N
                xmin_N = cx - half_N
                xmax_N = cx + half_N
        else:
            padded_data = ima.data

        # Central spaxel extraction
        if N % 2 == 1:  # Odd - extract full spaxel centered on peak pixel
            half = oversamp // 2
            ymin = cy - half
            ymax = cy + half + (oversamp % 2)
            xmin = cx - half
            xmax = cx + half + (oversamp % 2)
        else:  # Even - extract ONE of the 4 central spaxels (top-left)
            # Peak is at junction, so take the top-left spaxel
            ymin = cy - oversamp
            ymax = cy
            xmin = cx - oversamp
            xmax = cx

        flux_central_spaxel = padded_data[ymin:ymax, xmin:xmax].sum()

        # NxN region extraction
        flux_NxN = padded_data[ymin_N:ymax_N, xmin_N:xmax_N].sum()

        return flux_central_spaxel, flux_NxN

    # MOS function for the fraction of flux collected by the fiber aperture
    def mos_fiber_aperture(self, ins, ima, displacement=0.0):
        """
        Trim a source image based on the fiber aperture response function.
        
        This function applies a circular aperture mask to the input image, setting
        all pixels outside the fiber aperture to zero and calculating the fraction
        of original flux that is collected.
        
        Parameters
        ----------
        ins : dict
            instrument configuration dictionary containing 'aperture' (fiber diameter in arcsec)
            and 'spaxel_size' (spaxel size in arcsec)
        ima : MPDAF Image
            input source image to be convolved with fiber aperture
        displacement : float
            displacement of fiber center from image center along x-axis in arcsec (Default value = 0.0)
            
        Returns
        -------
            flux_fraction : float
                fraction of original flux collected by the fiber aperture
        """
        
        # Get aperture parameters
        aperture_diameter = ins['aperture']  # in arcsec
        aperture_radius = aperture_diameter / 2.0  # in arcsec

        # we need this because the original images are generated with the spaxel size
        spaxel_size = ins['spaxel_size']  # in arcsec
        
        # Create a copy of the input image
        #ima_out = ima.copy()
        
        # Calculate the aperture radius in pixels
        oversamp = ima.oversamp
        aperture_radius_pix = aperture_radius * oversamp / spaxel_size
        displacement_x_pix = displacement * oversamp / spaxel_size
        
        # Get image dimensions
        ny, nx = ima.shape
        
        # Calculate fiber center position (image center + displacement)
        image_center_y, image_center_x = ny // 2, nx // 2
        fiber_center_x = image_center_x + displacement_x_pix
        fiber_center_y = image_center_y  # No displacement in y
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:ny, 0:nx]
        
        # Calculate distance from fiber center for each pixel
        distances = np.sqrt((x_coords - fiber_center_x)**2 + (y_coords - fiber_center_y)**2)
        
        # Create circular aperture mask (True inside aperture, False outside)
        aperture_mask = distances <= aperture_radius_pix
        
        # Calculate original total flux, should always be 1 but just in case
        original_flux = np.sum(ima.data)
        
        # Apply aperture mask - set pixels outside aperture to zero
        #ima_out.data[~aperture_mask] = 0.0
        
        # Calculate flux after aperture application
        #collected_flux = np.sum(ima_out.data)
        
        collected_flux = np.sum(ima.data[aperture_mask])
        
        # Calculate flux fraction
        if original_flux > 0:
            flux_fraction = collected_flux / original_flux
        else:
            flux_fraction = 0.0
        
        return flux_fraction

    def mos_fiber_aperture_batch(self, ins, ima_list, displacement=0.0):
        """
        Compute fiber aperture flux fractions for multiple images at once.
        Optimized to compute the aperture mask only once if all images have the same shape.
        
        Parameters
        ----------
        ins : dict
            instrument configuration dictionary
        ima_list : list of MPDAF Image
            list of input source images
        displacement : float
            displacement of fiber center from image center along x-axis in arcsec
            
        Returns
        -------
        list of float
            flux fractions for each image
        """
        if not ima_list:
            return []
        
        # Get aperture parameters (same for all images)
        aperture_diameter = ins['aperture']
        aperture_radius = aperture_diameter / 2.0
        spaxel_size = ins['spaxel_size']
        
        # Cache for masks by image shape
        mask_cache = {}
        
        flux_fractions = []
        for ima in ima_list:
            ny, nx = ima.shape
            shape_key = (ny, nx)
            
            # Get or compute mask for this shape
            if shape_key not in mask_cache:
                oversamp = ima.oversamp
                aperture_radius_pix = aperture_radius * oversamp / spaxel_size
                displacement_x_pix = displacement * oversamp / spaxel_size
                
                image_center_y, image_center_x = ny // 2, nx // 2
                fiber_center_x = image_center_x + displacement_x_pix
                fiber_center_y = image_center_y
                
                y_coords, x_coords = np.mgrid[0:ny, 0:nx]
                distances = np.sqrt((x_coords - fiber_center_x)**2 + (y_coords - fiber_center_y)**2)
                mask_cache[shape_key] = distances <= aperture_radius_pix
            
            aperture_mask = mask_cache[shape_key]
            
            original_flux = np.sum(ima.data)
            if original_flux > 0:
                collected_flux = np.sum(ima.data[aperture_mask])
                flux_fractions.append(collected_flux / original_flux)
            else:
                flux_fractions.append(0.0)
        
        return flux_fractions

    # Vectorized version
    def rebin_spectrum(self, nph_source, tot_noise, bin_factor=2):
        """Rebin a MPDAF spectrum and its noise by combining adjacent pixels
    
        Parameters
        ----------
        nph_source : MPDAF Spectrum
            Original signal spectrum
        tot_noise : MPDAF Spectrum 
            Original noise spectrum
        bin_factor : int
            Number of pixels to bin together
        
        Returns
        -------
        bin_snr : MPDAF Spectrum
            Rebinned spectrum of the SNR on original wavelength grid
        """
        waves = nph_source.wave.coord()
        n_waves = len(waves)
        n_bins = n_waves // bin_factor
        n_valid = n_bins * bin_factor
        
        signal_reshaped = nph_source.data[:n_valid].reshape(n_bins, bin_factor)
        noise_reshaped = tot_noise.data[:n_valid].reshape(n_bins, bin_factor)
        
        binned_signal = signal_reshaped.sum(axis=1)
        binned_noise = np.sqrt((noise_reshaped**2).sum(axis=1))
        
        waves_reshaped = waves[:n_valid].reshape(n_bins, bin_factor)
        bin_centers = waves_reshaped.mean(axis=1)
        
        final_signal = np.interp(waves, bin_centers, binned_signal)
        final_noise = np.interp(waves, bin_centers, binned_noise)
        
        final_signal = mask_spectrum_edges(final_signal, bin_factor)
        final_noise = mask_spectrum_edges(final_noise, bin_factor)
        
        snr_data = final_signal / final_noise
        bin_snr = Spectrum(data=snr_data, wave=nph_source.wave)
        return bin_snr

    # # # # # # # # # # # # # # #
    # # # Optimal Coadd  # # # #
    # # # # # # # # # # # # # # #

    def get_coadd(self, full_obs, wave):
        """Compute an initial coadd estimate from the PSF FWHM at a given wavelength.

        The optimal extraction aperture scales with the PSF size. This returns
        the number of spaxels corresponding to ~3-sigma of the PSF, rounded
        to the nearest integer (minimum 1).

        Parameters
        ----------
        full_obs : dict
            Full observation dictionary (must contain 'INS' and 'CH' keys).
        wave : float
            Wavelength in Angstrom at which to evaluate the PSF FWHM.

        Returns
        -------
        int
            Initial coadd estimate.
        """
        insfam = getattr(self, full_obs["INS"])
        ins = insfam[full_obs["CH"]]
        fwhm, _ = get_seeing_fwhm(
            self.obs['seeing'],
            self.obs['airmass'],
            wave,
            self.tel['diameter'],
            ins['iq_fwhm_tel'],
            ins['iq_fwhm_ins'],
            glao=self.obs.get('glao', False)
        )
        if not np.isscalar(fwhm):
            fwhm = float(fwhm.ravel()[0])
        coadd = max(1, int(3 * fwhm / (2.35 * ins['spaxel_size']) + 0.5))
        return coadd

    def _resolve_best_coadd_ifs(self, ins, ima, wave_ref, spec=None, debug=False, max_coadd=40):
        """Resolve ima_coadd='best' by finding the coadd that maximizes aperture SNR.

        Uses a single reference wavelength and a bidirectional hill-climbing
        search starting from a PSF-based initial guess.  Works with scalar
        quantities so it can be called from snr_from_source_ifs,
        _snr_at_wave_ifs, and time_from_source_ifs alike.

        Parameters
        ----------
        ins : dict
            Instrument configuration.
        ima : MPDAF Image or None
            Source image (None for ps/sb).
        wave_ref : float
            Reference wavelength in Angstrom.
        spec : MPDAF Spectrum, float, or None
            Source spectrum (or scalar value at wave_ref). When provided, the
            full SNR metric SNR = fsq*S / sqrt(fsq*S + N^2*bg) is used.
            When None, falls back to the sky-dominated approximation fsq / N.
        debug : bool
            If True, log the result.
        max_coadd : int
            Upper bound for the search (default 20).
        """
        obs = self.obs
        if obs['ima_coadd'] != 'best':
            return

        if obs['ima_type'] == 'sb':
            obs['ima_coadd'] = 1
            if debug:
                self.logger.info("Coadd 'best' for surface brightness: set to 1 (per-spaxel)")
            return

        # ----- Compute scalar noise quantities at wave_ref -----
        sky_at_wave = float(np.interp(wave_ref, obs['skyemi'].wave.coord(), obs['skyemi'].data))
        ins_at_wave = float(np.interp(wave_ref, ins['instrans'].wave.coord(), ins['instrans'].data))
        atm_at_wave = float(np.interp(wave_ref, obs['skyabs'].wave.coord(), obs['skyabs'].data))

        tel_eff_area = self.tel['effective_area_IFS']
        dl = ins['dlbda']
        Ksky = ins_at_wave * ins['spaxel_size']**2 * tel_eff_area * (dl / 1e4)
        dark = ins['dcurrent'] * obs['dit'] * obs['ndit'] / 3600
        ron = ins['ron']**2 * obs['ndit']
        sky_ph_spaxel = sky_at_wave * Ksky * obs['dit'] * obs['ndit']
        bg_per_spaxel = sky_ph_spaxel + dark + ron

        # ----- Source counts per spaxel at wave_ref (for full SNR metric) -----
        # flux conversion (same as in snr_from_source_ifs / time_from_source_ifs)
        flux_conv = 1.0
        if obs['spec_type'] == 'cont':
            flux_conv *= ins['dlbda']
        # sb is excluded above (early return with coadd=1)

        factor_src = None
        if spec is not None:
            a = (wave_ref * 1e-8 / (H_cgs * C_cgs)) * (tel_eff_area * 1e4) * atm_at_wave
            Kt = ins_at_wave * a
            if np.isscalar(spec):
                spec_at_wave = float(spec)
            else:
                # MPDAF Spectrum: interpolate at wave_ref
                spec_at_wave = float(np.interp(wave_ref, spec.wave.coord(), spec.data))
            fs = spec_at_wave * flux_conv * Kt * obs['dit'] * obs['ndit']
            if fs > 0:
                factor_src = fs

        # Initial guess from PSF FWHM
        fwhm_ref, _ = get_seeing_fwhm(
            obs['seeing'], obs['airmass'], wave_ref,
            self.tel['diameter'], ins['iq_fwhm_tel'], ins['iq_fwhm_ins'],
            glao=obs.get('glao', False)
        )
        if not np.isscalar(fwhm_ref):
            fwhm_ref = float(fwhm_ref.ravel()[0])
        coadd0 = max(1, int(3 * fwhm_ref / (2.35 * ins['spaxel_size']) + 0.5))

        # For resolved sources the optimal aperture can exceed the PSF-based
        # default cap; derive the upper bound from the image extent instead.
        if obs['ima_type'] == 'resolved' and ima is not None:
            oversamp = getattr(ima, 'oversamp', 10)
            max_coadd = max(max_coadd, min(ima.data.shape) // oversamp)

        def _snr_for_coadd(N):
            """Aperture SNR metric at wave_ref for NxN coadd."""
            uneven = 1 if N % 2 == 1 else 0
            psf_ima = self.get_image_psf(ins, wave_ref, uneven=uneven)
            if obs['ima_type'] == 'resolved':
                if ima is None:
                    raise ValueError("For resolved sources, image must not be None.")
                psf_ima = convolve_and_center(ima, psf_ima)
            _, fsq = self.ifs_spaxel_aperture(ins, psf_ima, N=N)
            if factor_src is not None:
                # Full SNR metric: signal / sqrt(signal + N^2 * bg_per_spaxel)
                signal = fsq * factor_src
                noise_sq = signal + N**2 * bg_per_spaxel
                return signal / np.sqrt(max(noise_sq, 1e-30))
            else:
                # Fallback: sky-dominated approximation
                return fsq / N

        # Evaluate fractions for all candidate coadd values
        best_snr = -1.0
        best_coadd = coadd0

        # search downward from coadd0
        for c in range(coadd0, 0, -1):
            metric = _snr_for_coadd(c)
            if metric < best_snr:
                break
            best_coadd, best_snr = c, metric

        # search upward from coadd0 + 1
        for c in range(coadd0 + 1, max_coadd + 1):
            metric = _snr_for_coadd(c)
            if metric < best_snr:
                break
            best_coadd, best_snr = c, metric

        obs['ima_coadd'] = best_coadd
        if debug:
            self.logger.info(
                f"Optimal coadd: {best_coadd} "
                f"(at {wave_ref:.1f} AA, initial guess={coadd0})"
            )

    # # # # # # # # # # 
    # # # # SNR # # # #
    # # # # # # # # # #

    def snr_from_source(self, ins, ima, spec, debug=True, sat=True):
        """ main routine to perform the S/N computation for a given source

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image, can be None for surface brightness source or point source
        spec : MPDAF spectrum
            source spectrum
        debug :
            if True print some info in logger.debug mode (Default value = True)
        sat : bool
            if True compute the fraction of saturated pixels (Default value = True)

        Returns
        -------
        dict
            result dictionary with SNR and other useful info

        """

        # basic checks on obs parameters
        _checkobs(self.obs, keys=['dit', 'ndit'])

        # First we check that the line is inside the instrument range
        if self.obs['spec_type'] == 'line':
            if (ins['lbda1'] > self.obs['wave_line_center'] - tol_wave * self.obs['wave_line_fwhm']) or (ins['lbda2'] < self.obs['wave_line_center'] + tol_wave * self.obs['wave_line_fwhm']):
                res = {}
                res['message'] = 'The line center is outside (or partially outside) the instrument spectral range'
                print(res['message'])
                return res
        
        if ins['type'] == 'IFS':
            res = self.snr_from_source_ifs(ins, ima, spec, debug, sat)
        elif ins['type'] == 'MOS':
            res = self.snr_from_source_mos(ins, ima, spec, debug, sat)
        
        # we return also the saturation limit
        res['sat_limit'] = threshold_sat
        return res

    def snr_from_source_ifs(self, ins, ima, spec, debug=True, sat=True):
        
        obs = self.obs
        res = {}
        start_time = time.time()

        # unit conversion
        flux = 1.0
        # from per Angstrom to per spectel
        if obs['spec_type'] == 'cont':
            flux *= ins['dlbda']
            if debug:
                self.logger.debug(f"Flux converted from per Angstrom to per spectel by multiplying for dlbda: {ins['dlbda']}")
        # from 1/arcsec2 to 1/spaxel    
        if obs['ima_type'] == 'sb':
            flux *= ins['spaxel_size']**2
            if debug:
                self.logger.debug(f"Flux converted from per arcsec^2 to per spaxel by multiplying for spaxel_size^2: {ins['spaxel_size']**2}")
        
        if debug:
            if obs['skycalc']:
                self.logger.debug("Sky computed with skycalc")
            else:
                self.logger.info("Sky taken from static files")

        ins_sky = obs['skyemi']
        ins_ins = ins['instrans']
        ins_atm = obs['skyabs']

        # LSF convolution of the spectrum
        spec = spec.filter(width=ins['lsfpix'])
        if debug:
            self.logger.debug(f"Spectrum convolved with LSF of {ins['lsfpix']} pixels")

        # we select only wave_grid points in wavelength (for the variable PSF computation)
        wave = spec.wave.coord()
        indices = np.linspace(0, len(wave) - 1, wave_grid, dtype=int)
        selected_wave = wave[indices]

        # telescope effective area
        tel_eff_area = self.tel['effective_area_IFS']

        # pre-compute the conversion factor that is used in the SNR computation
        # to compute the number of photons source and sky
        dl = spec.wave.get_step(unit='Angstrom')
        a = (wave*1.e-8/(H_cgs*C_cgs)) * (tel_eff_area*1.e4) * (ins_atm.data)
        Kt =  ins_ins.data * a
        Ksky = ins_ins.data * ins['spaxel_size']**2 * tel_eff_area * (dl/1e4)
        
        # common factors for sb and ps/resolved cases
        dark = ins['dcurrent'] * obs['dit'] * obs['ndit'] / 3600
        ron = ins['ron']**2 * obs['ndit']
        sky_ph_spaxel = ins_sky * Ksky * obs['dit'] * obs['ndit']
        dark_spaxel = Spectrum(data=np.full(wave.shape, dark), wave=spec.wave)
        ron_spaxel = Spectrum(data=np.full(wave.shape, ron), wave=spec.wave)
        factor_source = spec * flux * Kt * obs['dit'] * obs['ndit']

        # Resolve 'best' coadd via optimal extraction search.
        # In DIT+NDIT mode snr_wave is not meaningful here, so use the centre
        # of the SNR window (if set) or the channel centre as reference.
        if obs['ima_coadd'] == 'best':
            if obs['spec_type'] == 'line':
                lbda_ref = float(obs['wave_line_center'])
            elif obs.get('snr_range') and obs.get('lam_win1') and obs.get('lam_win2'):
                lbda_ref = (obs['lam_win1'] + obs['lam_win2']) / 2.0
            else:
                lbda_ref = (ins['lbda1'] + ins['lbda2']) / 2.0
            lbda_ref = float(np.clip(lbda_ref, ins['lbda1'], ins['lbda2']))
            self._resolve_best_coadd_ifs(ins, ima, lbda_ref, spec=spec, debug=debug)

        sky_ph_square = sky_ph_spaxel * obs['ima_coadd']**2
        dark_square = dark_spaxel * obs['ima_coadd']**2
        ron_square = ron_spaxel * obs['ima_coadd']**2

        if obs['ima_type'] == 'sb':
            source_ph_peak = factor_source
            tot_noise_peak = np.sqrt(source_ph_peak + sky_ph_spaxel + dark_spaxel + ron_spaxel)
            snr_peak = source_ph_peak / tot_noise_peak

            source_ph_square = source_ph_peak * obs['ima_coadd']**2
            #tot_noise_square = np.sqrt(source_ph_square + obs['ima_coadd']**2 * (sky_ph_spaxel + dark_spaxel) + ron_spaxel * obs['ima_coadd']) #! ! ! check
            tot_noise_square = tot_noise_peak * obs['ima_coadd']
            
            #snr_square = source_ph_square / tot_noise_square
            snr_square = snr_peak * obs['ima_coadd']

        elif obs['ima_type'] in ['ps', 'resolved']:
            # added to distingish even/odd coadding for PSF images
            uneven = 1 if obs['ima_coadd'] % 2 == 1 else 0
            psf_array = self.get_image_psf(ins, selected_wave, uneven=uneven)
            
            if obs['ima_type'] == 'ps':
                array_of_images = psf_array
            
            elif obs['ima_type'] == 'resolved':
                array_of_images = []
                if ima is None:
                    raise ValueError("For resolved sources, source image must not be None.")

                for impsf in psf_array:
                    conv_ima = convolve_and_center(ima, impsf)
                    array_of_images.append(conv_ima)

            # we take the fraction of flux in the central spaxel and in the NxN region
            frac_peak_spaxel = []
            frac_square = []
            for selected_im in array_of_images:
                fpeak, fsq = self.ifs_spaxel_aperture(ins, selected_im, N=obs['ima_coadd'])
                frac_peak_spaxel.append(fpeak)
                frac_square.append(fsq)

            # Interpolate onto the full wave grid
            frac_peak_spaxel_full = np.interp(wave, selected_wave, frac_peak_spaxel)
            frac_square_full = np.interp(wave, selected_wave, frac_square)

            # we compute the SNR and noises
            source_ph_peak = factor_source * frac_peak_spaxel_full
            tot_noise_peak = np.sqrt(source_ph_peak + sky_ph_spaxel + dark_spaxel + ron_spaxel)
            snr_peak = source_ph_peak / tot_noise_peak

            source_ph_square = factor_source * frac_square_full
            tot_noise_square = np.sqrt(source_ph_square + obs['ima_coadd']**2 * (sky_ph_spaxel + dark_spaxel+ ron_spaxel))
            snr_square = source_ph_square / tot_noise_square

        frac_source_peak = source_ph_peak / tot_noise_peak**2
        frac_sky_peak    = sky_ph_spaxel / tot_noise_peak**2
        frac_dark_peak   = dark_spaxel / tot_noise_peak**2
        frac_ron_peak    = ron_spaxel / tot_noise_peak**2

        frac_source_square = source_ph_square / tot_noise_square**2
        frac_sky_square    = sky_ph_square / tot_noise_square**2
        frac_dark_square   = dark_square / tot_noise_square**2
        frac_ron_square    = ron_square / tot_noise_square**2

        res['input'] = dict(
            flux_source=spec, 
            atm_abs=ins_atm, 
            ins_trans=ins_ins, 
            atm_emi=ins_sky,
            QE_trans=ins['QE'],
            tel_trans=ins['telescope'],
            ins_noQE_trans=ins['total_instrumental'],
            total_trans = ins_ins*ins_atm
        )

        res['obs'] = obs

        res['peak'] = {}
        
        # source counts and square of the error in the peak spaxel
        res['peak']['nph_source'] = source_ph_peak
        res['peak']['nph_sky'] = sky_ph_spaxel
        res['peak']['dark'] = dark_spaxel
        res['peak']['ron'] = ron_spaxel
        res['peak']['snr'] = snr_peak

        res['peak']['noise'] = {}

        res['peak']['noise']['tot'] = tot_noise_peak
        res['peak']['noise']['frac_source'] = frac_source_peak
        res['peak']['noise']['frac_sky'] = frac_sky_peak
        res['peak']['noise']['frac_dark'] = frac_dark_peak
        res['peak']['noise']['frac_ron'] = frac_ron_peak

        res['spec'] = {}

        # source counts and square of the error in the aperture
        res['spec']['nph_source'] = source_ph_square
        res['spec']['nph_sky'] = sky_ph_square
        res['spec']['dark'] = dark_square
        res['spec']['ron'] = ron_square
        res['spec']['snr'] = snr_square

        res['spec']['noise'] = {}

        res['spec']['noise']['tot'] = tot_noise_square
        res['spec']['noise']['frac_source'] = frac_source_square
        res['spec']['noise']['frac_sky'] = frac_sky_square
        res['spec']['noise']['frac_dark'] = frac_dark_square
        res['spec']['noise']['frac_ron'] = frac_ron_square

        # Simulate 1D spectrum with noise in extraction aperture (obs['ima_coadd']**2)
        '''
        simulated_data = []

        for i in range(len(wave)):
            # Per-pixel values
            s_pix = source_ph_square.data[i] / (obs['ima_coadd']**2)
            sky_pix = sky_ph_spaxel.data[i]
            d_pix = dark
            
            # Clamp to non-negative values to avoid Poisson distribution errors, Nans, etc...
            s_pix = max(0.0, s_pix) if not np.isnan(s_pix) else 0.0
            sky_pix = max(0.0, sky_pix) if not np.isnan(sky_pix) else 0.0
            
            sim_counts = simulate_counts(
                npix=obs['ima_coadd']**2,
                source=s_pix,
                sky=sky_pix,
                dark=d_pix,
                RON=ins['ron']
            )
            simulated_data.append(sim_counts)
        '''
        # Vectorized simulation of counts for the entire spectrum
        npix = obs['ima_coadd']**2
        source_pix = np.clip(np.nan_to_num(source_ph_square.data / npix, nan=0.0), 0, None)
        sky_pix = np.clip(np.nan_to_num(sky_ph_spaxel.data, nan=0.0), 0, None)
        
        simulated_data = simulate_counts_vectorized(
            npix=npix, source_arr=source_pix, sky_arr=sky_pix,
            dark=dark, RON=ins['ron']
        )
        res['spec']['simulated_counts'] = Spectrum(data=simulated_data, wave=spec.wave)

        # if spectral rebinning is requested
        if 'spbin' in obs and obs['spbin'] > 1:
            res['spec']['snr_rebin'] = self.rebin_spectrum(res['spec']['nph_source'], res['spec']['noise']['tot'], obs['spbin']) 
            res['peak']['snr_rebin'] = self.rebin_spectrum(res['peak']['nph_source'], res['peak']['noise']['tot'], obs['spbin']) 
            if debug:
                self.logger.debug(f"Rebinned SNR computed with factor {obs['spbin']}")

        # simple check on the saturation based on source and sky counts
        if sat:
            flag_sat = False
            frac_sat = None
            # Divide by NDIT: nph_source and nph_sky include DIT*NDIT;
            # saturation is a per-single-exposure limit.
            _ndit_sat = max(1, int(obs.get('ndit') or 1))
            max_counts = (res['peak']['nph_source'] + res['peak']['nph_sky']) / _ndit_sat
            data_arr = np.array(max_counts.data)
            if np.any(data_arr > threshold_sat):
                flag_sat = True
                frac_sat = np.sum(data_arr > threshold_sat) / len(data_arr)
                if debug:
                    self.logger.debug(f'Spectrum saturated: {flag_sat}, fraction of saturated pixels: {frac_sat}')
        
        res['flag_sat'] = flag_sat
        res['frac_sat'] = frac_sat

        # Last check to clean the results if the spectrum is a line
        if obs['spec_type'] == 'line':
            center = obs['wave_line_center']
            fwhm = obs['wave_line_fwhm']
            for key in ['peak', 'spec']:
                if key in res:
                    mask_spectra_in_dict(res[key], center, fwhm, n_fwhm)
                            
        end_time = time.time()
        
        if debug:
            self.logger.debug(f"Total processing time: {end_time - start_time} seconds")

            # we print a proxy of the SNR at the line center or at the middle of the wavelength range
            if obs['spec_type'] == 'line':
                mid_idx = np.abs(res['peak']['snr'].wave.coord() - obs['wave_line_center']).argmin()
            else:
                mid_idx = len(res['peak']['snr'].data) // 2
            mid_wave = res['peak']['snr'].wave.coord(mid_idx)

            self.logger.debug("---- ETC Summary ----")
            self.logger.debug(f"Instrument: {obs['INS']} - Channel: {obs['CH']}")
            self.logger.debug(f"Spec type: {obs['spec_type']} | Ima type: {obs['ima_type']}")
            self.logger.debug(f"DIT: {obs['dit']} s | NDIT: {obs['ndit']} | Coadd: {obs['ima_coadd']}")
            self.logger.debug(f"Wavelength range: {res['peak']['snr'].wave.get_start()} - {res['peak']['snr'].wave.get_end()} AA")
            self.logger.debug(f"Rebinning factor (spbin): {obs.get('spbin', None)}")

            self.logger.debug(f"SNR (peak) at lam={mid_wave:.1f} AA: {res['peak']['snr'].data[mid_idx]:.2f}")
            self.logger.debug(f"SNR (aperture) at lam={mid_wave:.1f} AA: {res['spec']['snr'].data[mid_idx]:.2f}")

            if 'snr_rebin' in res['spec']:
                self.logger.debug(f"SNR (aperture rebinned) at lam={mid_wave:.1f} AA: {res['spec']['snr_rebin'].data[mid_idx]:.2f}")

            self.logger.debug(f"Fraction of saturated pixels: {res.get('frac_sat', None)}")
            self.logger.debug("---------------------")
    
        return res

    def snr_from_source_mos(self, ins, ima, spec, debug=True, sat=True):
        
        obs = self.obs
        res = {}
        start_time = time.time()

        # unit conversion
        flux = 1.0
        # from per Angstrom to per spectel
        if obs['spec_type'] == 'cont':
            flux *= ins['dlbda']
            if debug:
                self.logger.debug(f"Flux converted from per Angstrom to per spectel by multiplying for dlbda: {ins['dlbda']}")

        if debug:
            if obs['skycalc']:
                self.logger.debug("Sky computed with skycalc")
            else:
                self.logger.info("Sky taken from static files")

        ins_sky = obs['skyemi']
        ins_ins = ins['instrans']
        ins_atm = obs['skyabs']

        # LSF convolution of the spectrum
        spec = spec.filter(width=ins['lsfpix'])
        if debug:
            self.logger.debug(f"Spectrum convolved with LSF of {ins['lsfpix']} pixels")

        # we select only wave_grid points in wavelength (for the variable PSF computation)
        wave = spec.wave.coord()
        indices = np.linspace(0, len(wave) - 1, wave_grid, dtype=int)
        selected_wave = wave[indices]

        # telescope effective area
        tel_eff_area = self.tel['effective_area_MOS']

        # # # # # # # # # # # # # #  # # # # # MOS COMMENT # # # # # # # # # # # # # # # # # # # 
        # pre-compute the conversion factor that is used in the SNR computation                #
        # to compute the number of photons source and sky.                                      #
        # We assume that all the light collected from source + sky is spreaded uniformly       #
        # in num_trace * trace_pixel_width pixels, so each one of these pixels will receive    #
        # 1/(num_trace * trace_pixel_width) of the total counts                                #
        # # # # # # # # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # we get the slicing only for the moshr ! ! ! put this in a dictionary as slicing/splitting parameter
        if ins['name'] == 'moslr':
            num_trace = 1
            
        elif ins['name'] == 'moshr':
            num_trace = 7

        # number of vertical pixels needed to fully extract each trace in MOS HR and MOS LR
        trace_pixel_width = int(np.ceil(ins['lsfpix']))

        dl = spec.wave.get_step(unit='Angstrom')
        a = (wave*1.e-8/(H_cgs*C_cgs)) * (tel_eff_area*1.e4) * (ins_atm.data)
        Kt =  ins_ins.data * a
        Ksky = ins_ins.data * np.pi * (ins['aperture'] / 2)**2 * tel_eff_area * (dl/1e4)

        sky_ph_aperture = ins_sky * Ksky * obs['dit'] * obs['ndit']

        dark = ins['dcurrent'] * obs['dit'] * obs['ndit'] / 3600 * num_trace * trace_pixel_width
        ron = ins['ron']**2 * obs['ndit'] * num_trace * trace_pixel_width

        dark_tot = Spectrum(data=np.full(wave.shape, dark), wave=spec.wave)
        ron_tot = Spectrum(data=np.full(wave.shape, ron), wave=spec.wave)
        
        factor_source = spec * flux * Kt * obs['dit'] * obs['ndit']
        centering_eff = MOS_DEFAULT_CENTERING_EFF if obs['disp'] is None else 1.0
        fiber_injection_full = np.full_like(wave, centering_eff)
        
        if obs['ima_type'] == 'sb':
            source_ph_aperture = factor_source * np.pi * (ins['aperture'] / 2)**2
        
        elif obs['ima_type'] in ['ps', 'resolved']:
            psf_array = self.get_image_psf(ins, selected_wave)
            
            if obs['ima_type'] == 'ps':
                array_of_images = psf_array
            
            elif obs['ima_type'] == 'resolved':
                array_of_images = []
                if ima is None:
                    raise ValueError("For resolved sources, source image must not be None.")

                for impsf in psf_array:
                    conv_ima = convolve_and_center(ima, impsf)
                    array_of_images.append(conv_ima)

            # we take the fraction of flux collected by the fiber aperture
            displacement = 0.0 if obs['disp'] is None else obs['disp']
            frac_fiber = self.mos_fiber_aperture_batch(ins, array_of_images, displacement=displacement)

            # Interpolate onto the full wave grid
            frac_fiber_full = np.interp(wave, selected_wave, frac_fiber)
            fiber_injection_full = frac_fiber_full * centering_eff

            source_ph_aperture = factor_source * frac_fiber_full

        tot_noise_aperture = np.sqrt(source_ph_aperture + sky_ph_aperture + dark_tot + ron_tot)
        snr_aperture = source_ph_aperture / tot_noise_aperture

        frac_source_aperture = source_ph_aperture / tot_noise_aperture**2
        frac_sky_aperture    = sky_ph_aperture / tot_noise_aperture**2
        frac_dark_aperture   = dark_tot / tot_noise_aperture**2
        frac_ron_aperture    = ron_tot / tot_noise_aperture**2

        fiber_injection_spec = Spectrum(data=fiber_injection_full, wave=spec.wave)
        res['input'] = dict(
            flux_source=spec, 
            atm_abs=ins_atm, 
            ins_trans=ins_ins, 
            atm_emi=ins_sky,
            QE_trans=ins['QE'],
            tel_trans=ins['telescope'],
            ins_noQE_trans=ins['total_instrumental'],
            fiber_injection=fiber_injection_spec,
            total_trans=ins_ins * ins_atm * fiber_injection_spec
        )

        res['obs'] = obs

        res['spec'] = {}

        # source counts and square of the error in the aperture
        res['spec']['nph_source'] = source_ph_aperture
        res['spec']['nph_sky'] = sky_ph_aperture
        res['spec']['dark'] = dark_tot
        res['spec']['ron'] = ron_tot
        res['spec']['snr'] = snr_aperture

        res['spec']['noise'] = {}

        res['spec']['noise']['tot'] = tot_noise_aperture
        res['spec']['noise']['frac_source'] = frac_source_aperture
        res['spec']['noise']['frac_sky'] = frac_sky_aperture
        res['spec']['noise']['frac_dark'] = frac_dark_aperture
        res['spec']['noise']['frac_ron'] = frac_ron_aperture

        # Simulate 1D spectrum with noise in extraction aperture (num_trace * trace_pixel_width)
        npix = num_trace * trace_pixel_width
        
        # Per-pixel values for all wavelengths (vectorized)
        source_pix = np.clip(np.nan_to_num(source_ph_aperture.data / npix, nan=0.0), 0, None)
        sky_pix = np.clip(np.nan_to_num(sky_ph_aperture.data / npix, nan=0.0), 0, None)
        dark_pix = dark / npix
        
        simulated_data = simulate_counts_vectorized(
            npix=npix,
            source_arr=source_pix,
            sky_arr=sky_pix,
            dark=dark_pix,
            RON=ins['ron']
        )
        
        res['spec']['simulated_counts'] = Spectrum(data=simulated_data, wave=spec.wave)

        # if spectral rebinning is requested
        if 'spbin' in obs and obs['spbin'] > 1:
            res['spec']['snr_rebin'] = self.rebin_spectrum(res['spec']['nph_source'], res['spec']['noise']['tot'], obs['spbin']) 
            if debug:
                self.logger.debug(f"Rebinned SNR computed with factor {obs['spbin']}")

        # simple check on the saturation based on source and sky counts
        if sat:
            flag_sat = False
            frac_sat = None

            # Counts are spread over num_trace * trace_pixel_width pixels;
            # also divide by NDIT since nph_source/nph_sky include DIT*NDIT
            # and saturation is a per-single-exposure limit.
            _ndit_sat = max(1, int(obs.get('ndit') or 1))
            max_counts = (res['spec']['nph_source'] + res['spec']['nph_sky']) / (num_trace * trace_pixel_width * _ndit_sat)
            data_arr = np.array(max_counts.data)
            if np.any(data_arr > threshold_sat):
                flag_sat = True
                frac_sat = np.sum(data_arr > threshold_sat) / len(data_arr)
                if debug:
                    self.logger.debug(f'Spectrum saturated: {flag_sat}, fraction of saturated pixels: {frac_sat}')
        
        res['flag_sat'] = flag_sat
        res['frac_sat'] = frac_sat

        # we export the number of traces for every object and the
        # number of vertical pixels needed to fully extract each trace
        res['mos_num_trace'] = num_trace
        res['mos_trace_pixel_width'] = trace_pixel_width

        # Last check to clean the results if the spectrum is a line
        if obs['spec_type'] == 'line':
            center = obs['wave_line_center']
            fwhm = obs['wave_line_fwhm']
            for key in ['spec']:
                if key in res:
                    mask_spectra_in_dict(res[key], center, fwhm, n_fwhm)
                            
        end_time = time.time()
        
        if debug:
            self.logger.debug(f"Total processing time: {end_time - start_time} seconds")

            # we print a proxy of the SNR at the line center or at the middle of the wavelength range
            if obs['spec_type'] == 'line':
                mid_idx = np.abs(res['spec']['snr'].wave.coord() - obs['wave_line_center']).argmin()
            else:
                mid_idx = len(res['spec']['snr'].data) // 2
            mid_wave = res['spec']['snr'].wave.coord(mid_idx)

            self.logger.debug("---- ETC Summary ----")
            self.logger.debug(f"Instrument: {obs['INS']} - Channel: {obs['CH']}")
            self.logger.debug(f"Spec type: {obs['spec_type']} | Ima type: {obs['ima_type']}")
            self.logger.debug(f"DIT: {obs['dit']} s | NDIT: {obs['ndit']} | Coadd: {obs['ima_coadd']}")
            self.logger.debug(f"Wavelength range: {res['spec']['snr'].wave.get_start()} - {res['spec']['snr'].wave.get_end()} AA")
            self.logger.debug(f"Rebinning factor (spbin): {obs.get('spbin', None)}")

            self.logger.debug(f"SNR (aperture) at lam={mid_wave:.1f} AA: {res['spec']['snr'].data[mid_idx]:.2f}")

            if 'snr_rebin' in res['spec']:
                self.logger.debug(f"SNR (aperture rebinned) at lam={mid_wave:.1f} AA: {res['spec']['snr_rebin'].data[mid_idx]:.2f}")

            self.logger.debug(f"Fraction of saturated pixels: {res.get('frac_sat', None)}")
            self.logger.debug("---------------------")
    
        return res

    # # # # # # # # # # # # # # # # # 
    # # # # # SNR at single λ # # # #
    # # # # # # # # # # # # # # # # #

    def snr_at_wave(self, ins, ima, spec, wave_target=None, debug=False):
        """Compute SNR at a single wavelength (fast version).
        
        This is a lightweight version of snr_from_source that computes the SNR
        at a single wavelength only, avoiding unnecessary computations over 
        the full spectral range. Ideal for iterative optimizations.
        
        Parameters
        ----------
        ins : dict
            Instrument configuration (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image or None
            Source image (None for sb/ps)
        spec : MPDAF spectrum
            Source spectrum
        wave_target : float or None
            Target wavelength in Angstrom. If None, uses obs['snr_wave'] 
            or obs['wave_line_center'] for line spectra.
        debug : bool
            Print debug info (Default value = False)
        
        Returns
        -------
        dict
            SNR values at the target wavelength including:
            - wave: target wavelength
            - snr_peak: SNR in peak spaxel (IFS only)
            - snr_aperture: SNR in extraction aperture
            - nph_source_*: source photon counts
            - nph_sky_*: sky photon counts
            - noise_*: total noise
        """
        obs = self.obs
        _checkobs(self.obs, keys=['dit', 'ndit'])
        start_time = time.time()
        
        # Determine target wavelength (same logic as time_from_source)
        if wave_target is None:
            if obs['spec_type'] == 'line':
                wave_target = obs['wave_line_center']
            else:
                if obs.get('snr_wave') is None:
                    raise ValueError("snr_wave must be set in obs or passed as wave_target")
                wave_target = obs['snr_wave']
        
        # Check wavelength is in range (same checks as time_from_source)
        if obs['spec_type'] == 'line':
            if (ins['lbda1'] > wave_target - tol_wave * obs['wave_line_fwhm']) or \
               (ins['lbda2'] < wave_target + tol_wave * obs['wave_line_fwhm']):
                raise ValueError('The line center is outside (or partially outside) the instrument spectral range')
        else:
            if (ins['lbda1'] > wave_target - tol_wave * default_angstrom_edge) or \
               (ins['lbda2'] < wave_target + tol_wave * default_angstrom_edge):
                raise ValueError('The SNR wavelength is outside (or almost outside) the instrument spectral range')
        
        # Unit conversion factors
        flux = 1.0
        if obs['spec_type'] == 'cont':
            flux *= ins['dlbda']
        if obs['ima_type'] == 'sb' and ins['type'] == 'IFS':
            flux *= ins['spaxel_size']**2
        
        # LSF convolution of the spectrum (same as snr_from_source)
        spec_conv = spec.filter(width=ins['lsfpix'])
        
        # Get values at target wavelength via interpolation
        wave_full = spec_conv.wave.coord()
        spec_at_wave = np.interp(wave_target, wave_full, spec_conv.data)
        sky_at_wave = np.interp(wave_target, obs['skyemi'].wave.coord(), obs['skyemi'].data)
        ins_at_wave = np.interp(wave_target, ins['instrans'].wave.coord(), ins['instrans'].data)
        atm_at_wave = np.interp(wave_target, obs['skyabs'].wave.coord(), obs['skyabs'].data)
        
        # Telescope effective area
        if ins['type'] == 'IFS':
            tel_eff_area = self.tel['effective_area_IFS']
        else:
            tel_eff_area = self.tel['effective_area_MOS']
        
        dl = ins['dlbda']
        
        # Conversion factors at this wavelength
        a = (wave_target * 1e-8 / (H_cgs * C_cgs)) * (tel_eff_area * 1e4) * atm_at_wave
        Kt = ins_at_wave * a
        
        if ins['type'] == 'IFS':
            res = self._snr_at_wave_ifs(ins, ima, spec_at_wave, wave_target, flux, Kt, 
                                          ins_at_wave, sky_at_wave, tel_eff_area, dl, debug)
        else:
            res = self._snr_at_wave_mos(ins, ima, spec_at_wave, wave_target, flux, Kt,
                                          ins_at_wave, sky_at_wave, tel_eff_area, dl, debug)
        
        if debug:
            end_time = time.time()
            self.logger.debug(f"snr_at_wave processing time: {end_time - start_time:.4f} seconds")
        
        return res

    def _snr_at_wave_ifs(self, ins, ima, spec_at_wave, wave_target, flux, Kt, 
                         ins_at_wave, sky_at_wave, tel_eff_area, dl, debug):
        """Internal IFS SNR computation at single wavelength."""
        obs = self.obs

        # Resolve 'best' coadd at this wavelength
        if obs['ima_coadd'] == 'best':
            self._resolve_best_coadd_ifs(ins, ima, wave_target, spec=spec_at_wave, debug=debug)
        
        Ksky = ins_at_wave * ins['spaxel_size']**2 * tel_eff_area * (dl / 1e4)
        
        dark = ins['dcurrent'] * obs['dit'] * obs['ndit'] / 3600
        ron = ins['ron']**2 * obs['ndit']
        sky_ph_spaxel = sky_at_wave * Ksky * obs['dit'] * obs['ndit']
        
        factor_source = spec_at_wave * flux * Kt * obs['dit'] * obs['ndit']
        
        if obs['ima_type'] == 'sb':
            source_ph_peak = factor_source
            source_ph_square = source_ph_peak * obs['ima_coadd']**2
            
        elif obs['ima_type'] in ['ps', 'resolved']:
            # Compute PSF at single wavelength only
            uneven = 1 if obs['ima_coadd'] % 2 == 1 else 0
            psf_ima = self.get_image_psf(ins, wave_target, uneven=uneven)
            
            if obs['ima_type'] == 'resolved':
                if ima is None:
                    raise ValueError("For resolved sources, image must not be None.")
                psf_ima = convolve_and_center(ima, psf_ima)
            
            fpeak, fsq = self.ifs_spaxel_aperture(ins, psf_ima, N=obs['ima_coadd'])
            
            source_ph_peak = factor_source * fpeak
            source_ph_square = factor_source * fsq
        
        # SNR calculations
        sky_ph_square = sky_ph_spaxel * obs['ima_coadd']**2
        dark_square = dark * obs['ima_coadd']**2
        ron_square = ron * obs['ima_coadd']**2
        
        tot_noise_peak = np.sqrt(source_ph_peak + sky_ph_spaxel + dark + ron)
        snr_peak = source_ph_peak / tot_noise_peak
        
        tot_noise_square = np.sqrt(source_ph_square + sky_ph_square + dark_square + ron_square)
        snr_square = source_ph_square / tot_noise_square
        
        if debug:
            self.logger.debug(f"SNR at {wave_target:.1f} AA: peak={snr_peak:.2f}, aperture={snr_square:.2f}")
        
        return {
            'wave': wave_target,
            'ima_coadd': obs['ima_coadd'],
            'snr_peak': snr_peak,
            'snr_aperture': snr_square,
            'nph_source_peak': source_ph_peak,
            'nph_source_aperture': source_ph_square,
            'nph_sky_peak': sky_ph_spaxel,
            'nph_sky_aperture': sky_ph_square,
            'noise_peak': tot_noise_peak,
            'noise_aperture': tot_noise_square
        }

    def _snr_at_wave_mos(self, ins, ima, spec_at_wave, wave_target, flux, Kt,
                          ins_at_wave, sky_at_wave, tel_eff_area, dl, debug):
        """Internal MOS SNR computation at single wavelength."""
        obs = self.obs
        num_trace = 1 if ins['name'] == 'moslr' else 7
        trace_pixel_width = int(np.ceil(ins['lsfpix']))
        
        Ksky = ins_at_wave * np.pi * (ins['aperture'] / 2)**2 * tel_eff_area * (dl / 1e4)
        
        sky_ph_aperture = sky_at_wave * Ksky * obs['dit'] * obs['ndit']
        dark = ins['dcurrent'] * obs['dit'] * obs['ndit'] / 3600 * num_trace * trace_pixel_width
        ron = ins['ron']**2 * obs['ndit'] * num_trace * trace_pixel_width
        
        factor_source = spec_at_wave * flux * Kt * obs['dit'] * obs['ndit']
        
        if obs['ima_type'] == 'sb':
            source_ph_aperture = factor_source * np.pi * (ins['aperture'] / 2)**2
            
        elif obs['ima_type'] in ['ps', 'resolved']:
            psf_ima = self.get_image_psf(ins, wave_target)
            
            if obs['ima_type'] == 'resolved':
                if ima is None:
                    raise ValueError("For resolved sources, image must not be None.")
                psf_ima = convolve_and_center(ima, psf_ima)
            
            ffiber = self.mos_fiber_aperture(ins, psf_ima, displacement=obs.get("disp", 0))
            source_ph_aperture = factor_source * ffiber
        
        tot_noise = np.sqrt(source_ph_aperture + sky_ph_aperture + dark + ron)
        snr_aperture = source_ph_aperture / tot_noise
        
        if debug:
            self.logger.debug(f"SNR at {wave_target:.1f} AA: aperture={snr_aperture:.2f}")
        
        return {
            'wave': wave_target,
            'snr_aperture': snr_aperture,
            'nph_source': source_ph_aperture,
            'nph_sky': sky_ph_aperture,
            'noise': tot_noise
        }

    # # # # # # # # # # # # # 
    # # # # # TIMES # # # # # 
    # # # # # # # # # # # # #

    def time_from_source(self, ins, ima, spec, debug=True, compute='dit'):
        """ main routine to perform the NDIT/DIT computation for a given source, also to find the best combination
        of DIT and NDIT to achieve the target SNR without saturation

        Parameters
        ----------
        ins : dict
            instrument (eg self.ifs['blue'] or self.moslr['red'])
        ima : MPDAF image
            source image, can be None for surface brightness source or point source
        spec : MPDAF spectrum
            source spectrum
        debug :
            if True print some info in logger.debug mode (Default value = True)
        compute : str
            'dit' to compute the DIT for a given NDIT, 'ndit' to compute the NDIT for a given DIT, 
            'best' to compute the best combination of DIT and NDIT to achieve the target SNR without 
            saturation (Default value = 'dit')
        
        Returns
        -------
        dict
            result dictionary (see documentation)
        """

        # basic checks on compute parameter
        if compute not in ['dit', 'ndit', 'best']:
            raise ValueError("Parameter 'compute' must be one of: 'dit', 'ndit', 'best'")

        obs = self.obs

        # SNR_RANGE mode: target median SNR in a wavelength window [lam_win1, lam_win2]
        # instead of at a single reference wavelength (snr_wave).
        # Only for non-line sources; lines always fall through to the standard path below.
        if obs.get('snr_range') and obs['spec_type'] != 'line':
            lam_w1 = obs.get('lam_win1')
            lam_w2 = obs.get('lam_win2')
            if lam_w1 is None or lam_w2 is None:
                return {'message': 'SNR_RANGE=True but LAM_WIN1 / LAM_WIN2 not set.'}
            lam_w1, lam_w2 = float(min(lam_w1, lam_w2)), float(max(lam_w1, lam_w2))
            lbda1, lbda2 = ins['lbda1'], ins['lbda2']
            # Three-case range validation
            if lam_w2 < lbda1 or lam_w1 > lbda2:
                msg = (
                    f'SNR window [{lam_w1:.0f}–{lam_w2:.0f} Å] is completely outside '
                    f'instrument range [{lbda1:.0f}–{lbda2:.0f} Å].'
                )
                if debug:
                    self.logger.debug(msg)
                return {'message': msg}
            clip_note = None
            if lam_w1 < lbda1 or lam_w2 > lbda2:
                orig_w1, orig_w2 = lam_w1, lam_w2
                lam_w1 = max(lam_w1, lbda1)
                lam_w2 = min(lam_w2, lbda2)
                clip_note = (
                    f'SNR window [{orig_w1:.0f}–{orig_w2:.0f} Å] partially outside range, '
                    f'clipped to [{lam_w1:.0f}–{lam_w2:.0f} Å].'
                )
                if debug:
                    self.logger.debug(clip_note)
            _compute = 'ndit' if compute == 'best' else compute
            spbin = int(obs.get('spbin') or 1)
            if obs.get('snr') is None:
                return {'message': 'SNR_RANGE=True but SNR target not set.'}
            effective_snr = float(obs['snr']) / np.sqrt(spbin) if spbin > 1 else float(obs['snr'])
            if debug:
                self.logger.debug(
                    f"SNR_RANGE mode: window=[{lam_w1:.0f}–{lam_w2:.0f} Å], "
                    f"compute='{_compute}', target SNR={obs['snr']}, "
                    f"spbin={spbin}, effective target={effective_snr:.4f}"
                )
            try:
                res = self.time_from_source_window(
                    ins, ima, spec, lam_w1, lam_w2, effective_snr,
                    compute=_compute, debug=debug
                )
            except (RuntimeError, ValueError, TypeError) as e:
                return {'message': str(e)}
            if clip_note:
                res['clip_note'] = clip_note
            return res

        # we check that the line is inside the instrument range
        if self.obs['spec_type'] == 'line':
            if (ins['lbda1'] > self.obs['wave_line_center'] - tol_wave * self.obs['wave_line_fwhm']) or (ins['lbda2'] < self.obs['wave_line_center'] + tol_wave * self.obs['wave_line_fwhm']):
                res = {}
                res['message'] = 'The line center is outside (or partially outside) the instrument spectral range'
                print(res['message'])
                return res
        # if not line we check that the snr_wave is inside the instrument range
        else:
            if (ins['lbda1'] > self.obs['snr_wave'] - tol_wave * default_angstrom_edge) or (ins['lbda2'] < self.obs['snr_wave'] + tol_wave * default_angstrom_edge):
                res = {}
                res['message'] = 'The SNR wavelength is outside (or almost outside) the instrument spectral range'
                print(res['message'])
                return res
        
        if ins['type'] == 'IFS':
            res = self.time_from_source_ifs(ins, ima, spec, debug, compute)
        elif ins['type'] == 'MOS':
            res = self.time_from_source_mos(ins, ima, spec, debug, compute)

        return res
    
    def time_from_source_ifs(self, ins, ima, spec, debug=True, compute='dit'):

        obs = self.obs
        res = {}
        start_time = time.time()

        # Resolve 'best' coadd at the SNR reference wavelength
        _coadd_was_best = (obs['ima_coadd'] == 'best')
        if _coadd_was_best:
            lbda_ref = obs.get('snr_wave')
            if lbda_ref is None and obs['spec_type'] == 'line':
                lbda_ref = obs['wave_line_center']
            if lbda_ref is None:
                lbda_ref = (ins['lbda1'] + ins['lbda2']) / 2.0
            self._resolve_best_coadd_ifs(ins, ima, lbda_ref, spec=spec, debug=debug)

        # unit conversion
        flux = 1.0
        # from per Angstrom to per spectel
        if obs['spec_type'] == 'cont':
            flux *= ins['dlbda']
            if debug:
                self.logger.debug(f"Flux converted from per Angstrom to per spectel by multiplying for dlbda: {ins['dlbda']}")
        # from 1/arcsec2 to 1/spaxel    
        if obs['ima_type'] == 'sb':
            flux *= ins['spaxel_size']**2
            if debug:
                self.logger.debug(f"Flux converted from per arcsec^2 to per spaxel by multiplying for spaxel_size^2: {ins['spaxel_size']**2}")
        
        if debug:
            if obs['skycalc']:
                self.logger.debug("Sky computed with skycalc")
            else:
                self.logger.info("Sky taken from static files")

        ins_sky = obs['skyemi']
        ins_ins = ins['instrans']
        ins_atm = obs['skyabs']

        # LSF convolution of the spectrum
        spec = spec.filter(width=ins['lsfpix'])
        if debug:
            self.logger.debug(f"Spectrum convolved with LSF of {ins['lsfpix']} pixels")

        # For time_from_source, we only need PSF at snr_wave, not full wave_grid
        wave = spec.wave.coord()
        snr_idx = np.abs(wave - obs['snr_wave']).argmin()
        snr_wave_actual = wave[snr_idx]

        # telescope effective area
        tel_eff_area = self.tel['effective_area_IFS']

        # pre-compute the conversion factor that is used in the SNR computation
        # to compute the number of photons source and sky
        dl = spec.wave.get_step(unit='Angstrom')
        a = (wave*1.e-8/(H_cgs*C_cgs)) * (tel_eff_area*1.e4) * (ins_atm.data)
        Kt =  ins_ins.data * a
        Ksky = ins_ins.data * ins['spaxel_size']**2 * tel_eff_area * (dl/1e4)
        
        # common factors for sb and ps/resolved cases
        dark = ins['dcurrent'] / 3600
        ron = ins['ron']**2
        sky_ph_spaxel = ins_sky * Ksky 
        dark_spaxel = Spectrum(data=np.full(wave.shape, dark), wave=spec.wave)
        ron_spaxel = Spectrum(data=np.full(wave.shape, ron), wave=spec.wave)
        factor_source = spec * flux * Kt

        sky_ph_square = sky_ph_spaxel * obs['ima_coadd']**2
        dark_square = dark_spaxel * obs['ima_coadd']**2
        ron_square = ron_spaxel * obs['ima_coadd']**2

        if obs['ima_type'] == 'sb':
            source_ph_peak = factor_source
            source_ph_square = source_ph_peak * obs['ima_coadd']**2

        elif obs['ima_type'] in ['ps', 'resolved']:
            # Compute PSF only at snr_wave (1 wavelength instead of wave_grid)
            uneven = 1 if obs['ima_coadd'] % 2 == 1 else 0
            psf_single = self.get_image_psf(ins, snr_wave_actual, uneven=uneven)
            
            if obs['ima_type'] == 'ps':
                selected_image = psf_single
            
            elif obs['ima_type'] == 'resolved':
                if ima is None:
                    raise ValueError("For resolved sources, source image must not be None.")
                selected_image = convolve_and_center(ima, psf_single)

            # Compute aperture fractions for single image
            frac_peak_snr, frac_square_snr = self.ifs_spaxel_aperture(ins, selected_image, N=obs['ima_coadd'])

            # Apply fractions to full spectrum (use snr_wave fraction for all)
            source_ph_peak = factor_source * frac_peak_snr
            source_ph_square = factor_source * frac_square_snr

        snrv = obs['snr']

        if compute == 'dit': 
            _checkobs(self.obs, keys=['ndit', 'snr', 'snr_wave'])
            nditv = obs['ndit']

            if obs['spbin'] == 1:
                if debug:
                    self.logger.debug(f"Computing DIT without spectral rebinning")
                # nearest wave idx to snr_wave
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                wave_snr = wave[snr_idx]

                sv = source_ph_square.data[snr_idx]
                skyv = sky_ph_square.data[snr_idx]
                darkv = dark_square.data[snr_idx]
                ronv = ron_square.data[snr_idx]

            elif obs['spbin'] > 1:
                if debug:
                    self.logger.debug(f"Computing DIT with spectral rebinning, factor of {obs['spbin']}")
                
                # Find the bin containing snr_wave
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                bin_start = (snr_idx // obs['spbin']) * obs['spbin']
                bin_end = min(bin_start + obs['spbin'], len(wave))
                
                # Sum directly over the bin
                sv = np.sum(source_ph_square.data[bin_start:bin_end])
                skyv = np.sum(sky_ph_square.data[bin_start:bin_end])
                darkv = np.sum(dark_square.data[bin_start:bin_end])
                ronv = np.sum(ron_square.data[bin_start:bin_end])
                
                wave_snr = np.mean(wave[bin_start:bin_end])

            # we solve numerically for the DIT
            A = - (sv**2 * nditv) / (snrv**2)
            B = sv + skyv + darkv
            C = ronv

            roots = np.roots([A, B, C])
            ditv = roots[np.isreal(roots) & (roots > 0)].real[0]

            # saturation check: compute the max-DIT that avoids saturation
            counts_sat = source_ph_peak.data + sky_ph_spaxel.data
            dit_sat = threshold_sat / max(counts_sat)
            flag_sat = bool(ditv > dit_sat)
            if debug:
                self.logger.debug(f"Maximum DIT to avoid saturation: {dit_sat} seconds")
                if flag_sat:
                    self.logger.debug(f"WARNING: computed DIT ({ditv:.2f} s) exceeds saturation limit ({dit_sat:.2f} s)")
                self.logger.debug(f"Computed DIT: {ditv} seconds for NDIT: {nditv} to achieve SNR: {snrv} at wavelength: {wave_snr} AA (nearest to requested SNR wavelength: {obs['snr_wave']} AA), with spectral rebinning factor: {obs['spbin']}")
                self.logger.debug(f"Overriding DIT in the observation dictionary...")
            res['dit'] = ditv
            res['dit_sat'] = dit_sat
            res['flag_sat'] = flag_sat
            obs['dit'] = ditv
        
        elif compute == 'ndit':
            _checkobs(self.obs, keys=['dit', 'snr', 'snr_wave'])
            ditv = obs['dit']
                
            if obs['spbin'] == 1:
                if debug:
                    self.logger.debug(f"Computing NDIT without spectral rebinning")
                # nearest wave idx to snr_wave
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                wave_snr = wave[snr_idx]

                sv = source_ph_square.data[snr_idx]
                skyv = sky_ph_square.data[snr_idx]
                darkv = dark_square.data[snr_idx]
                ronv = ron_square.data[snr_idx] 
            
            elif obs['spbin'] > 1:
                if debug:
                    self.logger.debug(f"Computing NDIT with spectral rebinning, factor of {obs['spbin']}")
                # Find the bin containing snr_wave
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                bin_start = (snr_idx // obs['spbin']) * obs['spbin']
                bin_end = min(bin_start + obs['spbin'], len(wave))
                
                # Sum directly over the bin
                sv = np.sum(source_ph_square.data[bin_start:bin_end])
                skyv = np.sum(sky_ph_square.data[bin_start:bin_end])
                darkv = np.sum(dark_square.data[bin_start:bin_end])
                ronv = np.sum(ron_square.data[bin_start:bin_end])
                
                wave_snr = np.mean(wave[bin_start:bin_end])

            nditv = snrv**2 * (sv + skyv + darkv + ronv / ditv) / (sv**2 * ditv)

            # saturation check: compute the max-DIT that avoids saturation
            counts_sat = source_ph_peak.data + sky_ph_spaxel.data
            dit_sat = threshold_sat / max(counts_sat)
            flag_sat = bool(ditv > dit_sat)
            if debug:
                self.logger.debug(f"Maximum DIT to avoid saturation: {dit_sat} seconds")
                if flag_sat:
                    self.logger.debug(f"WARNING: computed DIT ({ditv:.2f} s) exceeds saturation limit ({dit_sat:.2f} s)")
                self.logger.debug(f"Computed NDIT: {nditv} exposures for DIT: {ditv} to achieve SNR: {snrv} at wavelength: {wave_snr} AA (nearest to requested SNR wavelength: {obs['snr_wave']} AA), with spectral rebinning factor: {obs['spbin']}")
                self.logger.debug(f"Overriding NDIT in the observation dictionary...")
            res['ndit'] = nditv
            res['dit_sat'] = dit_sat
            res['flag_sat'] = flag_sat
            obs['ndit'] = nditv

        elif compute == 'best':
            _checkobs(self.obs, keys=['snr', 'snr_wave'])
            snrv = obs['snr']

            if obs['spbin'] == 1:
                if debug:
                    self.logger.debug(f"Computing best DITxNDIT combination without spectral rebinning")
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                wave_snr = wave[snr_idx]
            elif obs['spbin'] > 1:
                if debug:
                    self.logger.debug(f"Computing best DITxNDIT combination with spectral rebinning, factor of {obs['spbin']}")
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                bin_start = (snr_idx // obs['spbin']) * obs['spbin']
                bin_end = min(bin_start + obs['spbin'], len(wave))
                wave_snr = np.mean(wave[bin_start:bin_end])

            def _get_svs(src_sq, sky_sq, drk_sq, ron_sq):
                """Extract signal scalars at the SNR reference pixel/bin."""
                if obs['spbin'] == 1:
                    return (src_sq.data[snr_idx], sky_sq.data[snr_idx],
                            drk_sq.data[snr_idx], ron_sq.data[snr_idx])
                bs = (snr_idx // obs['spbin']) * obs['spbin']
                be = min(bs + obs['spbin'], len(src_sq.data))
                return (np.sum(src_sq.data[bs:be]), np.sum(sky_sq.data[bs:be]),
                        np.sum(drk_sq.data[bs:be]), np.sum(ron_sq.data[bs:be]))

            # Coadd-DIT feedback loop: iterate until the optimal coadd is
            # consistent with the solved DIT+NDIT (converges in 2-3 steps).
            ditv = dit_sat = ndit_raw = nditv = None
            for _fb in range(5):
                sv, skyv, darkv, ronv = _get_svs(
                    source_ph_square, sky_ph_square, dark_square, ron_square)

                counts = source_ph_peak.data + sky_ph_spaxel.data
                dit_sat = threshold_sat / max(counts)

                ndit_raw = snrv**2 * (sv + skyv + darkv + ronv / dit_sat) / (sv**2 * dit_sat)
                nditv = max(1, int(np.ceil(ndit_raw)))

                A = -(sv**2 * nditv) / (snrv**2)
                B = sv + skyv + darkv
                C = ronv
                roots = np.roots([A, B, C])
                ditv = roots[np.isreal(roots) & (roots > 0)].real[0]

                if not _coadd_was_best:
                    break  # fixed coadd — no iteration needed

                # Re-optimise coadd for the found DIT+NDIT
                old_coadd = obs['ima_coadd']
                obs['dit'] = ditv
                obs['ndit'] = nditv
                self._resolve_best_coadd_ifs(ins, ima, lbda_ref, spec=spec, debug=False)
                new_coadd = obs['ima_coadd']

                # Rebuild spectral arrays for the new coadd
                sky_ph_square = sky_ph_spaxel * new_coadd**2
                dark_square   = dark_spaxel   * new_coadd**2
                ron_square    = ron_spaxel    * new_coadd**2
                if obs['ima_type'] == 'sb':
                    source_ph_square = source_ph_peak * new_coadd**2
                else:
                    # Recompute PSF if coadd parity changed (uneven flag differs)
                    new_uneven = 1 if new_coadd % 2 == 1 else 0
                    old_uneven = 1 if old_coadd % 2 == 1 else 0
                    if new_uneven != old_uneven:
                        _psf_new = self.get_image_psf(ins, snr_wave_actual, uneven=new_uneven)
                        if obs['ima_type'] == 'ps':
                            selected_image = _psf_new
                        else:  # resolved
                            selected_image = convolve_and_center(ima, _psf_new)
                    _fp, _fs = self.ifs_spaxel_aperture(ins, selected_image, N=new_coadd)
                    source_ph_peak   = factor_source * _fp
                    source_ph_square = factor_source * _fs

                if new_coadd == old_coadd:
                    break  # coadd converged

            if debug:
                self.logger.debug(f"Maximum DIT to avoid saturation: {dit_sat} seconds")
                self.logger.debug(f"Computed NDIT: {ndit_raw}, rounded to the next integer: {nditv}")
                self.logger.debug(f"Final NDIT: {nditv} and DIT: {ditv} to achieve SNR: {snrv} at wavelength: {wave_snr} AA (nearest to requested SNR wavelength: {obs['snr_wave']} AA), spbin: {obs['spbin']}")
                self.logger.debug(f"Overriding NDIT in the observation dictionary...")
                self.logger.debug(f"Overriding DIT in the observation dictionary...")

            res['ndit'] = nditv
            obs['ndit'] = nditv
            res['dit'] = ditv
            obs['dit'] = ditv
            res['dit_sat'] = dit_sat        # saturation-limited DIT (most efficient)
            res['ndit_raw'] = ndit_raw      # fractional NDIT before ceiling
            if _coadd_was_best:
                res['ima_coadd'] = obs['ima_coadd']  # optimal coadd integer, frozen

        res['input'] = dict(
            flux_source=spec, 
            atm_abs=ins_atm, 
            ins_trans=ins_ins, 
            atm_emi=ins_sky,
            QE_trans=ins['QE'],
            tel_trans=ins['telescope'],
            ins_noQE_trans=ins['total_instrumental'],
            total_trans = ins_ins*ins_atm
        )
        res['obs'] = obs

        end_time = time.time()

        if debug:
            self.logger.debug(f"Total processing time: {end_time - start_time} seconds")
        return res

    def time_from_source_mos(self, ins, ima, spec, debug=True, compute='dit'):
        
        obs = self.obs
        res = {}
        start_time = time.time()

        # unit conversion
        flux = 1.0
        # from per Angstrom to per spectel
        if obs['spec_type'] == 'cont':
            flux *= ins['dlbda']
            if debug:
                self.logger.debug(f"Flux converted from per Angstrom to per spectel by multiplying for dlbda: {ins['dlbda']}")

        if debug:
            if obs['skycalc']:
                self.logger.debug("Sky computed with skycalc")
            else:
                self.logger.info("Sky taken from static files")

        ins_sky = obs['skyemi']
        ins_ins = ins['instrans']
        ins_atm = obs['skyabs']

        # LSF convolution of the spectrum
        spec = spec.filter(width=ins['lsfpix'])
        if debug:
            self.logger.debug(f"Spectrum convolved with LSF of {ins['lsfpix']} pixels")

        # For time_from_source, we only need PSF at snr_wave, not full wave_grid
        wave = spec.wave.coord()
        snr_idx = np.abs(wave - obs['snr_wave']).argmin()
        snr_wave_actual = wave[snr_idx]

        # telescope effective area
        tel_eff_area = self.tel['effective_area_MOS']
        
        # pre-compute the conversion factor that is used in the SNR computation
        dl = spec.wave.get_step(unit='Angstrom')
        a = (wave*1.e-8/(H_cgs*C_cgs)) * (tel_eff_area*1.e4) * (ins_atm.data)
        Kt =  ins_ins.data * a
        Ksky = ins_ins.data * np.pi * (ins['aperture'] / 2)**2 * tel_eff_area * (dl/1e4)

        sky_ph_aperture = ins_sky * Ksky

        # we get the slicing only for the moshr
        if ins['name'] == 'moslr':
            num_trace = 1
        elif ins['name'] == 'moshr':
            num_trace = 7

        # number of vertical pixels needed to fully extract each trace in MOS HR and MOS LR
        trace_pixel_width = int(np.ceil(ins['lsfpix']))

        dark = ins['dcurrent'] / 3600 * num_trace * trace_pixel_width
        ron = ins['ron']**2 * num_trace * trace_pixel_width

        dark_tot = Spectrum(data=np.full(wave.shape, dark), wave=spec.wave)
        ron_tot = Spectrum(data=np.full(wave.shape, ron), wave=spec.wave)
        
        factor_source = spec * flux * Kt
        centering_eff = MOS_DEFAULT_CENTERING_EFF if obs['disp'] is None else 1.0
        fiber_injection_snr = centering_eff
        
        if obs['ima_type'] == 'sb':
            source_ph_aperture = factor_source * np.pi * (ins['aperture'] / 2)**2
        
        elif obs['ima_type'] in ['ps', 'resolved']:
            # Compute PSF only at snr_wave (1 wavelength instead of wave_grid)
            psf_single = self.get_image_psf(ins, snr_wave_actual)
            
            if obs['ima_type'] == 'ps':
                selected_image = psf_single
            
            elif obs['ima_type'] == 'resolved':
                if ima is None:
                    raise ValueError("For resolved sources, source image must not be None.")
                selected_image = convolve_and_center(ima, psf_single)

            # Compute fiber aperture fraction for single image
            displacement = 0.0 if obs['disp'] is None else obs['disp']
            frac_fiber_snr = self.mos_fiber_aperture(ins, selected_image, displacement=displacement)
            fiber_injection_snr = frac_fiber_snr * centering_eff

            # Apply fiber fraction to full spectrum (use snr_wave fraction for all)
            source_ph_aperture = factor_source * frac_fiber_snr

        snrv = obs['snr']

        if compute == 'dit': 
            _checkobs(self.obs, keys=['ndit', 'snr', 'snr_wave'])
            nditv = obs['ndit']

            if obs['spbin'] == 1:
                if debug:
                    self.logger.debug(f"Computing DIT without spectral rebinning")
                # nearest wave idx to snr_wave
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                wave_snr = wave[snr_idx]

                sv = source_ph_aperture.data[snr_idx]
                skyv = sky_ph_aperture.data[snr_idx]
                darkv = dark_tot.data[snr_idx]
                ronv = ron_tot.data[snr_idx]

            elif obs['spbin'] > 1:
                if debug:
                    self.logger.debug(f"Computing DIT with spectral rebinning, factor of {obs['spbin']}")
                # Find the bin containing snr_wave
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                bin_start = (snr_idx // obs['spbin']) * obs['spbin']
                bin_end = min(bin_start + obs['spbin'], len(wave))
                
                # Sum directly over the bin
                sv = np.sum(source_ph_aperture.data[bin_start:bin_end])
                skyv = np.sum(sky_ph_aperture.data[bin_start:bin_end])
                darkv = np.sum(dark_tot.data[bin_start:bin_end])
                ronv = np.sum(ron_tot.data[bin_start:bin_end])
                
                wave_snr = np.mean(wave[bin_start:bin_end])

            # we solve numerically for the DIT
            A = - (sv**2 * nditv) / (snrv**2)
            B = sv + skyv + darkv
            C = ronv

            roots = np.roots([A, B, C])
            ditv = roots[np.isreal(roots) & (roots > 0)].real[0]

            # saturation check: compute the max-DIT that avoids saturation
            counts_sat = (source_ph_aperture.data + sky_ph_aperture.data) / (num_trace * trace_pixel_width)
            dit_sat = threshold_sat / max(counts_sat)
            flag_sat = bool(ditv > dit_sat)
            if debug:
                self.logger.debug(f"Maximum DIT to avoid saturation: {dit_sat} seconds")
                if flag_sat:
                    self.logger.debug(f"WARNING: computed DIT ({ditv:.2f} s) exceeds saturation limit ({dit_sat:.2f} s)")
                self.logger.debug(f"Computed DIT: {ditv} seconds for NDIT: {nditv} to achieve SNR: {snrv} at wavelength: {wave_snr} AA (nearest to requested SNR wavelength: {obs['snr_wave']} AA), with spectral rebinning factor: {obs['spbin']}")
                self.logger.debug(f"Overriding DIT in the observation dictionary...")
            res['dit'] = ditv
            res['dit_sat'] = dit_sat
            res['flag_sat'] = flag_sat
            obs['dit'] = ditv

        elif compute == 'ndit':
            _checkobs(self.obs, keys=['dit', 'snr', 'snr_wave'])
            ditv = obs['dit']
             
            if obs['spbin'] == 1:
                if debug:
                    self.logger.debug(f"Computing NDIT without spectral rebinning")
                # nearest wave idx to snr_wave
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                wave_snr = wave[snr_idx]

                sv = source_ph_aperture.data[snr_idx]
                skyv = sky_ph_aperture.data[snr_idx]
                darkv = dark_tot.data[snr_idx]
                ronv = ron_tot.data[snr_idx] 
            
            elif obs['spbin'] > 1:
                if debug:
                    self.logger.debug(f"Computing NDIT with spectral rebinning, factor of {obs['spbin']}")
                # Find the bin containing snr_wave
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                bin_start = (snr_idx // obs['spbin']) * obs['spbin']
                bin_end = min(bin_start + obs['spbin'], len(wave))
                
                # Sum directly over the bin
                sv = np.sum(source_ph_aperture.data[bin_start:bin_end])
                skyv = np.sum(sky_ph_aperture.data[bin_start:bin_end])
                darkv = np.sum(dark_tot.data[bin_start:bin_end])
                ronv = np.sum(ron_tot.data[bin_start:bin_end])
                
                wave_snr = np.mean(wave[bin_start:bin_end])

            nditv = snrv**2 * (sv + skyv + darkv + ronv / ditv) / (sv**2 * ditv)

            # saturation check: compute the max-DIT that avoids saturation
            counts_sat = (source_ph_aperture.data + sky_ph_aperture.data) / (num_trace * trace_pixel_width)
            dit_sat = threshold_sat / max(counts_sat)
            flag_sat = bool(ditv > dit_sat)
            if debug:
                self.logger.debug(f"Maximum DIT to avoid saturation: {dit_sat} seconds")
                if flag_sat:
                    self.logger.debug(f"WARNING: computed DIT ({ditv:.2f} s) exceeds saturation limit ({dit_sat:.2f} s)")
                self.logger.debug(f"Computed NDIT: {nditv} exposures for DIT: {ditv} to achieve SNR: {snrv} at wavelength: {wave_snr} AA (nearest to requested SNR wavelength: {obs['snr_wave']} AA), with spectral rebinning factor: {obs['spbin']}")
                self.logger.debug(f"Overriding NDIT in the observation dictionary...")
            res['ndit'] = nditv
            res['dit_sat'] = dit_sat
            res['flag_sat'] = flag_sat
            obs['ndit'] = nditv

        elif compute == 'best':
            _checkobs(self.obs, keys=['snr', 'snr_wave'])
            snrv = obs['snr']

            if obs['spbin'] == 1:
                if debug:
                    self.logger.debug(f"Computing best DITxNDIT combination without spectral rebinning")
                # nearest wave idx to snr_wave
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                wave_snr = wave[snr_idx]

                sv = source_ph_aperture.data[snr_idx]
                skyv = sky_ph_aperture.data[snr_idx]
                darkv = dark_tot.data[snr_idx]
                ronv = ron_tot.data[snr_idx] 
            
            elif obs['spbin'] > 1:
                if debug:
                    self.logger.debug(f"Computing best DITxNDIT combination with spectral rebinning, factor of {obs['spbin']}")
                # Find the bin containing snr_wave
                snr_idx = np.abs(wave - obs['snr_wave']).argmin()
                bin_start = (snr_idx // obs['spbin']) * obs['spbin']
                bin_end = min(bin_start + obs['spbin'], len(wave))
                
                # Sum directly over the bin
                sv = np.sum(source_ph_aperture.data[bin_start:bin_end])
                skyv = np.sum(sky_ph_aperture.data[bin_start:bin_end])
                darkv = np.sum(dark_tot.data[bin_start:bin_end])
                ronv = np.sum(ron_tot.data[bin_start:bin_end])
                
                wave_snr = np.mean(wave[bin_start:bin_end])
                
            # we compute the maximum DIT to avoid saturation
            counts = (source_ph_aperture.data + sky_ph_aperture.data) / (num_trace * trace_pixel_width)
            dit_sat = threshold_sat / max(counts)

            # now we compute the NDIT to achieve the target SNR with this DIT
            ndit_raw = snrv**2 * (sv + skyv + darkv + ronv / dit_sat) / (sv**2 * dit_sat)

            # we approximate NDIT to the next integer
            nditv = max(1, int(np.ceil(ndit_raw)))
            
            if debug:
                self.logger.debug(f"Maximum DIT to avoid saturation: {dit_sat} seconds")
                self.logger.debug(f"Computed NDIT: {ndit_raw}, rounded to the next integer: {nditv}")

            # we solve numerically for the DIT
            A = - (sv**2 * nditv) / (snrv**2)
            B = sv + skyv + darkv
            C = ronv

            roots = np.roots([A, B, C])
            ditv = roots[np.isreal(roots) & (roots > 0)].real[0]

            if debug:
                self.logger.debug(f"Final NDIT: {nditv}  and exposures for DIT: {ditv} to achieve SNR: {snrv} at wavelength: {wave_snr} AA (nearest to requested SNR wavelength: {obs['snr_wave']} AA), with spectral rebinning factor: {obs['spbin']}")
                self.logger.debug(f"Overriding NDIT in the observation dictionary...")
                self.logger.debug(f"Overriding DIT in the observation dictionary...")
            
            res['ndit'] = nditv
            obs['ndit'] = nditv

            res['dit'] = ditv
            obs['dit'] = ditv
            res['dit_sat'] = dit_sat        # saturation-limited DIT (most efficient)
            res['ndit_raw'] = ndit_raw      # fractional NDIT before ceiling

        fiber_injection_full = Spectrum(data=np.full(wave.shape, fiber_injection_snr), wave=spec.wave)
        res['input'] = dict(
            flux_source=spec, 
            atm_abs=ins_atm, 
            ins_trans=ins_ins, 
            atm_emi=ins_sky,
            QE_trans=ins['QE'],
            tel_trans=ins['telescope'],
            ins_noQE_trans=ins['total_instrumental'],
            fiber_injection=fiber_injection_full,
            total_trans=ins_ins * ins_atm * fiber_injection_full
        )

        res['obs'] = obs

        end_time = time.time()

        if debug:
            self.logger.debug(f"Total processing time: {end_time - start_time} seconds")
        return res

    # # # # # # # # # # # # # # # # # # #
    # # # # SPECTRAL WINDOW TOOLS  # # # #
    # # # # # # # # # # # # # # # # # # #

    def time_from_source_window(self, ins, ima, spec, lam1, lam2,
                                target_snr, unit='pixel', compute='ndit',
                                n_iter=20, debug=True):
        """Find the DIT or NDIT that achieves a target *median* SNR in a
        spectral window [lam1, lam2].

        This is an iterative wrapper around :meth:`snr_from_source` that
        scales the exposure parameter quadratically until the median SNR in
        the requested wavelength window converges to ``target_snr``.

        Parameters
        ----------
        ins : dict
            Instrument configuration dict (e.g. ``wst.ifs['blue']``).
        ima : MPDAF image or None
            Source image (None for sb/ps).
        spec : MPDAF spectrum
            Source spectrum.
        lam1 : float
            Start wavelength of the window (Angstrom).
        lam2 : float
            End wavelength of the window (Angstrom).
        target_snr : float
            Desired median SNR inside the window.
        unit : str
            ``'pixel'`` — SNR per spectral pixel (default and only supported option).
        compute : str
            ``'ndit'`` — fix DIT, find NDIT (default).
            ``'dit'``  — fix NDIT, find DIT.
        n_iter : int
            Maximum number of scaling iterations (default 20).
        debug : bool
            Log iteration details (default True).

        Returns
        -------
        dict with keys:
            - ``'ndit'`` or ``'dit'``: the required value (int for NDIT, float for DIT)
            - ``'median_snr'``: achieved median SNR in the window
            - ``'lam1'``, ``'lam2'``: the wavelength window used
            - ``'unit'``: SNR unit used (always ``'pixel'``)
            - ``'res'``: the full :meth:`snr_from_source` result for the final iteration
        """
        obs = self.obs
        dlbda = float(ins.get('dlbda', 1.0))

        if unit != 'pixel':
            raise ValueError("Only unit='pixel' is supported")

        if compute == 'ndit':
            param_key = 'ndit'
            param_start = max(1, float(obs.get('ndit') or 1))
        elif compute == 'dit':
            param_key = 'dit'
            param_start = max(0.1, float(obs.get('dit') or 600))
        else:
            raise ValueError(f"compute must be 'ndit' or 'dit', got '{compute}'")

        param_val = param_start
        res = None
        med = None
        evaluated = {}   # ndit_int → med, for bracket detection (NDIT)
        _history = []    # (param_val, med) pairs for secant acceleration (DIT)

        for i in range(n_iter):
            obs[param_key] = max(1, int(round(param_val))) if compute == 'ndit' else max(0.1, float(param_val))
            res = self.snr_from_source(ins, ima, spec, debug=False)
            if 'message' in res:
                raise RuntimeError(f"snr_from_source error: {res['message']}")
            med = snr_in_window(res, lam1, lam2, dlbda=dlbda, unit=unit)
            if med is None or med <= 0:
                raise ValueError(f"No valid SNR data in window [{lam1}–{lam2}] Å")
            scale = (target_snr / med) ** 2
            new_val = param_val * scale
            if compute == 'dit':
                new_val = max(0.1, new_val)
                # Secant acceleration: interpolate linearly from last two points
                _history.append((param_val, med))
                if len(_history) >= 2:
                    p1, s1 = _history[-2]
                    p2, s2 = _history[-1]
                    if abs(s2 - s1) > 1e-10:
                        secant = p1 + (target_snr - s1) * (p2 - p1) / (s2 - s1)
                        if secant > 0:
                            new_val = secant
            if debug:
                self.logger.debug(
                    f"  iter {i+1}: {param_key}={obs[param_key]}, "
                    f"median SNR/{unit}={med:.3f}, scale={scale:.4f} → {param_key}={new_val:.6f}"
                )
            param_val = new_val
            # NDIT sub-1: pinned at obs=1 every iteration — one analytical step is exact
            if compute == 'ndit' and param_val < 1.0:
                break
            # Converged: SNR within 0.01% of target
            if abs(med - target_snr) / target_snr < 1e-4:
                break
            # Bracket detection for NDIT: if consecutive integers N and N+1 are
            # both evaluated and straddle the target, the answer is ceil = N+1.
            if compute == 'ndit':
                evaluated[obs[param_key]] = med
                n = int(param_val)
                if n >= 1 and n in evaluated and (n + 1) in evaluated:
                    if evaluated[n] < target_snr < evaluated[n + 1]:
                        param_val = float(n + 1)
                        break

        # Final value
        param_val_raw = param_val
        obs[param_key] = max(1, int(np.ceil(param_val))) if compute == 'ndit' else float(param_val)
        if debug and compute == 'ndit' and abs(param_val_raw - obs[param_key]) > 0.01:
            self.logger.debug(
                f"  NDIT: {param_val_raw:.2f} → ceiled to {obs[param_key]}"
            )
        res = self.snr_from_source(ins, ima, spec, debug=debug)
        if 'message' in res:
            raise RuntimeError(f"snr_from_source error: {res['message']}")
        med = snr_in_window(res, lam1, lam2, dlbda=dlbda, unit=unit)

        if debug:
            self.logger.debug(
                f"  → Required {param_key}: {obs[param_key]}, "
                f"achieved median SNR/{unit} in [{lam1}–{lam2}] Å: {med:.3f}"
            )

        result = {
            param_key: obs[param_key],
            'median_snr': med,
            'lam1': lam1,
            'lam2': lam2,
            'unit': unit,
            'res': res
        }
        if compute == 'ndit':
            result['ndit_raw'] = param_val_raw
        if compute == 'dit' and abs(obs['dit'] - 0.1) < 1e-6:
            # DIT hit the minimum floor — source too bright to reach target SNR
            result['dit_at_min_floor'] = True
        return result

# # # # # # # # # # # # # # # #
# # # # GENERAL METHODS # # # #
# # # # # # # # # # # # # # # #

# function to get the static sky tables & the tel.+inst. transmission curves, they should always be present in the right directory
def snr_in_window(res, lam1, lam2, dlbda=None, unit='pixel', stat='median'):
    """Compute a summary statistic of the SNR inside a spectral window.

    Parameters
    ----------
    res : dict
        Result dictionary returned by :meth:`ETC.snr_from_source`.
    lam1 : float
        Start wavelength of the window in Angstrom.
    lam2 : float
        End wavelength of the window in Angstrom.
    dlbda : float or None
        Dispersion in Å/pixel (kept for backward compatibility; not used).
    unit : str
        ``'pixel'`` — return SNR per spectral pixel (default and only supported option).
    stat : str
        Summary statistic: ``'median'`` (default) or ``'mean'``.

    Returns
    -------
    float or None
        The requested statistic of the SNR values inside the window,
        or *None* if no data fall within [lam1, lam2].
    """
    try:
        snr_spec = res['spec']['snr']
        wave = snr_spec.wave.coord()
        snr_data = snr_spec.data.data
    except (KeyError, AttributeError):
        return None

    mask = (wave >= lam1) & (wave <= lam2)
    if not np.any(mask):
        return None

    vals = snr_data[mask].astype(float)

    if unit != 'pixel':
        raise ValueError("Only unit='pixel' is supported")

    if stat == 'median':
        return float(np.nanmedian(vals))
    elif stat == 'mean':
        return float(np.nanmean(vals))
    else:
        raise ValueError(f"stat must be 'median' or 'mean', got '{stat}'")



# # # image generation functions # # #

def sersic(samp, reff, n, ell=0, kreff=4, oversamp=10, uneven=1):
    """ compute a 2D Sersic image

    Parameters
    ----------
    samp : float
        image sampling in arcsec
    reff : float
        effective radius (arcsec)
    n : float
        Sersic index (4 for elliptical, 1 for elliptical disk)
    ell : float
         image ellipticity (Default value = 0)
    kreff : float
         factor relative to the effective radius to compute the size of the image (Default value = 5)
    oversamp : int
         oversampling factor (Default value = 10)
    uneven : int
         if 1 the image size will have an uneven number of spaxels (Default value = 1)

    Returns
    -------
    MPDAF image
         Sersic image
    """

    ns = (int((kreff*reff/samp+1)/2)*2 + uneven)*oversamp
    pixreff = oversamp*reff/samp          
    x,y = np.meshgrid(np.arange(ns), np.arange(ns))
    x0,y0 = ns/2-0.5,ns/2-0.5
    
    mod = Sersic2D(amplitude=1, r_eff=pixreff, n=n, x_0=x0, y_0=y0,
                   ellip=ell, theta=0)
    data = mod(x, y)            
    ima = Image(data=data)
    ima.data /= ima.data.sum()

    # copy the WCS from a dummy Moffat since the Sersic does not have it 
    dummy = moffat_image(fwhm=(1,1), n=10, shape=(ns,ns), flux=1.0, unit_fwhm=None)
    ima.wcs = dummy.wcs
    ima.oversamp = oversamp
    return ima

def moffat(samp, fwhm, beta, ell=0, kfwhm=5, oversamp=10, uneven=1):
    """ compute a 2D Moffat image

    Parameters
    ----------
    samp : float
        image sampling in arcsec
    fwhm : float
        FWHM of the MOFFAT (arcsec)
    beta : float
        MOFFAT shape parameter (beta > 4 for Gaussian, 1 for Lorentzien)
    ell : float
         image ellipticity (Default value = 0)
    kfwhm : float
         factor relative to the FWHM to compute the size of the image (Default value = 4)
    oversamp : int
         oversampling factor (Default value = 10)
    uneven : int
         if 1 the image size will have an uneven number of spaxels (Default value = 1)

    Returns
    -------
    MPDAF image
         MOFFAT image
    """

    ns = (int((kfwhm*fwhm/samp+1)/2)*2 + uneven)*oversamp
    pixfwhm = oversamp*fwhm/samp
    pixfwhm2 = pixfwhm*(1-ell)
    ima = moffat_image(fwhm=(pixfwhm2,pixfwhm), n=beta, shape=(ns,ns), flux=1.0, unit_fwhm=None)
    ima.data /= ima.data.sum()
    ima.oversamp = oversamp
    return ima

# # # check functions # # #

def _checkline(cen, fwhm, M_min, M_max):
    """ check that the line is inside M_min and M_max """
    if cen > M_max:
        print('Line outside the last pixel!')
    elif cen + fwhm > M_max:
        print('Line near the last pixel!')
    if cen < M_min:
        print('Line outside the first pixel!')
    elif cen - fwhm < M_min:
        print('Line near the first pixel!')
    return

def _checkrange(arr, M_min, M_max):
    """ check that the range is inside M_min and M_max """
    if arr[0] > M_min:
        print('Trace starts after the first pixel!')
    if arr[-1] < M_max:
        print('Trace ends before the last pixel!')
    return

def _checkobs(obs, keys):
    """ check existence of keywords """
    for key in keys:
        if key not in obs.keys() or obs[key] is None:
            raise KeyError(f'keyword {key} missing/None in obs dictionary')
    return

# seeing computation function, fwhm at a specific wavelength/array of wavelengths
def get_seeing_fwhm(seeing, airmass, wave, diam, iq_tel, iq_ins, glao=False):
    """ compute FWHM for the Paranal ESO ETC model

    Parameters
    ----------
    seeing : float
        seeing (arcsec) at 5000A — ignored when glao=True (IFS GLAO formula used)
    airmass : float
        airmass of the observation
    wave : numpy array of float
        wavelengths in Angstrom
    diam : float
        telescope primary mirror diameter in m
    iq_tel : float of numpy array
        image quality of the telescope
    iq_ins : float of numpy array
        image quality of the instrument
    glao : bool
        if True, use the GLAO IQ formula instead of the natural-seeing model.
        The IFS GLAO delivered IQ (arcsec) as a function of wavelength is:
          IQ_glao(wave) = (A * wave_nm^2 + B * wave_nm + C) * AM^0.6
        where wave_nm is the wavelength in nanometres,
        A = 1.22465e-7, B = -0.000576386, C = 0.717164.
        This formula already accounts for the average atmospheric correction;
        the 'seeing' argument is not used.

    Returns
    -------
    numpy array of float
        FWHM (arcsec) as function of wavelengths

    """

    if glao:
        # GLAO delivered image quality (IFS mode).
        # Quadratic polynomial fit in wave_nm (wavelength in nanometres).
        # Coefficients from WST GLAO model.
        # The 'seeing' argument is intentionally ignored here.
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "get_seeing_fwhm: GLAO active — user seeing (%s) ignored, using GLAO polynomial",
            seeing
        )
        A_glao, B_glao, C_glao = 1.22465e-7, -0.000576386, 0.717164
        wave_nm = np.atleast_1d(wave).astype(float) / 10.0   # Å → nm
        iq_atm = (A_glao * wave_nm**2 + B_glao * wave_nm + C_glao) * airmass**0.6
        iq_atm = np.maximum(iq_atm, 0.05)  # physical safety floor
    else:
        import logging as _logging
        _logging.getLogger(__name__).info(
            "Compute iq with seeing (%s) ",seeing
        )
        # from ESPRESSO (Schmidt+24)
        r0 = 0.1*seeing**(-1)*(wave/5000)**(1.2)*airmass**(-0.6)
        l0 = 46 # for VLT (in ETC)

        Fkolb = 1/(1+300*diam/l0)-1
        iq_atm = seeing*(wave/5000)**(-1/5)*airmass**(3/5) * np.sqrt(1+Fkolb*2.183*(r0/l0)**0.356)

    iq = np.sqrt(iq_atm**2 + iq_tel**2 + iq_ins**2)
    iq_before_ins = np.sqrt(iq_atm**2 + iq_tel**2)

    return iq, iq_before_ins

# handy function to compute the sky spectra for different airmasses and moon phases
# if an update of the static sky files is needed
def compute_sky(outdir):
    """
    Computes sky spectra for different airmasses and moon phases using SkyCalc,
    and saves each spectrum as a FITS file in the specified output directory.
    
    Parameters
    ----------
    outdir : str
        Path to the output directory where the FITS files will be saved.
    
    Notes
    -----
    - Uses fixed wavelength grid from 300 to 2500 nm with 0.01 nm step.
    - Moon phases are mapped 1:1 to moon-sun separations:
      'darksky'->0 deg, 'greysky'->90 deg, 'brightsky'->180 deg.
    - Saves one FITS file per airmass per moon phase.
    """
    if _SkyCLIModel is None:
        raise ImportError("skycalc_cli is required to compute sky spectra")

    os.makedirs(outdir, exist_ok=True)

    all_moons = ['darksky', 'greysky', 'brightsky']
    all_mss = [0, 90, 180]
    all_airmass = [1.0, 1.2, 1.5, 2.0]
    all_pwv = [1.0, 3.5, 10.0]

    for am in all_airmass:
        for moon, mss in zip(all_moons, all_mss):
            for pwv in all_pwv:
                _skm = _SkyCLIModel()
                _skm.stop_on_errors_and_exceptions = False
                _skm.callwith({
                    'msolflux': 130.0,
                    'observatory': 'paranal',
                    'wgrid_mode': 'fixed_wavelength_step',
                    'wmin': 300.0,
                    'wmax': 1200.0,
                    'wdelta': 0.01,
                    'airmass': am,
                    'moon_sun_sep': float(mss),
                    'pwv': pwv,
                })
                if _skm.data is None:
                    print(f"SkyCalc failed for AM={am}, moon={moon}, PWV={pwv}; skipping.")
                    continue
                tbl = Table.read(BytesIO(_skm.data), format='fits')
                tbl.meta['AIRMASS'] = am
                tbl.meta['MOONPH'] = moon
                tbl.meta['MSS'] = mss
                tbl.meta['PWV'] = pwv
                fname = f"{moon}_{am:.1f}_{pwv:.1f}.fits"
                outpath = os.path.join(outdir, fname)
                tbl.write(outpath, format='fits', overwrite=True)
                print(f"Saved: {outpath}")

    print("\nAll sky spectra successfully saved.")

# # # spectrum masking functions # # #

def mask_spectrum_edges(spectrum, N):
    spectrum[:N] = spectrum[N]
    spectrum[-N:] = spectrum[-N-1]
    return spectrum

def mask_line_region(spectrum, wave, center, fwhm, n_fwhm=4):
    """Set spectrum to zero outside +/- n_fwhm * FWHM from center."""
    mask = (wave >= center - n_fwhm*fwhm) & (wave <= center + n_fwhm*fwhm)
    new_data = np.where(mask, spectrum.data, 0)
    spectrum.data = new_data
    return spectrum

def mask_spectra_in_dict(d, center, fwhm, n_fwhm=4):
    """Apply line region masking to all Spectrum objects in a nested dictionary."""
    for key, val in d.items():
        if isinstance(val, Spectrum):
            mask_line_region(val, val.wave.coord(), center, fwhm, n_fwhm)
        elif isinstance(val, dict):
            mask_spectra_in_dict(val, center, fwhm, n_fwhm)

# useful function to convolve image for resolved case with PSF and center it, used for the IFS case
def convolve_and_center(ima, impsf):
    """
    Convolve an image with a PSF and center the result.
    """
    from mpdaf.obj import Image

    # Perform convolution using FFT
    conv_data = fftconvolve(ima.data, impsf.data, mode='full')
    # Find the position of the maximum value
    maxpos = np.unravel_index(np.argmax(conv_data), conv_data.shape)
    # Calculate the shift needed to center the maximum
    ny, nx = conv_data.shape
    shift_y = (ny // 2) - maxpos[0]
    shift_x = (nx // 2) - maxpos[1]
    # Shift the image
    conv_data_centered = np.roll(conv_data, shift=(shift_y, shift_x), axis=(0,1))
    # Normalize
    conv_data_centered /= conv_data_centered.sum()
    # Create new MPDAF image with correct shape
    conv_ima = Image(data=conv_data_centered)
    conv_ima.oversamp = ima.oversamp
    return conv_ima

# function to plot the noise components, we can call the res['spec']['noise'] or res['peak']['noise'] dictionary
def plot_noise_components(spec_dict):
    """
    Plot the total noise and its fractional components.

    Parameters
    ----------
    spec_dict : dict
        Dictionary with keys like 'tot', 'frac_source', 'frac_sky', 'frac_dark', 'frac_ron'.
        Each value should be a Spectrum object with `.wave.coord()` and `.data.data`.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})

    # --- Top panel: total noise ---
    wave = spec_dict['tot'].wave.coord()
    flux = spec_dict['tot'].data.data
    ax1.plot(wave, flux, color='black', lw=1.5)
    ax1.set_ylabel('Total noise')
    ax1.set_title('Total noise and fractional contributions')
    ax1.grid(True, ls='--', alpha=0.3)

    # --- Bottom panel: fractional components ---
    colors = {
        'frac_source': 'tab:blue',
        'frac_sky': 'tab:orange',
        'frac_dark': 'tab:green',
        'frac_ron': 'tab:red'
    }

    for key, color in colors.items():
        if key in spec_dict:
            w = spec_dict[key].wave.coord()
            y = spec_dict[key].data.data
            ax2.plot(w, y, label=key.replace('frac_', ''), color=color)

    ax2.set_xlabel('Wavelength')
    ax2.set_ylabel('Fraction')
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()

import numpy as np

# function to simulate the 1d spectra including noise components
def simulate_counts(npix, source=None, sky=None, dark=None, RON=None, seed=None):
    """
    Simulate the total observed counts within a number of pixels,
    assuming uniform per-pixel values for source, sky, and dark current.

    Parameters
    ----------
    npix : int
        number of pixels.
    source : float
        Mean source counts per pixel.
    sky : float
        Mean sky counts per pixel.
    dark : float
        Mean dark current counts per pixel.
    RON : float
        Read-Out Noise (standard deviation of Gaussian noise per pixel).
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    total_counts : float
        Total observed counts in the NxN region (including noise).
    """
    rng = np.random.default_rng(seed)

    # Mean signal per pixel
    mean_signal = source + sky + dark

    # Photon and dark noise (Poisson distributed)
    # For very large lambda, use Gaussian approximation (Poisson -> Normal)
    POISSON_LIMIT = 1e15
    if np.any(np.asarray(mean_signal) > POISSON_LIMIT):
        poisson_counts = rng.normal(mean_signal, np.sqrt(np.maximum(mean_signal, 0)), size=npix)
        poisson_counts = np.maximum(poisson_counts, 0)
    else:
        poisson_counts = rng.poisson(mean_signal, size=npix)

    # Add read-out noise (Gaussian distributed)
    noisy_counts = poisson_counts + rng.normal(0, RON, size=npix)

    # Totals
    total_counts = noisy_counts.sum()

    return total_counts

# simulate counts function vectorized
def simulate_counts_vectorized(npix, source_arr, sky_arr, dark, RON, seed=None):
    """Vectorized simulation of total observed counts for all wavelength pixels at once."""
    rng = np.random.default_rng(seed)
    n_wave = len(source_arr)
    mean_signal = source_arr + sky_arr + dark
    mean_2d = np.broadcast_to(mean_signal[:, np.newaxis], (n_wave, npix))
    # For very large lambda, use Gaussian approximation (Poisson -> Normal)
    POISSON_LIMIT = 1e15
    if np.any(mean_signal > POISSON_LIMIT):
        poisson_counts = rng.normal(mean_2d, np.sqrt(np.maximum(mean_2d, 0)))
        poisson_counts = np.maximum(poisson_counts, 0)
    else:
        poisson_counts = rng.poisson(mean_2d)
    ron_noise = rng.normal(0, RON, size=(n_wave, npix))
    noisy_counts = poisson_counts + ron_noise
    total_counts = noisy_counts.sum(axis=1)
    return total_counts

# Alias to maintain retro comp.
get_data = ETC.get_data

# # # # # # # # # # # # # # # #

def lsf_resolution(ins, wavelength_AA: float) -> float:
    """
    return the spectral resolution R = lambda / delta_lambda of the LSF at a given wavelength in Angstrom
    """

    lstep_AA = ins['instrans'].get_step()
    fwhm_AA = ins['lsfpix'] * lstep_AA

    return float(wavelength_AA) / float(fwhm_AA)



# # # # # # # MORE # # # # # #
# Add a way to easily recompute the static sky files if needed, and maybe to implement transmission curves too
# Add the rounding of computed NDIT to the nearest integer in the time_from_source methods, this is done only in the best case now
# Add a way to not compute again the snr_from_source from scratch when computing time_from_source, we have everything we need there (fractions, source counts, sky counts, etc.) > we can just add a flag to save everything in the res dictionaries
# There could be problems when requesting a SNR at a wavelength near the edge of the spectrum, in case of rebinning this could lead to errors, we should add checks for that
# # # # # # # # # # # # # # #