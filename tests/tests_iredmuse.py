import numpy as np
import pytest

from pyetc_iredmuse.iredmuse import iredMUSE
from pyetc_iredmuse.etc import (
    get_seeing_fwhm,
    mask_spectrum_edges,
    mask_line_region,
    mask_spectra_in_dict,
    convolve_and_center,
    snr_in_window,
    sersic,
    moffat,
)
from mpdaf.obj import Spectrum, Image, WaveCoord


def test_get_seeing_fwhm_natural_seeing():
    wave = np.array([9000, 11000, 13000])
    iq, iq_before = get_seeing_fwhm(
        seeing=0.8,
        airmass=1.2,
        wave=wave,
        diam=8.0,
        iq_tel=0.1,
        iq_ins=0.2,
        glao=False,
    )

    assert iq.shape == wave.shape
    assert iq_before.shape == wave.shape
    assert np.all(iq >= iq_before)
    assert np.all(iq > 0)


def test_get_seeing_fwhm_glao_ignores_seeing():
    wave = np.array([9000.0, 11000.0, 13000.0])
    iq_default, _ = get_seeing_fwhm(
        seeing=0.5,
        airmass=1.0,
        wave=wave,
        diam=8.0,
        iq_tel=0.1,
        iq_ins=0.2,
        glao=True,
    )
    iq_changed, _ = get_seeing_fwhm(
        seeing=2.0,
        airmass=1.0,
        wave=wave,
        diam=8.0,
        iq_tel=0.1,
        iq_ins=0.2,
        glao=True,
    )

    assert np.allclose(iq_default, iq_changed)


def make_sample_spectrum(num_pixels=20):
    wave = WaveCoord(cdelt=10.0, crval=5000.0, cunit='angstrom')
    data = np.arange(num_pixels, dtype=float)
    return Spectrum(data=data, wave=wave)


def test_mask_spectrum_edges():
    spectrum = np.arange(10, dtype=float)
    masked = mask_spectrum_edges(spectrum.copy(), 2)
    assert masked[0] == masked[2]
    assert masked[1] == masked[2]
    assert masked[-1] == masked[-3]
    assert masked[-2] == masked[-3]


def test_mask_line_region():
    spectrum = make_sample_spectrum(10)
    result = mask_line_region(spectrum, spectrum.wave.coord(), center=5040.0, fwhm=20.0, n_fwhm=1)
    data = result.data
    assert data[0] == 0
    assert data[-1] == 0
    assert np.any(data > 0)


def test_mask_spectra_in_dict_deep():
    spec = make_sample_spectrum(10)
    nested = {'a': spec, 'b': {'inner': make_sample_spectrum(10)}}
    mask_spectra_in_dict(nested, center=5040.0, fwhm=20.0, n_fwhm=1)
    assert np.all(nested['a'].data[:3] == 0)
    assert np.all(nested['b']['inner'].data[:3] == 0)


def test_convolve_and_center_normalizes_and_centers():
    ima = Image(data=np.ones((5, 5)))
    psf = Image(data=np.ones((3, 3)))
    conv = convolve_and_center(ima, psf)
    assert np.isclose(conv.data.sum(), 1.0)
    maxpos = np.unravel_index(np.argmax(conv.data), conv.data.shape)
    center = (conv.data.shape[0] // 2, conv.data.shape[1] // 2)
    assert maxpos == center


def test_snr_in_window_median_and_mean():
    wave = WaveCoord(cdelt=10.0, crval=5000.0, cunit='angstrom')
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    spec = Spectrum(data=data, wave=wave)
    res = {'spec': {'snr': spec}}

    assert snr_in_window(res, 5000.0, 5020.0, stat='median') == 2.0
    assert snr_in_window(res, 5000.0, 5030.0, stat='mean') == pytest.approx(3.0)
    assert snr_in_window(res, 5100.0, 5200.0) is None


def test_sersic_returns_image():
    img = sersic(0.1, reff=0.5, n=2.0, ell=0.2, oversamp=3, uneven=1)
    assert isinstance(img, Image)
    assert np.isclose(img.data.sum(), 1.0)
    assert img.data.shape[0] == img.data.shape[1]


def test_moffat_returns_image():
    img = moffat(0.1, fwhm=1.0, beta=2.5, ell=0.1, oversamp=3, uneven=1)
    assert isinstance(img, Image)
    assert np.isclose(img.data.sum(), 1.0)
    assert img.data.shape[0] == img.data.shape[1]


def test_snr_in_window_invalid_stat():
    wave = WaveCoord(cdelt=10.0, crval=5000.0, cunit='angstrom')
    data = np.ones(5)
    spec = Spectrum(data=data, wave=wave)
    res = {'spec': {'snr': spec}}
    with pytest.raises(ValueError):
        snr_in_window(res, 5000.0, 5100.0, stat='sum')
