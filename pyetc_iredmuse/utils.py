import numpy as np

def add_gaussian(wrange, fluxval, lpeak, flux, fwhm):
    """Add a gaussian on spectrum in place.
    Parameters
    ----------
    lpeak : float
        Gaussian center.
    flux : float
        Integrated gaussian flux or gaussian peak value if peak is True.
    fwhm : float
        Gaussian fwhm.
    """
    gauss = lambda p, x: p[0] * (1 / np.sqrt(2 * np.pi * (p[2] ** 2))) * np.exp(-(x - p[1]) ** 2 / (2 * p[2] ** 2))
    line = gauss([flux, lpeak, fwhm / 2.35], wrange)
    fluxval += line
    return line


def rebin(a, newshape):
    '''
        Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old) / new) for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')  # choose the biggest smaller integer index
    return