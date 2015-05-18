"""
SPECTOOLS contains useful functions for handling spectroscopic data


Author                 Version      Date
-----------------------------------------------
S. M. Crawford (SAAO)    1.0        8 Nov 2009

"""
import copy
from astropy.io import fits
import numpy as np
from scipy import signal
from scipy import interpolate as scint
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import minimize
from iterfit import iterfit
import WavelengthSolution

from PySpectrograph.Spectra import Spectrum

import pylab as pl

from . import SpecError



default_kernal = [0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0]


def mcentroid(xarr, yarr, kern=default_kernal, xc=None, xdiff=None):
    """Find the centroid of a line following a similar algorithm as
       the centroid algorithm in IRAF.   xarr and yarr should be an area
       around the desired feature to be centroided.  The default kernal
       is used if the user does not specific one.

       The algorithm solves for the solution to the equation

       ..math:: \int (I-I_0) f(x-x_0) dx = 0

    Parameters
    ----------
    xarr: numpy.ndarry
        array of x values

    yarr: numpy.ndarry
        array of y values

    kern: numpy.ndarray
         kernal to convolve the array with.  It has a default shape
         of [0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0]

    xc: int
        Initial guess

    xdiff: int
        Pixels around xc to use for convolution


    Returns
    -------
    cx: float
       Centroided x-value for feature

    """

    if xdiff < len(kern):
        xdiff = len(kern)
 
    if xc is not None and xdiff:
        mask = (abs(xarr - xc) < xdiff)
    else:
        mask = np.ones(len(xarr), dtype=bool)

    # convle the input array with the default kernal
    warr = np.convolve(yarr[mask], kern, mode='same')

    # interpolate the results
    # imask is used to make sure we are only gettin the
    # center pixels
    imask = (abs(xarr[mask]-xarr[mask].mean()) < 3)
    cx = np.interp(0, warr[imask], xarr[mask][imask])
    return cx


def interpolate(x, x_arr, y_arr, type='interp', order=3, left=None,
                right=None):
    """Perform interpolation on value x using arrays x_arr
       and y_arr.  The type of interpolate is defined by interp

       type:
       interp--use numpy.interp
       spline--use scipy.splrep and splev

       return
    """
    if type == 'interp':
        y = np.interp(x, x_arr, y_arr, left=left, right=right)
    if type == 'spline':
        if left is None:
            y_arr[0] = left
        if right is None:
            y_arr[-1] = right

        tk = scint.splrep(x_arr, y_arr, k=order)
        y = scint.splev(x, tk, der=0)

    return y


def clipstats(yarr, thresh, iter):
    """Return sigma-clipped mean of yarr"""
    mean = yarr.mean()
    std = yarr.std()
    for i in range(iter):
        mask = (abs(yarr - mean) < thresh * std)
        if mask.sum() <= 1:
            return yarr.mean(), yarr.std()
        mean = yarr[mask].mean()
        std = yarr[mask].std()

    return mean, std


def find_points(xarr, farr, kernal_size=3, sections=0):
    """Find all the peaks and the peak flux in a spectrum

    Parameters
    ----------
    xarr: numpy.ndarry
        array of x values

    yarr: numpy.ndarry
        array of y values

    kernal_size: int
        size of dection kernal
 
    sections: int
        Number of sections to divide the image into
        for detection

    Returns
    -------
    xp: numpy.ndarry
        array of x values for peaks

    fp: numpy.ndarry
        array of flux values for peaks

    """
    if sections:
        nsec = len(xarr) / sections
        xp = None
        for i in range(sections):
            x1 = i * nsec
            x2 = x1 + nsec
            xa = detect_lines(xarr[x1:x2], farr[x1:x2], kernal_size=kernal_size,
                              center=True)
            if xp is None:
                xp = xa.copy()
            else:
                xp = np.concatenate((xp, xa))
    else:
        xp = detect_lines(xarr, farr, kernal_size=sigma, center=True)

    # create the list of the fluxes for each line
    xc = xp.astype(int)
    xf = farr[xc]
    return xp, xf


def find_backstats(f_arr, sigma, niter):
    """Iteratively calculate the statistics of an array"""
    ave = f_arr.mean()
    std = f_arr.std()
    for i in range(niter):
        mask = (abs(f_arr - ave) < sigma * std)
        ave = f_arr[mask].mean()
        std = f_arr[mask].std()
    return ave, std


def find_peaks(f_arr, sigma, niter, bsigma=None):
    """Go through an ordered array and find any element which is a peak"""
    # set up the variables
    if bsigma is None:
        bsigma = sigma

    # determine the background statistics
    back_ave, back_std = find_backstats(f_arr, sigma, niter)

    # calculate the differences between the pixels
    dfh = f_arr[1:-1] - f_arr[:-2]
    dfl = f_arr[1:-1] - f_arr[2:]

    # find the objects
    mask = (dfh > 0) * (dfl > 0) * \
        (abs(f_arr[1:-1] - back_ave) > back_std * sigma)
    t = np.where(mask)[0]
    return t + 1


def detect_lines(x, y, kernal_size=3, centroid_kernal=default_kernal, 
                 center=False):
    """Detect lines goes through a 1-D spectra and detect peaks

    Parameters
    ----------
    x: ~numpy.ndarray
        Array describing the x-position 

    y: ~numpy.ndarray
        Array describing the counts in each x-position

    kernal_size: int
        Size for the detection kernal

    centroid_kernal: 
        Kernal to be used for centroiding

    center: boolean
        If True, centroid for detected peaks will be calculated

    Returns
    -------
    xp: ~numpy.ndarray
        Array of x-positions of peaks in the spectrum
    """
    # find all peaks
    xp = signal.find_peaks_cwt(y, np.array([kernal_size]))
    xp = np.array(xp)

    # set the output values
    if center:
        xdiff = int(0.5 * len(centroid_kernal) + 1)
        x_arr = np.arange(len(x))
        xp = xp * 1.0
        for i in range(len(xp)):
            xp[i] = mcentroid(x, y, kern=centroid_kernal, xdiff=xdiff, xc=x[xp[i]])

    return xp


def flatspectrum(xarr, yarr, mode='mean', thresh=3, iter=5, order=3):
    """Remove the continuum from a spectrum either by masking it or fitting
       and subtracting it.

       xarr= input x-vales (pixels or wavelength)
       yarr= flux or counts for the spectrum
       mode=None--no subtraction
       mean--subtract off the mean
       poly--subtact off a fit
       mask--return a spectra with continuum set to zero
    """
    if mode == 'mean':
        # subtract off the mean value
        sarr = yarr - clipstats(yarr, thresh, iter)[0]
    elif mode == 'poly':
        # calculate the statistics and mask all of the mask with values above
        # these
        it = interfit(xarr, yarr, function='poly', order=order)
        it.interfit()
        sarr = yarr - it(xarr)
    elif mode == 'mask':
        # mask the values
        mean, std = clipstats(yarr, thresh, iter)
        mask = (yarr < mean + thresh * std)
        sarr = yarr.copy()
        sarr[mask] = 0
    else:
        sarr = yarr.copy()
    return sarr


def findwavelengthsolution(xarr, farr, sl, sf, ws, mdiff=20, wdiff=20, sigma=5,
                           niter=5):
    """Calculates the wavelength solution given a spectra and a set of lines.
       Hopefully an accurate first guess (ws) is provided and relative fluxes
       are provided as well, but if not, then the program is still designed
       to attempt to handle it.

       returns ws
    """
    # match up the features
    # xp, wp=findfeatures(xarr, farr, sl, sf, ws, mdiff=mdiff, wdiff=wdiff,
    #                    sigma=sigma, niter=niter)
    xp, wp = crosslinematch(xarr, farr, sl, sf, ws, mdiff=mdiff, wdiff=wdiff,
                            sigma=sigma, niter=niter)

    # find the solution to the best fit
    mask = (wp > 0)
    if mask.sum() >= ws.order:
        nws = WavelengthSolution.WavelengthSolution(
            xp[mask], wp[mask], model=ws.model)
        nws.fit()
    else:
        nws = None
    # for i in range(len(xp)): print xp[i], wp[i], wp[i]-nws.value(xp[i])
    # print nws.sigma(xp,wp)
    return nws


def findfeatures(xarr, farr, sl, sf, ws, mdiff=20, wdiff=20, sigma=5, niter=5,
                 sections=3):
    """Given a spectra, detect lines in the spectra, and find lines in
       the line list that correspond to those lines
    """

    # detect lines in the input spectrum and identify the peaks and peak values
    xp, xf = find_points(xarr, farr, kernal_size=sigma, sections=sections)

    # return no solution if no peaks were found
    if len(xp) == 0:
        return None

    # find the best match to the lines
    wp = findmatch(xarr, farr, xp, xf, sl, sf, ws, xlimit=mdiff, wlimit=wdiff)

    try:
        for i in range(len(xp)):
            if wp[i] > -1:
                pass
    except Exception as e:
        message = 'Unable to match line lists because %s' % e
        raise SpecError(message)
    return xp, wp


def findmatch(xarr, farr, xp, xf, sl, sf, ws, xlimit=10, wlimit=2):
    """Find the best match between the observed arc lines and the spectral
       line list.  If available, use the line fluxes and the wavelength
       solution.  Returns a an array that is a wavelength for each peak
       wavelength

       returns wp
    """
    wp = xp * 0.0 - 1
    px = xp * 0.0

    # calculate it using only xp and sl
    if sf is None and not ws:
        print 'Currently not available'

    # calculate it without any wavelength solution
    elif not ws:
        pass

    # calculate it without any flux information
    elif sf is None and ws:
        for i in xf.argsort()[::-1]:
            cx = mcentroid(xarr, farr, xc=xp[i], xdiff=4)
            if abs(cx - xp[i]) < xlimit:
                w = wavematch(ws(cx), wp, sl)
                wp[i] = w

    # calculate it using all of the information
    else:
        dcoef = ws.coef * 0.0
        dcoef[0] = 10
        dcoef[1] = dcoef[1] * 0.2
        ndstep = 20
        # this matches up the spectra but only varies the first
        # two coefficients by a small amount
        nws = spectramatch(
            xarr, farr, sl, sf, ws, dcoef, ndstep=ndstep, res=2, dres=0.1)
        for i in range(len(xf)):  # xf.argsort()[::-1]:
            cx = mcentroid(xarr, farr, xc=xp[i], xdiff=4)
            if abs(cx - xp[i]) < xlimit:
                w = wavematch(nws(cx), wp, sl, wlimit=wlimit)
                wp[i] = w
                px[i] = matchprob(cx, w, xf[i], xp, xf, sl, nws, dw=0.8)
            # print cx, nws.value(cx), wp[i], px[i], xp[i], xf[i]
    return wp


def matchprob(x, w, f, xp, xf, sl, ws, dw=5):
    """Calculate the probability that the line is the correct match.
       If it is matched up correctly and the solution is correct, then the
       other lines should be found in the right place.   The probabilibty will
       decrease by a factor of 0.1 for each line not found in the right place.
    """
    if w == -1:
        return 0
    p = 1.0
    # first assume that the zero point of the solution is set by the value
    try:
        nws = copy.deepcopy(ws)
    except:
        nws = WavelengthSolution.WavelengthSolution(
            ws.x, ws.wavelength, model=ws.model)
        nws.fit()
    nws.coef[0] = nws.coef[0] - (nws(x) - w)

    # Now loop through and see how well other objects end up fitting
    # if they are not present or there is problems, reduce the probability
    for i in xf.argsort()[::-1]:
        dist = abs(sl - nws(xp[i]))
        if dist.min() > dw:
            p = p * (1 - 0.1)
        else:
            p = p * (1 - 0.1 * dist.min() / dw * xf[i] / xf.max())
        # print x, w, xp[i], nws.value(xp[i]),sl[dist.argmin()],xf[i],
        # dist.min(),p

    return p


def spectramatch(xarr, farr, sw, sf, ws, dcoef, ndstep, res=2, dres=0.1,
                 inttype='interp'):
    """Using all the information which is available, cross correlate the
       observed spectra and the wavelength spectra to find the best
       coefficients and match the data
    """
    # create an artificial spectrum of the lines
    lmax = farr.max()

    swarr, sfarr = makeartificial(sw, sf, lmax, res, dres)

    nws = findxcor(xarr, farr, swarr, sfarr, ws, dcoef=dcoef, ndstep=ndstep,
                   inttype=inttype)
    return nws


def mod_coef(coef, dcoef, index, ndstep):
    """For a given index, return a list of modulations in that coefficient
    """
    dlist = []

    # if we have reached the last coefficient,
    if index >= len(coef):
        return dlist

    # if the coefficient doesn't need any modulation,
    # then move on to the next coefficient
    if dcoef[index] == 0:
        if index < len(coef) - 1:
            dlist.extend((mod_coef(coef, dcoef, index + 1, ndstep)))
        else:
            dlist.append(coef)
        return dlist

    # if the index does need variation, then proceed in one of two ways:
    # if it isn't the last coefficient, iterate over the values and then
    #   step down and do all the other coefficients
    # if it is the last coefficient, then iterate over the values and
    #   create the lowest level coefficient
    if index < len(coef) - 1:
        for x in np.arange(-dcoef[index], dcoef[index],
                           2 * dcoef[index] / float(ndstep)):
            ncoef = coef.copy()
            ncoef[index] = coef[index] + x
            dlist.extend(mod_coef(ncoef, dcoef, index + 1, ndstep))
    else:
        for x in np.arange(-dcoef[index],
                           dcoef[index], 2 * dcoef[index] / float(ndstep)):
            ncoef = coef.copy()
            ncoef[index] = coef[index] + x
            dlist.append(ncoef)
    return dlist


def makeartificial(sw, sf, fmax, res, dw, pad=10, nkern=200, wrange=None):
    """For a given line list with fluxes, create an artifical spectrum"""
    if wrange is None:
        wrange = [sw.min() - pad, sw.max() + pad]
    spec = Spectrum.Spectrum(
        sw, sf, wrange=wrange, dw=dw, stype='line', sigma=res)
    spec.flux = spec.flux * fmax / spec.flux.max()

    return spec.wavelength, spec.flux


def ncor(x, y):
    """Calculate the normalized correlation of two arrays"""
    d = np.correlate(x, x) * np.correlate(y, y)
    if d <= 0:
        return 0
    return np.correlate(x, y) / d ** 0.5


def wavematch(w, wp, sl, wlimit=10):
    """Compare a wavelength to an observed list and see if it matches up.  Skip
       if the lines is already in the wp list

    """

    # first remove anything already in the self.wp from the sl list
    lines = []
    for x in sl:
        if x not in wp:
            lines.append(x)
    if not lines:
        return -1
    lines = np.array(lines)

    # find the best match
    dist = abs(lines - w)
    if dist.min() < wlimit:
        i = dist.argmin()
    else:
        return -1

    # return the values
    return lines[i]


def findfit(xp, wp, ws=None, **kwargs):
    """Find the fit using just the matched points of xp and wp"""
    if ws is None:
        ws = WavelengthSolution.WavelengthSolution(xp, wp, **kwargs)
    else:
        ws.x = xp
        ws.wavelength = wp
    if len(xp) < ws.order:
        msg = 'Not enough points to determine an accurate fit'
        raise SpecError(msg)
    ws.fit()
    return ws


def findzeropoint(xarr, farr, swarr, sfarr, ws, dc=10, ndstep=20,
                  inttype='interp'):
    """Uses cross-correlation to find the best fitting zeropoint

    Parameters
    ----------
    xarr: numpy.ndarry
        array of x values

    farr: numpy.ndarry
        array of flux values

    swarr: numpy.ndarry
        array of wavelengths for known lines

    sfarr: numpy.ndarry
        array of flux values for known lines

    ws: ~WavelengthSolution.WavelengthSolution
        wavelength solution transforming between x and wavelength

    dc: float
        initial guess for range of zeropoint

    ndsteps: int
        number of steps to search over

    Returns
    -------
    ws: ~WavelengthSolution.WavelengthSolution
        wavelength solution with an updated zeropoint term

    """

    # if an initial solution, then cut the template lines to just be the
    # length of the spectrum
    if ws is None:
        return ws

    # set up the the dc coefficient
    dcoef = ws.coef * 0.0
    dcoef[0] = dc

    ws = findxcor(xarr, farr, swarr, sfarr, ws, dcoef=dcoef,
                  ndstep=ndstep, inttype=inttype)
    return ws


def xcorfun(p, xarr, farr, swarr, sfarr, interptype, ws):
    ws.coef=px
    # set the wavelegnth coverage
    warr = ws(xarr)
    # resample the artificial spectrum at the same wavelengths as the observed
    # spectrum
    asfarr = interpolate(
        warr, swarr, sfarr, type=interptype, left=0.0, right=0.0)
    return abs(1.0 / ncor(farr, asfarr))


def fitxcor(xarr, farr, swarr, sfarr, ws, interptype='interp', debug=False):
    """Maximize the normalized cross correlation coefficient for the full
        wavelength solution
    """
    try:
        nws = copy.deepcopy(ws)
    except:
        nws = WavelengthSolution.WavelengthSolution(
            ws.x, ws.wavelength, ws.model)
        nws.coef=ws.coef

    res = minimize(xcorfun, nws.coef, method='Nelder-Mead',
                   args=(xarr, farr, swarr, sfarr, interptype, nws))
    bcoef = res['x']
    nws.coef=bcoef
    return nws


def findxcor(xarr, farr, swarr, sfarr, ws, dcoef=None, ndstep=20, best=False,
             inttype='interp', debug=False):
    """Find the solution using crosscorrelation of the wavelength solution.
       An initial guess needs to be supplied along with the variation in
       each coefficient and the number of steps to calculate the correlation.
       The input wavelength and flux for the known spectral features should
       be in the format where they have already
       been convolved with the response function of the spectrograph

       xarr--Pixel coordinates of the image

       farr--Flux values for each pixel

       swarr--Input wavelengths of known spectral features

       sfarr--fluxes of known spectral features

       ws--current wavelength solution

       dcoef--Variation over each coefficient for correlation

       ndstep--number of steps to sample over

       best--if True, return the best value
             if False, return an interpolated value

       inttype--type of interpolation

    """

    # cross-correlate the spectral lines and the observed fluxes in order to
    # refine the solution
    try:
        nws = copy.deepcopy(ws)
    except:
        nws = WavelengthSolution.WavelengthSolution(
            ws.x, ws.wavelength, ws.model)

    # create the range of coefficents
    if dcoef is None:
        dcoef = ws.coef * 0.0 + 1.0

    dlist = mod_coef(ws.coef, dcoef, 0, ndstep)
    # loop through them and deteremine the best cofficient
    cc_arr = np.zeros(len(dlist), dtype=float)

    for i in range(len(dlist)):
        # set the coeficient
        nws.coef=dlist[i]
     
        # set the wavelegnth coverage
        warr = nws(xarr)

        # resample the artificial spectrum at the same wavelengths as the
        # observed spectrum
        asfarr = interpolate(
            warr, swarr, sfarr, type=inttype, left=0.0, right=0.0)

        # calculate the correlation value
        cc_arr[i] = ncor(farr, asfarr)
        #if debug: print cc_arr[i], " ".join(["%f" % k for k in dlist[i]])

    # now set the best coefficients
    i = cc_arr.argmax()
    bcoef = dlist[i]
    nws.coef=bcoef
    if best:
        return nws

    # interpoloate over the values to determine the best value
    darr = np.array(dlist)
    for j in range(len(nws.coef)):
        if dcoef[j] != 0.0:
            i = cc_arr.argsort()[::-1]
            tk = np.polyfit(darr[:, j][i[0:5]], cc_arr[i[0:5]], 2)

            if tk[0] == 0:
                bval = 0
            else:
                bval = -0.5 * tk[1] / tk[0]

            # make sure that the best value is close
            if abs(bval - bcoef[j]) < 2 * dcoef[j] / ndstep:
                bcoef[j] = bval

            # coef=np.polyfit(dlist[:][j], cc_arr, 2)
            # nws.coef[j]=-0.5*coef[1]/coef[0]

    nws.coef=bcoef

    return nws


def useRSSModel(xarr, rss, model,  gamma=0.0):
    """Returns the wavelength solution using the RSS model for the spectrograph


    """
    d = rss.detector.xbin * rss.detector.pix_size * \
        (xarr - rss.detector.get_xpixcenter())
    alpha = rss.alpha()
    beta = -rss.beta()
    dbeta = -np.degrees(np.arctan(d / rss.camera.focallength))
    y = 1e7 * rss.calc_wavelength(alpha, beta + dbeta, gamma=gamma)

    # for these models, calculate the wavelength solution
    ws = findfit(xarr, y, model=model)
    try:
        ws = findfit(xarr, y, model=model)
    except Exception as e:
        raise SpecError(e)
    return ws


def readlinelist(linelist):
    """Read in the line lists.  Determine what type of file it is.  The default
       is an ascii file with line and relative intensity.  The other types are
       just line, or a wavelenght calibrated fits file

       return lines, fluxes, and status
    """
    slines = []
    sfluxes = []
    status = 0

    # Check to see if it is a fits file
    # if not, then read in the ascii file
    if linelist[-4:] == 'fits':
        try:
            slines, sfluxes = readfitslinelist(linelist)
        except Exception as e:
            message = 'Unable to read in the line list %s because %s' % (
                linelist, e)
            raise SpecError(message)
    else:
        try:
            slines, sfluxes = readasciilinelist(linelist)
        except Exception as e:
            message = 'Unable to read in the line list %s because %s' % (
                linelist, e)
            raise SpecError(message)

    # conver to numpy arrays
    try:
        slines = np.asarray(slines)
        sfluxes = np.asarray(sfluxes)
    except Exception as e:
        message = 'Unable to create numpy arrays because %s' % (e)
        raise SpecError(message)

    return slines, sfluxes


def readfitslinelist(linelist):
    """Read in the line lists from an fits file.  If it is a 2-D array
       it will assume that it is an image and select the central wavlength

       return lines, fluxes, and status
    """
    slines = []
    sfluxes = []

    # open the image
    shdu = fits.open(linelist)
    nhdu = len(shdu)
    # determine if it is a one or two-d image
    # if ndhu=0 then assume that it is in the zeroth image
    # otherwise assume the data is in the first extension
    # assumes the x-axis is the wavelength axis
    if nhdu == 1:
        ctype1 = shdu[0].header['CTYPE1']
        crval1 = shdu[0].header['CRVAL1']
        cdelt1 = shdu[0].header['CDELT1']
        if shdu[0].data.ndim == 1:
            data = shdu[0].data
            wave = crval1 + cdelt1 * np.arange(len(shdu[0].data))

    # detect lines in the input spectrum and identify the peaks and peak values
    slines, sfluxes = find_points(wave, data, kernal_size=3)
    """
    figure(figsize=(8,8), dpi=72)
    axes([0.1, 0.1, 0.8, 0.8])
    plot(wave, data, ls='-')
    plot(slines, sfluxes, ls='', marker='o')
    xlim(4220,4900)
    show()
    """

    return slines, sfluxes


def readasciilinelist(linelist):
    """Read in the line lists from an ascii file.  It can either be a
        file with one or two columns.  Only read in lines that are not
        commented out.

       return lines, fluxes, and status
    """
    slines = []
    sfluxes = []

    # read in the file
    f = open(linelist)
    lines = f.readlines()
    f.close()

    # for each line,
    for l in lines:
        l = l.strip()
        if l and not l.startswith('#'):
            l = l.split()
            slines.append(float(l[0]))
            try:
                sfluxes.append(float(l[1]))
            except IndexError:
                sfluxes.append(-1)
    return slines, sfluxes


def getslitsize(slitname, config_file=''):
    """Return the slit size for a given slit name"""
    if slitname.strip() == 'PL0060N001':
        return 0.6
    if slitname.strip() == 'PL0100N001':
        return 1.0
    if slitname.strip() == 'PL0120P001':
        return 1.2
    if slitname.strip() == 'PL0125N001':
        return 1.25
    if slitname.strip() == 'PL0150N001':
        return 1.5
    if slitname.strip() == 'PL0200N001':
        return 2.0
    if slitname.strip() == 'PL0300N001':
        return 3.0
    if slitname.strip() == 'PL0400N001':
        return 4.0

    try:
        return int(slitname.strip())
    except:
        pass
    msg = 'Assuming a slit size of 1.0'
    return 1.0


def makesection(section):
    """Convert a section that is a list of coordinates into
       a list of indices
    """
    s = []
    if section is None:
        return s
    try:
        for i in section.split(':'):
            s.append(int(i))
    except Exception as e:
        msg = 'Not able to convet section to list because %s' % e
        raise SpecError(msg)
    return s


def vac2air(w):
    """following the definition used by SDSS based on Morton (1991, ApJS, 77, 119)
       AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)

       returns wavelength
    """
    return w / (1.0 + 2.735182E-4 + 131.4182 / w ** 2 + 2.76249E8 / w ** 4)

def readspectrum(specfile, stype='continuum', error=True, cols=None,
                 ftype=None):
    """Given a specfile, read in the spectra and return a spectrum object

       specfile--file containing the input spectra
       error--include an error column in the creation of the spectrum object
       cols--columns or column names for the wavelength, flux, and/or flux
             error
       ftype--type of file (ascii or fits)

    """

    # set the ftype for a fits file
    if ftype is None:
        if specfile[-5] == '.fits':
            ftype = 'fits'
        else:
            ftype = 'ascii'

    if ftype == 'ascii':
        if error:
            if cols is None:
                cols = (0, 1, 2)
            warr, farr, farr_err = np.loadtxt(
                specfile, usecols=cols, unpack=True)
            spectra = Spectrum.Spectrum(warr, farr, farr_err, stype=stype)
        else:
            if cols is None:
                cols = (0, 1)
            warr, farr = np.loadtxt(specfile, usecols=cols, unpack=True)
            spectra = Spectrum.Spectrum(warr, farr, stype=stype)
    elif ftype == 'fits':
        message = 'Support for FITS files not provided yet'
        raise SpecError(message)
    else:
        message = 'Support for %s files is not provided'
        raise SpecError(message)

    return spectra


def writespectrum(spectra, outfile, error=False, ftype=None):
    """Given a spectrum, write it out to a file"""
    fout = saltio.openascii(outfile, 'w')
    for i in range(spectra.nwave):
        fout.write('%8.6f ' % spectra.wavelength[i])
        fout.write('%8.6e ' % spectra.flux[i])
        if error:
            fout.write('%8.6e ' % spectra.var[i])
        fout.write('\n')
    fout.close()


def crosslinematch(xarr, farr, sl, sf, ws, mdiff=20, wdiff=20, res=2, dres=0.1,
                   sigma=5, niter=5, dc=20, sections=3):
    """Cross line match takes a line list and matches it with the observed
       spectra.

       The following steps are employed in order to achive the match:

    """
    # setup initial wavelength array
    warr = ws(xarr)
    # detect lines in the input spectrum and identify the peaks and peak values
    xp, xf = find_points(xarr, farr, kernal_size=sigma, sections=sections)

    # create an artificial lines for comparison
    lmax = farr.max()
    swarr, sfarr = makeartificial(sl, sf, lmax, res, dres)

    # now loop through the lines
    # exclude those lines that are outside of the source
    # then use the wdiff region to do a cross correlation around
    # a source and then proceed to calculate what the match is
    si = sf.argsort()
    dcoef = ws.coef * 0.0
    dcoef[0] = dc
    xp_list = []
    wp_list = []
    for i in si[::-1]:
        if sl[i] < warr.max() and sl[i] > warr.min():
            mask = abs(warr - sl[i]) < wdiff
            smask = abs(swarr - sl[i]) < wdiff
            nws = findxcor(xarr[mask], farr[mask], swarr[smask], sfarr[smask],
                           ws, dcoef=dcoef, ndstep=20, best=False,
                           inttype='interp', debug=False)
            # now find the best matching point
            # require it to be very close using the nws values
            # require  also that the fluxes match somehow or are close
            # ie if it is the third brightest thing in that region, then
            # it should be the third brightest thing
            # also require a good fit between observed and artificial
            nwarr = nws(xarr)
            nwp = nws(xp)
            d = abs(nwp - sl[i])
            j = d.argmin()
            if d.min() < res:
                if lineorder(xp, xf, sl, sf, sl[i], xp[j], wdiff, nws) and \
                   abs(ws(xp[j]) - sl[i]) < mdiff:
                    xp_list.append(xp[j])
                    wp_list.append(sl[i])
    return np.array(xp_list), np.array(wp_list)


def lineorder(xp, xf, sl, sf, sw, xb, wdiff, nws):
    """Determines the rank order of sw inside the set of lines and
       then determines if the xp line is the same rank order.
       Returns True if it is
    """

    # first cut the two line list down to the same size
    mask = abs(nws(xp) - sw) < wdiff
    smask = abs(sl - sw) < wdiff

    # identify the order of the spectral lines
    i = sf[smask].argsort()
    i_ord = i[sl[smask][i] == sw]
    if len(i_ord) > 1:
        return False

    # identify the order of the observed lines
    j = xf[mask].argsort()
    j_ord = j[xp[mask][j] == xb]
    if len(j_ord) > 1:
        return False
    return i_ord == j_ord


def smooth_spectra(xarr, farr, sigma=3, nkern=20):
    """Given a xarr and flux, smooth the spectrum"""
    xkern = np.arange(nkern)
    kern = np.exp(-(xkern - 0.5 * nkern) ** 2 / (sigma) ** 2)

    return gaussian_filter1d(farr, sigma)


def boxcar_smooth(spec, smoothwidth):
    # get the average wavelength separation for the observed spectrum
    # This will work best if the spectrum has equal linear wavelength spacings
    wavespace = np.diff(spec.wavelength).mean()
    # kw
    kw = int(smoothwidth / wavespace)
    # make sure the kernel width is odd
    if kw % 2 == 0:
        kw += 1
    kernel = np.ones(kw)
    # Conserve flux
    kernel /= kernel.sum()
    smoothed = spec.flux.copy()
    smoothed[(kw / 2):-(kw / 2)] = np.convolve(spec.flux, kernel, mode='valid')
    return smoothed
