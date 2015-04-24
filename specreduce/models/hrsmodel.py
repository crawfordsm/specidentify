import numpy as np
from PySpectrograph.Spectrograph import Spectrograph, Grating, Optics, CCD, \
    Detector, Slit
from PySpectrograph import SpectrographError


class HRSModel (Spectrograph):

    """HRSModel is a class that describes the High Resolution Specotrgraph  on SALT
    """

    def __init__(self, grating_name='hrs', camera_name='hrdet', slit=2.0,
                 order=83, gamma=None, xbin=1, ybin=1, xpos=0.00, ypos=0.00):

        # set up the parts of the grating
        self.grating_name = grating_name

        # set the telescope
        self.set_telescope('SALT')

        # set the collimator
        self.set_collimator('hrs')

        # set the camera
        self.set_camera(camera_name)

        # set the detector
        self.set_detector(
            camera_name,
            xbin=xbin,
            ybin=ybin,
            xpos=xpos,
            ypos=ypos)

        # set up the grating
        self.set_grating(self.grating_name, order=order)

        # set up the slit
        self.set_slit(slit)

        # set up the grating angle
        if gamma is not None:
            self.gamma = gamma

    def alpha(self, da=0.00):
        """Return the value of alpha for the spectrograph"""
        return self.grating.blaze + self.gamma

    def beta(self, db=0):
        """Return the value of beta for the spectrograph

           Beta_o=(1+fA)*(camang)-gratang+beta_o
        """
        return self.grating.blaze - self.gamma + db

    def get_wavelength(self, xarr, gamma=0.0):
        """For a given spectrograph configuration, return the wavelength coordinate
           associated with a pixel coordinate.

           xarr: 1-D Array of pixel coordinates
           gamma: Value of gamma for the row being analyzed

           returns an array of wavelengths in mm
        """
        d = self.detector.xbin * self.detector.pix_size * \
            (xarr - self.detector.get_xpixcenter())
        dbeta = np.degrees(np.arctan(d / self.camera.focallength))
        return self.calc_wavelength(
            self.alpha(), -self.beta() + dbeta, gamma=gamma)

    def set_telescope(self, name='SALT'):
        if name == 'SALT':
            self.telescope = Optics(name=name, focallength=46200.0)
        else:
            raise SpectrographError('%s is not a supported Telescope' % name)

    def set_collimator(self, name='hrs', focallength=2000.0):
        if name == 'hrs':
            self.collimator = Optics(name=name, focallength=focallength)
        else:
            msg = '{0} is not a supported collimator'.format(name)
            raise SpectrographError(msg)

    def set_camera(self, name='hrdet', focallength=None):
        if name == 'hrdet':
            self.camera = Optics(name=name, focallength=402.26)
            self.gamma = 2.43
        elif name == 'hbdet':
            self.camera = Optics(name=name, focallength=333.6)
            self.gamma = 2.00
        else:
            raise SpectrographError('%s is not a supported camera' % name)

    def set_detector(
            self, name='hrdet', geom=None, xbin=1, ybin=1, xpos=0, ypos=0):
        if name == 'hrdet':
            ccd = CCD(name='hrdet', xpix=4122, ypix=4112,
                      pix_size=0.015, xpos=0.00, ypos=0.00)
            self.detector = Detector(name=name, ccd=[ccd], xbin=xbin,
                                     ybin=ybin, xpos=xpos, ypos=ypos)
        elif name == 'hbdet':
            ccd = CCD(name='hrdet', xpix=2100, ypix=4112,
                      pix_size=0.015, xpos=0.00, ypos=0.00)
            self.detector = Detector(name=name, ccd=[ccd], xbin=xbin,
                                     ybin=ybin, xpos=xpos, ypos=ypos)
        else:
            raise SpectrographError('%s is not a supported detector' % name)

    def set_grating(self, name=None, order=83):
        if name == 'hrs':
            self.grating = Grating(name='hrs', spacing=41.59, blaze=76.0,
                                   order=order)
            self.set_order(order)
        elif name == 'red beam':
            self.grating = Grating(name='red beam', spacing=855, blaze=0,
                                   order=1)
            self.alpha_angle = 17.5
            self.set_order(1)
        elif name == 'blue beam':
            self.grating = Grating(
                name='blue beam',
                spacing=1850,
                blaze=0,
                order=1)
            self.alpha = 24.6
            self.set_order(1)
        else:
            raise SpectrographError('%s is not a supported grating' % name)

    def set_order(self, order):
        self.order = order
        self.grating.order = order

    def set_slit(self, slitang=2.2):
        self.slit = Slit(name='Fiber', phi=slitang)
        self.slit.width = self.slit.calc_width(self.telescope.focallength)
