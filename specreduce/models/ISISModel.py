import numpy as np
from PySpectrograph.Spectrograph import Spectrograph, Grating, Optics, CCD, \
    Detector, Slit
from PySpectrograph import SpectrographError


class ISISModel (Spectrograph):

    """ISISModel is a class that describes the ISIS Specotrgraph

    """

    def __init__(self, grating_name='GR1200R', camera_name='Red arm', slit=1.0,
                 gratang=0.0, order=1, gamma=0, xbin=1, ybin=1, xpos=0.00, ypos=0.00):

        # set up the parts of the grating
        self.grating_name = grating_name

        # set the telescope
        self.set_telescope('WHT')

        # set the collimator
        self.set_collimator('ISIS')

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

        self.gratang = gratang
        self.gamma = gamma


    def alpha(self, da=0.00):
        """Return the value of alpha for the spectrograph"""
        return self.gratang + da

    def beta(self, db=0):
        """Return the value of beta for the spectrograph

           Beta_o=(1+fA)*(camang)-gratang+beta_o
        """
        return self.gratang + db

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
            self.alpha(), -self.beta() - dbeta, gamma=gamma)

    def set_telescope(self, name='WHT'):
        if name == 'WHT':
            self.telescope = Optics(name=name, focallength=10500.0)
        else:
            raise SpectrographError('%s is not a supported Telescope' % name)

    def set_collimator(self, name='ISIS', focallength=1650.0):
        if name == 'ISIS':
            self.collimator = Optics(name=name, focallength=focallength)
        else:
            msg = '{0} is not a supported collimator'.format(name)
            raise SpectrographError(msg)

    def set_camera(self, name='Red arm', focallength=None):
        if name == 'Red arm':
            self.camera = Optics(name=name, focallength=500.00)
            #self.gamma = 2.43
        elif name == 'Blue arm':
            self.camera = Optics(name=name, focallength=500.00)
            #self.gamma = 2.00
        else:
            raise SpectrographError('%s is not a supported camera' % name)

    def set_detector(
            self, name='Red arm', geom=None, xbin=1, ybin=1, xpos=0, ypos=0):
        if name == 'Red arm':
            ccd = CCD(name='REDPLUS', xpix=4190, ypix=966,
                      pix_size=0.015, xpos=0.00, ypos=0.00)
            self.detector = Detector(name=name, ccd=[ccd], xbin=xbin,
                                     ybin=ybin, xpos=xpos, ypos=ypos)
        elif name == 'Blue arm':
            ccd = CCD(name='BLUEPLUS', xpix=4190, ypix=966,
                      pix_size=0.015, xpos=0.00, ypos=0.00)
            self.detector = Detector(name=name, ccd=[ccd], xbin=xbin,
                                     ybin=ybin, xpos=xpos, ypos=ypos)
        else:
            raise SpectrographError('%s is not a supported detector' % name)

    def set_grating(self, name=None, order=1):
        if name == 'R1200R':
            self.grating = Grating(name='R1200R', spacing=1200, blaze=0.0,
                                   order=order)
            self.set_order(order)
        elif name == 'R1200B':
            self.grating = Grating(name='R1200B', spacing=1200, blaze=0.0,
                                   order=order)
            self.set_order(order)
        else:
            raise SpectrographError('%s is not a supported grating' % name)

    def set_order(self, order):
        self.order = order
        self.grating.order = order

    def set_slit(self, slitang=2.2):
        self.slit = Slit(name='slit', phi=slitang)
        self.slit.width = self.slit.calc_width(self.telescope.focallength)
