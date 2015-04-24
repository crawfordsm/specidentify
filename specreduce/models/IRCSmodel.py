"""IRCSmodel is a class that describes The Infrared Camera and Spectrograph on
Subaru.   For more information about this camera:
http://www.naoj.org/Observing/Instruments/IRCS/index.html


"""

import numpy as np

from PySpectrograph.Spectrograph import Spectrograph, Grating, Optics, CCD, Detector, Slit


class IRCSModel (Spectrograph):

    """A class describing Infrared Camera and Spectrograph on
Subaru.   For more information about this camera:
http://www.naoj.org/Observing/Instruments/IRCS/index.html
    """

    def __init__(self, grating_name='None', alpha=0, beta=0, slit=1.0,
                 xbin=1, ybin=1, xpos=0.0, ypos=0.00, wavelength=None):

        # set up the parts of the grating
        self.grating_name = grating_name
        self.slitang = slit

        #TODO: Not sure if I understand the spectrograph 
        #layout and so how alpha/beta and grism angle work
        #so right now just leaving this as user defined
        self._alpha = alpha
        self._beta = beta

        # set the telescope
        self.set_telescope('Subaru')

        # set the collimator
        self.set_collimator('IRCS')

        # set the camera
        self.set_camera('IRCS')

        # set the detector
        self.set_detector('IRCS', xbin=xbin, ybin=ybin, xpos=xpos, ypos=ypos)

        # set up the grating
        self.set_grating(self.grating_name)

        # set up the slit
        self.set_slit(self.slitang)


    #TODO: replace with decorator
    def alpha(self):
        """Return the value of alpha for the spectrograph"""
        return self._alpha

    #TODO: replace with decorator/prperty
    def beta(self):
        """Return the value of beta for the spectrograph

        """
        return self._beta

    def get_wavelength(self, xarr, gamma=0.0):
        """For a given spectrograph configuration, return the wavelength coordinate
           associated with a pixel coordinate.

        Parameters
        ----------
        xarr: ~numpy.ndarray
            Array of pixel coordinates in the dispersion direction

        gamma: float
            The off-axis angle for the row being analyzed

        Returns
        -------
        warr: ~numpy.ndarray
            returns an array of wavelengths in mm
        """
        d = self.detector.xbin * self.detector.pix_size * (xarr - self.detector.get_xpixcenter())
        dbeta = -np.degrees(np.arctan(d / self.camera.focallength))
        return self.calc_wavelength(self.alpha(), -self.beta() + dbeta, gamma=gamma)

    def set_telescope(self, name='Subaru'):
        """Set the parameters of the telescope to be used

        Parameter
        ---------
        name: str
            Name of the telescope

        """
        if name == 'Subaru':
            self.telescope = Optics(name=name, focallength=15189.0)
        else:
            raise KeyError('%s is not a supported Telescope' % name)

    def set_collimator(self, name='IRCS', focallength=320.0):
        """Set the parameters of the collimator to be used

        Parameter
        ---------
        name: str
            Name of the instrument

        """
        if name == 'IRCS':
            self.collimator = Optics(name=name, focallength=focallength)
        else:
            raise KeyError('%s is not a supported collimator' % name)

    def set_camera(self, name='IRCS', focallength=315):
        """Set the parameters of the camera to be used

        Parameter
        ---------
        name: str
            Name of the instrument

        """
        if name == 'IRCS':
            self.camera = Optics(name=name, focallength=315)
        else:
            raise KeyError('%s is not a supported camera' % name)

    def set_detector(self, name='IRCS', geom=None, xbin=2, ybin=2, xpos=0, ypos=0):
        """Set the detector to be used

        Parameters
        ----------
        name: str
            Name of detector to be used

        TODO: Not sure if these are right and/or if they can have different values
  
        """
        if name == 'IRCS':
            ccd1 = CCD(name='CCD1', xpix=1024, ypix=1024, pix_size=0.027, xpos=0.00, ypos=0.00)
            self.detector = Detector(name=name, ccd=[ccd1], xbin=xbin, ybin=ybin,
                                     xpos=xpos, ypos=ypos, plate_scale=2.02)
        else:
            raise KeyError('%s is not a supported detector' % name)

    def set_grating(self, name=None): 
        """Set the grating to be used

        Parameters
        ----------
        name: str
            Name of grating to be used
  
        TODO:  There is only one grism currently in here
        """
 
        if name == 'G20':
            self.grating = Grating(name='G20', spacing=20, blaze=14.75, order=1)
        else:
            raise KeyError('%s is not a supported grating' % name)

    def set_slit(self, slitang=1.0):
        self.slit = Slit(name='LongSlit', phi=slitang)
        self.slit.width = self.slit.calc_width(self.telescope.focallength)
