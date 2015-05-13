
import numpy as np

from astropy import modeling as md
from astropy import stats



class WavelengthSolution:

    """A class describing the solution between x-position and wavelength.


    Parameters
    ----------
    x: ~numpy.ndarray
        Array of the x-positions  

    wavelength: ~numpy.ndarray
        Array of the wavelength at each x-position
 
    model: ~astropy.modeling.models
        A 1D model describing the transformation between x and wavelength

    Raises
    ------

    Notes
    -----
 
    Examples
    --------
  
    """
    


    def __init__(self, x, wavelength, model):
        self.x = x
        self.wavelength = wavelength
        self.model = model

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        #TODO: Add checker that it is an astropy model
        self._model = value

    @property
    def coef(self):
        return self.model.parameters

    @coef.setter
    def coef(self, value):
        self.model.parameters = coef

    @property
    def order(self):
        return self.model.degree


    def __call__(self, x):
        return self.model(x)

    def fit(self, niter=5):
        """Determine the fit of the model to the data points with rejection

        For each iteraction, a weight is calculated based on the distance a source
        is from the relationship and outliers are rejected.

        Parameters
        ----------
        niter: int
            Number of iteractions for the fit

        """
        fitter = md.fitting.LinearLSQFitter()
        weights = np.ones_like(self.x)
        for i in range(niter):
            self.model = fitter(self.model, self.x, self.wavelength, weights=weights)

            #caculate the weights based on the median absolute deviation
            r = (self.wavelength - self.model(self.x))
            s = stats.mad_std(r)
            biweight = lambda x: ((1.0 - x ** 2) ** 2.0) ** 0.5
            if s!=0:
                weights = 1.0/biweight(r / s)
            else:
                weights = np.ones(len(self.x))


    def sigma(self, x, w):
        """Return the RMS of the fit 
       
        Parameters
        ----------
        x: ~numpy.ndarray
            Array of the x-positions  
        w: ~numpy.ndarray
            Array of the wavelength-positions  

        Returns
        -------
        
        """
        # if there aren't many data points return the RMS
        if len(x) < 4:
            sigma = (((w - self(x)) ** 2).mean()) ** 0.5
        # Otherwise get the average distance between the 16th and
        # 84th percentiles
        # of the residuals divided by 2
        # This should be less sensitive to outliers
        else:
            # Sort the residuals
            rsdls = np.sort(w - self(x))
            # Get the correct indices and take their difference
            sigma = (rsdls[int(0.84 * len(rsdls))] -
                   rsdls[int(0.16 * len(rsdls))]) / 2.0
        return sigma

    def chisq(self, x, y, err):
        """Return the chi^2 of the fit"""
        return (((y - self.value(x)) / err) ** 2).sum()
