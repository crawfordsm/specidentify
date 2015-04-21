
import numpy as np

from iterfit import iterfit
from PySpectrograph.WavelengthSolution.ModelSolution import ModelSolution


class WavelengthSolution:

    """Wavelength Solution is a task describing the functional form for
    transforming pixel position to wavelength.  The inputs for this task
    are the given pixel position and the corresponding wavelength.  The user
    selects an input functional form and order for that form.  The task then
    calculates the coefficients for that form.  Possible options for the
    wavelength solution include polynomial, legendre, spline.
    """

    func_options = ['poly', 'polynomial', 'spline', 'legendre',
                    'chebyshev', 'model']

    def __init__(self, x, w, function='poly', order=3, niter=5, thresh=3,
                 sgraph=None, cfit='both', xlen=3162, yval=0):
        self.sgraph = sgraph
        self.function = function
        self.order = order
        self.niter = niter
        self.thresh = thresh
        self.cfit = cfit
        self.xlen = xlen
        self.yval = yval

        self.set_array(x, w)
        self.set_func()

    def set_array(self, x, w):
        self.x_arr = x
        self.w_arr = w

    def set_thresh(self, thresh):
        self.thresh = thresh

    def set_niter(self, niter):
        self.niter = niter

    def set_func(self):
        if self.function in [
                'poly', 'polynomial', 'spline', 'legendre', 'chebyshev']:
            self.func = iterfit(self.x_arr, self.w_arr,
                                 function=self.function,
                                 order=self.order, niter=self.niter,
                                 thresh=self.thresh)
        if self.function == 'model':
            self.func = ModelSolution(self.x_arr, self.w_arr,
                                      sgraph=self.sgraph, xlen=self.xlen,
                                      yval=self.yval, order=self.order)

    def fit(self):
        if self.function in [
                'poly', 'polynomial', 'spline', 'legendre', 'chebyshev']:
            self.func.iterfit()
            self.coef = self.func.coef
        if self.function in ['model']:
            self.func.fit(cfit=self.cfit)
            self.coef = np.array([c() for c in self.func.coef])
        # self.set_coef(coef)

    def set_coef(self, coef):
        if self.function in [
                'poly', 'polynomial', 'spline', 'legendre', 'chebyshev']:
            self.func.coef = coef
            self.coef = self.func.coef
        if self.function in ['model']:
            for i in range(len(self.func.coef)):
                self.func.coef[i].set(coef[i])
            self.coef = np.array([c() for c in self.func.coef])

    def value(self, x):
        return self.func(x)

    def invvalue(self, w):
        """Given a wavelength, return the pixel position

        """
        return w

    def sigma(self, x, y):
        """Return the RMS of the fit """
        # if there aren't many data points return the RMS
        if len(x) < 4:
            sig = (((y - self.value(x)) ** 2).mean()) ** 0.5
        # Otherwise get the average distance between the 16th and
        # 84th percentiles
        # of the residuals divided by 2
        # This should be less sensitive to outliers
        else:
            # Sort the residuals
            rsdls = np.sort(y - self.value(x))
            # Get the correct indices and take their difference
            sig = (rsdls[int(0.84 * len(rsdls))] -
                   rsdls[int(0.16 * len(rsdls))]) / 2.0
        return sig

    def chisq(self, x, y, err):
        """Return the chi^2 of the fit"""
        return (((y - self.value(x)) / err) ** 2).sum()
