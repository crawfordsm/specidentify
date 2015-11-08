# Licensed under a 3-clause BSD style license - see LICENSE.rst
# This module implements the base CCDData class.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy.testing import assert_array_equal
from astropy.tests.helper import pytest
from astropy.units.quantity import Quantity
import astropy.units as u
from astropy import modeling as mod

from ..spectools import *

# tests for mccentroid
def test_mccentroid():
    x = np.arange(-10, 10, 0.01) 
    g_init = mod.models.Gaussian1D(5, 0.52, 2.5)
    y = g_init(x)
    xc = mcentroid(x,y)
    assert np.isclose(0.52, xc)
