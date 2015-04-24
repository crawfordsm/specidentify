# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is software for the reduction of data from
optical spectrographs

"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

class SpecError(Exception):

    """Errors involving this package should cause this exception to be raised.
    """
    pass



# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .spectools import *
    from .guitools import * 
    from iterfit import *
    from . import models

