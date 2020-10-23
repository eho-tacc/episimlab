import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..seir.base import BaseSEIR

@xs.process
class BaseFOI:
    """
    """
    beta = xs.foreign(BaseSEIR, 'beta', intent='in')
    omega = xs.foreign(BaseSEIR, 'omega', intent='in')

    # TODO: needs a vertex dimension
    foi = xs.variable(intent='out')

    def initialize(self):
        self.foi = 0.
