import xsimlab as xs
import xarray as xr
import numpy as np
import logging

from ...seir.base import BaseSEIR
from .base import BaseSetupEpi


@xs.process
class SetupDefaultNu(BaseSetupEpi):
    """
    """
    nu = xs.foreign(BaseSEIR, 'nu', intent='out')

    def initialize(self):
        self.nu = self.get_nu()

    def get_nu(self):
        data = np.array([0.02878229, 0.09120554, 0.02241002, 0.07886779, 0.17651128])
        dims = ['age_group']
        coords = [(k, self.counts_coords[k]) for k in dims]
        data = self.trim_data_to_coords(data, coords)
        return xr.DataArray(data=data, dims=dims, coords=coords)


@xs.process
class SetupStaticNu(SetupDefaultNu):
    """Calculate nu after sampling once from this triangular distibution,
    at the beginning of the simulation.
    """
    hosp_f_ratio = xs.variable(dims=('age_group'),
                               static=True, intent='in')
    gamma = xs.foreign(BaseSEIR, 'gamma', intent='in')
    mu = xs.foreign(BaseSEIR, 'mu', intent='in')

    def get_nu(self) -> xr.DataArray:
        dims = ['age_group']
        gamma_h = float(self.gamma.loc[dict(compartment='Ih')])
        da = (self.hosp_f_ratio * gamma_h /
              (self.mu + (gamma_h - self.mu) * self.hosp_f_ratio))
        assert isinstance(da, xr.DataArray), type(da)
        assert all((dim in da.coords for dim in dims)), (da.coords, dims)
        return da


@xs.process
class SetupDynamicNu(SetupStaticNu):
    """Like SetupStaticNu, but recalculate at every step.
    """

    def run_step(self):
        self.nu = self.get_nu()
