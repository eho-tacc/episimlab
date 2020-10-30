import pytest
import logging
import xarray as xr
from episimlab.foi.brute_force import BruteForceFOI


class TestBruteForceFOI:

    def test_can_run_step(self, omega, counts_basic, phi_grp_mapping, phi_t):
        """
        """
        inputs = {
            'age_group': counts_basic.coords['age_group'],
            'risk_group': counts_basic.coords['risk_group'],
            'vertex': counts_basic.coords['vertex'],
            'beta': 0.035,
            'omega': omega,
            'counts': counts_basic,
            'phi_t': phi_t,
            'phi_grp_mapping': phi_grp_mapping
        }
        foi_getter = BruteForceFOI(**inputs)
        foi_getter.run_step()
        result = foi_getter.foi

        # logging.debug(f"phi_grp_mapping: {phi_grp_mapping}")
        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)

