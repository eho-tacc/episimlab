import pytest
import logging
import xarray as xr
import numpy as np
from episimlab.setup.phi import InitPhi


class TestInitPhi:

    def test_can_run_step(self, counts_coords, phi_grp_mapping):
        inputs = {
            'phi_grp1': np.arange(10),
            'phi_grp2': np.arange(10),
            'phi_grp_mapping': phi_grp_mapping,
        }
        proc = InitPhi(**inputs)
        proc.initialize()
        proc.run_step(step=0)
        result = proc.phi
        assert isinstance(result, xr.DataArray)
        # logging.debug(f"result: {result}")

