import pytest
import logging
import numpy as np
import xarray as xr
import xsimlab as xs

from episimlab.setup.epi import (
    SetupDefaultPi,
    SetupStaticPi,
    SetupDynamicPi
)


@pytest.fixture
def symp_h_ratio_w_risk(counts_coords):
    data = np.array([[0.00040205, 0.00402054],
            [0.00030913, 0.00309131],
            [0.01903482, 0.19034819],
            [0.04114127, 0.41141273],
            [0.04878947, 0.48789469]]).T
    dims = ['risk_group', 'age_group']
    return xr.DataArray(
        data=data,
        dims=dims,
        coords={dim: counts_coords[dim] for dim in dims}
    )


@pytest.fixture()
def expected():
    return 1 / np.array(
        [[1686.58480529, 2193.46500471,   35.93928726,   16.80105527, 14.21781764],
         [ 168.94830933,  219.63632927,    3.88375753,    1.96993433, 1.71161056]])


class TestSetupPi:

    def test_can_setup_static(self, counts_coords, symp_h_ratio_w_risk, gamma,
                              eta, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'symp_h_ratio_w_risk': symp_h_ratio_w_risk,
            'gamma': gamma,
            'eta': eta,
        })

        proc = SetupStaticPi(**inputs)
        proc.initialize()
        result = proc.pi
        assert isinstance(result, xr.DataArray)

        logging.debug(f"1 / result: {1 / result}")
        logging.debug(f"1 / expected: {1 / expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    @pytest.mark.parametrize('n_steps', (1, 10))
    def test_can_setup_dynamic(self, symp_h_ratio_w_risk, gamma, eta,
                               n_steps, counts_coords, expected):
        inputs = counts_coords.copy()
        inputs.update({
            'symp_h_ratio_w_risk': symp_h_ratio_w_risk,
            'gamma': gamma,
            'eta': eta,
        })

        proc = SetupDynamicPi(**inputs)
        proc.initialize()
        for _ in range(n_steps):
            proc.run_step()
        result = proc.pi
        assert isinstance(result, xr.DataArray)

        logging.debug(f"1 / result: {1 / result}")
        logging.debug(f"1 / expected: {1 / expected}")
        np.testing.assert_allclose(result, expected, rtol=1e-4)
