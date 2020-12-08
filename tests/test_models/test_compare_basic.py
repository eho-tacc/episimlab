import pytest
import xarray as xr
import xsimlab as xs
import pandas as pd
import numpy as np
import logging
import episimlab
from episimlab.seir import (
    base as seir_base,
    brute_force as seir_bf,
    bf_cython as seir_bf_cython,
    bf_cython_w_foi as seir_bf_cython_w_foi
)
from episimlab.foi import (
    base as foi_base,
    brute_force as foi_bf,
    bf_cython as foi_bf_cython,
)
from episimlab.pytest_utils import plotter

VERBOSE = False


@pytest.fixture
def config_dict(seed_entropy, sto_toggle, counts_basic, tri_h2r, symp_h_ratio,
               tri_exposed_para, prop_trans_in_p, hosp_f_ratio,
               symp_h_ratio_w_risk, tri_pa2ia, asymp_relative_infect,
               asymp_rate, tri_h2d, t_onset_to_h, tri_y2r_para, tri_py2iy):
    return {
        'seed_entropy': seed_entropy,
        'sto_toggle': sto_toggle,
        'tri_h2r': tri_h2r,
        'symp_h_ratio': symp_h_ratio,
        'tri_exposed_para': tri_exposed_para,
        'prop_trans_in_p': prop_trans_in_p,
        'hosp_f_ratio': hosp_f_ratio,
        'symp_h_ratio_w_risk': symp_h_ratio_w_risk,
        'tri_pa2ia': tri_pa2ia,
        'asymp_relative_infect': asymp_relative_infect,
        'asymp_rate': asymp_rate,
        'tri_h2d': tri_h2d,
        't_onset_to_h': t_onset_to_h,
        'tri_y2r_para': tri_y2r_para,
        'tri_py2iy': tri_py2iy
    }


@pytest.fixture
def input_vars(config_fp, config_dict):
    return dict(read_config__config_fp=config_fp(config_dict))


class TestCompareBasicModels:

    # @plotter(flavor='mpl', plotter_kwargs=dict())
    @pytest.mark.parametrize('foi1, seir1', [
        # Python SEIR and FOI
        (foi_bf.BruteForceFOI, seir_bf.BruteForceSEIR),
        # Cython SEIR with Python FOI
        (foi_bf.BruteForceFOI, seir_bf_cython.BruteForceCythonSEIR),
        # Cython SEIR with Cython FOI
        (foi_bf_cython.BruteForceCythonFOI, seir_bf_cython.BruteForceCythonSEIR),
        # Python SEIR with Cython FOI
        (foi_bf_cython.BruteForceCythonFOI, seir_bf.BruteForceSEIR),
        # Cython SEIR with FOI
        (foi_base.BaseFOI, seir_bf_cython_w_foi.BruteForceCythonWFOI),
    ])
    @pytest.mark.parametrize('foi2, seir2', [
        # Python SEIR and FOI
        # (foi_bf.BruteForceFOI, seir_bf.BruteForceSEIR),
        # Cython SEIR with FOI
        (foi_base.BaseFOI, seir_bf_cython_w_foi.BruteForceCythonWFOI),
        # Cython SEIR with Cython FOI
        # (foi_bf_cython.BruteForceCythonFOI, seir_bf_cython.BruteForceCythonSEIR),
    ])
    def test_seir_foi_combos_deterministic(self, step_clock, foi1, config_fp,
                                           config_dict, seir1, foi2, seir2):
        """Check that, at model scope, different implementations of a basic
        SEIR model produce consistent results. For instance, a FOI implemented
        in Python should produce the same results as FOI implemented in Cython
        at all timepoints.

        NOTE: some of these tests are not expected to pass. For instance,
        Python and Cython implementations of stochastic SEIR dynamics use
        different random number generators (RNGs), and are expected not to
        produce the same output given the same inputs (including RNG seed).
        There is logic below to catch these expected failures implicitly, so
        that this test should always pass.
        """
        # generate input variables
        input_vars = dict(read_config__config_fp=config_fp(config_dict))

        # load default model
        model = episimlab.models.toy.slow_seir()

        # construct models
        model1 = model.update_processes(dict(
            seir=seir1,
            foi=foi1,
        ))
        model2 = model.update_processes(dict(
            seir=seir2,
            foi=foi2,
        ))

        # shared inputs based on the default model
        out_var_key = 'apply_counts_delta__counts'
        in_ds = xs.create_setup(
            model=model,
            clocks={k: step_clock[k] for k in step_clock},
            input_vars=input_vars,
            output_vars={out_var_key: 'step'}
        )

        # run both models
        result1 = in_ds.xsimlab.run(
            model=model1, decoding=dict(mask_and_scale=False))[out_var_key]
        result2 = in_ds.xsimlab.run(
            model=model2, decoding=dict(mask_and_scale=False))[out_var_key]

        # check typing and equality
        assert isinstance(result1, xr.DataArray)
        assert isinstance(result2, xr.DataArray)

        # Expect failure if the point of comparison SEIR engine has a Python
        # RNG, and if stochasticity is enabled for this simulation
        failure_expected = bool(
            seir1 is seir_bf.BruteForceSEIR and
            config_dict['sto_toggle'] >= 0
        )
        # implicitly account for results that are not expected to be the same
        # e.g. Python vs. Cython SEIR with different RNGs
        if failure_expected:
            logging.debug("Skipping pytest due to expected discrepancy " +
                          "between Python and Cython RNG")
        else:
            xr.testing.assert_allclose(result1, result2, rtol=1e-3)
