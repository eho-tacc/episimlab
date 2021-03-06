import pytest
import logging
import xarray as xr
from episimlab.seir.bf_cython import BruteForceCythonSEIR
from episimlab.seir.brute_force import BruteForceSEIR


class TestBruteForceCythonSEIR:

    def test_can_run_step(self, foi, seed_entropy, stochastic, counts_basic,
                          step_delta, epis):
        inputs = {
            'counts': counts_basic,
            'foi': foi,
            'seed_state': seed_entropy,
            'stochastic': stochastic,
        }
        inputs.update(epis)
        proc = BruteForceCythonSEIR(**inputs)
        proc.run_step(step_delta)

        proc.finalize_step()
        result = proc.counts_delta_seir

        # logging.debug(f"result: {result}")
        assert isinstance(result, xr.DataArray)

        # TODO
        # assert not result.isnull().any()

    # Python and Cython use different RNG
    # so it's useless to compare them even with the same seed
    @pytest.mark.parametrize('stochastic', [False])
    def test_same_as_python(self, foi, stochastic, seed_entropy, counts_basic,
                            step_delta, epis):
        inputs = {
            'counts': counts_basic,
            'foi': foi,
            'seed_state': seed_entropy,
            'stochastic': stochastic,
        }
        inputs.update(epis)

        # run in cython
        cy_proc = BruteForceCythonSEIR(**inputs)
        # cy_proc.initialize()
        cy_proc.run_step(step_delta)
        cy_proc.finalize_step()
        cy_result = cy_proc.counts_delta_seir
        assert isinstance(cy_result, xr.DataArray)

        # run in python
        py_proc = BruteForceSEIR(**inputs)
        # cy_proc.initialize()
        py_proc.run_step(step_delta)
        # py_proc.finalize_step()
        py_result = py_proc.counts_delta_seir
        assert isinstance(py_result, xr.DataArray)

        # assert are the same
        xr.testing.assert_allclose(py_result, cy_result)
