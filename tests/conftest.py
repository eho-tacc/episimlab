import os
import yaml
import pytest
import numpy as np
import pandas as pd
import logging
import xarray as xr


@pytest.fixture()
def counts_dims():
    return ['vertex', 'age_group', 'risk_group', 'compartment']


@pytest.fixture(params=[
    3,
    # 5
])
def counts_coords(request):
    return {
        'vertex': list(range(request.param)),
        'age_group': ['0-4', '5-17', '18-49', '50-64', '65+'],
        'risk_group': ['low', 'high'],
        'compartment': ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih',
                        'R', 'D', 'E2P', 'E2Py', 'P2I', 'Pa2Ia',
                        'Py2Iy', 'Iy2Ih', 'H2D']
    }


@pytest.fixture(params=[
    0.75,
])
def foi(request, counts_coords):
    dims = ('vertex', 'age_group', 'risk_group')
    return xr.DataArray(
        data=request.param,
        dims=dims,
        coords={dim: counts_coords[dim] for dim in dims}
    )


@pytest.fixture(params=[
    'realistic',
    # 'mid_sim',
    # 'ones'
])
def counts_basic(counts_dims, counts_coords, request):
    if request.param == 'realistic':
        da = xr.DataArray(
            data=0.,
            dims=counts_dims,
            coords=counts_coords,
        )
        # Houston
        da.loc[dict(vertex=0, compartment='S')] = 2.326e6 / 10.
        # Austin
        da.loc[dict(vertex=1, compartment='S')] = 1e6 / 10.
        # Beaumont
        da.loc[dict(vertex=2, compartment='S')] = 1.18e5 / 10.
        # Start with 50 infected asymp in Austin
        da.loc[dict(vertex=1, compartment='Ia')] = 50.
    elif request.param == 'ones':
        da = xr.DataArray(
            data=1.,
            dims=counts_dims,
            coords=counts_coords,
        )
    elif request.param == 'mid_sim':
        raise NotImplementedError()
    else:
        raise ValueError()
    return da


@pytest.fixture()
def adj_grp_mapping(counts_dims, counts_coords) -> xr.DataArray:
    """A MultiIndex mapping for an adj array"""
    dims = ('vertex', 'age_group', 'risk_group', 'compartment')
    coords = {
        k: v for k, v in counts_coords.items()
        if k in dims
    }
    shape = np.array([len(coords[d]) for d in dims])
    data = np.arange(np.prod(shape)).reshape(shape)
    return xr.DataArray(data=data, dims=dims, coords=coords)


@pytest.fixture()
def phi_grp_mapping(counts_dims, counts_coords) -> xr.DataArray:
    """A MultiIndex mapping for a phi array"""
    dims = ['vertex', 'age_group', 'risk_group']
    coords = {
        k: v for k, v in counts_coords.items()
        if k in dims
    }
    shape = np.array([len(coords[d]) for d in dims])
    data = np.arange(np.prod(shape)).reshape(shape)
    return xr.DataArray(data=data, dims=dims, coords=coords)


@pytest.fixture()
def adj_t(counts_coords) -> xr.DataArray:
    dims = ('vertex1', 'vertex2', 'age_group', 'risk_group', 'compartment')
    # we assume that all the dimensions in counts are also dimensions
    # on the adjacency matrix
    coords = {
        'vertex1': counts_coords['vertex'],
        'vertex2': counts_coords['vertex']
    }
    coords.update({
        dim: counts_coords[dim] for dim in ('age_group', 'risk_group', 'compartment')
    })
    return xr.DataArray(
        data=0.1,
        dims=dims,
        coords=coords
    )


@pytest.fixture()
def phi_t(counts_coords):
    data = 0.2
    dims = (
        'vertex1',
        'vertex2',
        'age_group1',
        'age_group2',
        'risk_group1',
        'risk_group2',
    )
    coords = {
        'vertex1': counts_coords['vertex'],
        'vertex2': counts_coords['vertex'],
        'age_group1': counts_coords['age_group'],
        'age_group2': counts_coords['age_group'],
        'risk_group1': counts_coords['risk_group'],
        'risk_group2': counts_coords['risk_group'],
    }
    return xr.DataArray(data=data, dims=dims, coords=coords)


@pytest.fixture()
def epis(rho, gamma, sigma, pi, eta, nu, mu, tau):
    return dict(
        rho=rho, gamma=gamma, sigma=sigma, pi=pi,
        eta=eta, nu=nu, mu=mu, tau=tau
    )


@pytest.fixture()
def rho(counts_coords):
    data = 0.43478261
    dims = ['compartment']
    coords = {k: counts_coords[k] for k in dims}
    return xr.DataArray(data=data, dims=dims, coords=coords)


@pytest.fixture()
def gamma(counts_coords):
    data = 0.
    dims = ['compartment']
    coords = {k: counts_coords[k] for k in dims}
    da = xr.DataArray(data=data, dims=dims, coords=coords)
    da.loc[dict(compartment=['Ia', 'Iy', 'Ih'])] = [0.25, 0.25, 0.09118541]
    return da


@pytest.fixture()
def sigma():
    return 0.34482759


@pytest.fixture()
def pi(counts_coords):
    # pi = np.array(
        # [[5.92915812e-04, 4.55900959e-04, 2.78247788e-02, 5.95202276e-02, 7.03344654e-02],
         # [5.91898663e-03, 4.55299354e-03, 2.57483139e-01, 5.07631836e-01, 5.84245731e-01]])
    data = np.array([
        [5.92915812e-04, 4.55900959e-04, 2.78247788e-02, 5.95202276e-02, 7.03344654e-02],
        [5.91898663e-03, 4.55299354e-03, 2.57483139e-01, 5.07631836e-01, 5.84245731e-01]
    ])
    dims = ('risk_group', 'age_group')
    coords = {k: counts_coords[k] for k in dims}
    return xr.DataArray(data=data, dims=dims, coords=coords)


@pytest.fixture()
def eta():
    return 0.169492


@pytest.fixture()
def nu(counts_coords):
    dims = ['age_group']
    return xr.DataArray(
        [0.02878229, 0.09120554, 0.02241002, 0.07886779, 0.17651128],
        dims=dims,
        coords={k: counts_coords[k] for k in dims}
    )


@pytest.fixture()
def mu():
    return 0.12820513


@pytest.fixture()
def beta():
    return 0.035


@pytest.fixture()
def tau():
    return 0.57


@pytest.fixture(params=[
    # deterministic at all steps
    -1,
    # stochastic at all steps
    0,
    5,
    20
])
def sto_toggle(request):
    return request.param


@pytest.fixture()
def omega(counts_coords):
    # omega_a = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667])
    # omega_y = np.array([1.        , 1.        , 1.        , 1.        , 1.        ])
    # omega_h = np.array([0.        , 0.        , 0.        , 0.        , 0.        ])
    # omega_pa = np.array([0.91117513, 0.91117513, 0.92460653, 0.95798887, 0.98451149])
    # omega_py = np.array([1.36676269, 1.36676269, 1.3869098 , 1.43698331, 1.47676724])
    data = np.array([[0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667],
                     [1.        , 1.        , 1.        , 1.        , 1.        ],
                     [0.91117513, 0.91117513, 0.92460653, 0.95798887, 0.98451149],
                     [1.36676269, 1.36676269, 1.3869098 , 1.43698331, 1.47676724]])
    dims = ['age_group', 'compartment']
    coords = {k: counts_coords[k] for k in dims}

    da = xr.DataArray(data=0., dims=dims, coords=coords)
    da.loc[dict(compartment=['Ia', 'Iy', 'Pa', 'Py'])] = data.T
    assert isinstance(da, xr.DataArray), type(da)
    return da


@pytest.fixture()
def seed_entropy():
    return 12345


@pytest.fixture()
def seed_state(seed_entropy):
    return seed_entropy


@pytest.fixture(params=[
    True,
    False
])
def stochastic(request):
    return request.param


@pytest.fixture(params=[
    # int_per_day == 1
    '24H',
    # int_per_day == 2
    '12H'
])
def step_clock(request):
    return {
        'step': pd.date_range(
            start='1/1/2018', end='1/15/2018', freq=request.param
        )
    }


@pytest.fixture(params=[
    # int_per_day == 1
    24,
    # int_per_day == 2
    12
])
def step_delta(request):
    try:
        return np.timedelta64(request.param, 'h')
    except ValueError:
        logging.debug(f"type(request.param): {type(request.param)}")
        raise


@pytest.fixture
def config_fp_static():
    return './tests/config/example_v1.yaml'


@pytest.fixture
def config_fp(tmpdir):
    """Fixture factory that writes a dictionary to a YAML file and
    provides the file path.
    """
    fp = tmpdir.mkdir("config").join("config.yaml")

    def _config_fp(config_dict):
        assert not os.path.isfile(fp)
        with open(fp, 'w') as f:
            yaml.dump(config_dict, f)
        assert os.path.isfile(fp)
        return str(fp)

    return _config_fp


@pytest.fixture()
def symp_h_ratio_w_risk(counts_coords):
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [[0.0002791, 0.0002146, 0.0132154, 0.0285634, 0.0338733],
            [0.002791, 0.002146, 0.132154, 0.285634, 0.338733]]


@pytest.fixture()
def symp_h_ratio(counts_coords):
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [0.00070175, 0.00070175, 0.04735258, 0.16329827, 0.25541833]


@pytest.fixture()
def prop_trans_in_p():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 0.44


@pytest.fixture()
def hosp_f_ratio(counts_coords) -> xr.DataArray:
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [0.04, 0.12365475, 0.03122403, 0.10744644, 0.23157691]


@pytest.fixture()
def asymp_relative_infect():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 0.666666666


@pytest.fixture()
def tri_h2d():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [5.2, 8.1, 10.1]


@pytest.fixture()
def tri_h2r():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [9.4, 10.7, 12.8]


@pytest.fixture()
def tri_exposed_para():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [1.9, 2.9, 3.9]


@pytest.fixture()
def tri_pa2ia():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 2.3


@pytest.fixture()
def tri_py2iy():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 2.3


@pytest.fixture()
def asymp_rate():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 0.43


@pytest.fixture()
def t_onset_to_h():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return 5.9


@pytest.fixture()
def tri_h2r():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [9.4, 10.7, 12.8]


@pytest.fixture()
def tri_y2r_para():
    """example_meyers_demo.yaml from SEIRcity v2"""
    return [3.0, 4.0, 5.0]


@pytest.fixture
def config_dict(seed_entropy, sto_toggle, counts_basic, tri_h2r, symp_h_ratio,
                tri_exposed_para, prop_trans_in_p, hosp_f_ratio,
                symp_h_ratio_w_risk, tri_pa2ia, asymp_relative_infect,
                asymp_rate, tri_h2d, t_onset_to_h, tri_y2r_para, tri_py2iy,
                counts_coords):
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
        'tri_py2iy': tri_py2iy,
        'coords': counts_coords
    }
