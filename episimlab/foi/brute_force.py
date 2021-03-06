import xsimlab as xs
import xarray as xr
import logging
from itertools import product
from numbers import Number

from ..apply_counts_delta import ApplyCountsDelta
from .base import BaseFOI


@xs.process
class BruteForceFOI(BaseFOI):
    """A readable, brute force algorithm for calculating force of infection (FOI).
    """

    phi_t = xs.global_ref('phi_t')
    counts = xs.foreign(ApplyCountsDelta, 'counts', intent='in')

    def run_step(self):
        """
        """
        # Instantiate as array of zeros
        self.foi = xr.DataArray(
            data=0.,
            dims=self.FOI_DIMS,
            coords={dim: getattr(self, dim) for dim in self.FOI_DIMS}
        )

        # Iterate over every pair of unique vertex-age-risk combinations
        for v1, a1, r1, v2, a2, r2 in product(*[self.vertex, self.age_group,
                                                self.risk_group] * 2):
            total_pop = self.counts.loc[dict(
                vertex=v2, age_group=a2, risk_group=r2
            )].sum(dim=['compartment'])

            # Get the value of phi
            phi = self.phi_t.loc[dict(
                vertex1=v1,
                vertex2=v2,
                age_group1=a1,
                age_group2=a2,
                risk_group1=r1,
                risk_group2=r2,
            )].values

            # Get S compt
            counts_S = self.counts.loc[{
                'vertex': v1,
                'age_group': a1,
                'risk_group': r1,
                'compartment': 'S'
            }].values

            # Get value of beta
            beta = self.beta
            assert isinstance(beta, Number)

            # Get infectious compartments
            compt_I = ['Ia', 'Iy', 'Pa', 'Py']
            counts_I = self.counts.loc[{
                'vertex': v2,
                'age_group': a2,
                'risk_group': r2,
                'compartment': compt_I
            }]

            # Get value of omega for these infectious compartments
            omega_I = self.omega.loc[{'age_group': a2, 'compartment': compt_I}]

            # Calculate force of infection
            common_term = beta * phi * counts_S / total_pop
            _sum = (common_term * omega_I * counts_I).sum(dim='compartment').values
            self.foi.loc[dict(vertex=v1, age_group=a1, risk_group=r1)] += _sum


def get_foi_numpy(compt_ia, compt_iy, compt_pa, compt_py, compt_s, phi_, beta, kappa,
                  omega_a, omega_y, omega_pa, omega_py, age_pop, n_age, n_risk):
    """From SEIR-city v1.4 d0902c0af796a6ac3a15ec833ae24dcfa81d9f2b"""

    # reshape phi
    phi_tile = np.tile(phi_, (n_risk, n_risk))

    def _si_contact(s_compt, i_compt, omega):

        # transpositions are needed b/c the state arrays have rows = age, cols = risk,
        # but the parameters have rows = risk, cols = age
        it = i_compt.transpose()
        st = s_compt.transpose()
        si_outer = np.outer(st, it)
        si_omega = si_outer * omega.ravel()
        si_contact = si_omega * phi_tile
        si_beta = si_contact * beta.ravel()
        age_tile = np.tile(age_pop, (1, n_risk))
        if np.isinf(age_tile).any():
            raise ValueError('Tiled age structure array contains np.inf values')
        si_pop = si_beta/age_tile
        si_a1_r = si_pop.sum(axis=1).reshape(n_risk, n_age)

        return si_a1_r

    foi = np.zeros((n_risk, n_age))

    # from compt_ia
    foi += _si_contact(compt_s, compt_ia, omega_a)

    # from compt_iy
    foi += _si_contact(compt_s, compt_iy, omega_y)

    # from compt_pa
    foi += _si_contact(compt_s, compt_pa, omega_pa)

    # from compt_py
    foi += _si_contact(compt_s, compt_py, omega_py)

    return foi.transpose()
