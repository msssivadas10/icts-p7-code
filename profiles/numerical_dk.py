from __future__ import annotations
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline
from scipy.integrate import quad_vec, quad
from functools import partial


# NUMERICAL $\Delta\Sigma(R)$ for Diemer Kravstov Profile

def dk14_numerical(rho_s: float, rho_o: float,
                   r_s: float, r_t: float,  # , r_out:float
                   alpha: float, beta: float, gamma: float,
                   s_e, R: ArrayLike) -> Union[ArrayLike, float]:

    r_out = 1.5  # h^-1 Mpc
    # rho inner
    rho_in = rho_s * np.exp(-2/alpha * ((R/r_s)**alpha - 1))
    # rho outer
    rho_outer = rho_o * (R / r_out)**(-s_e)
    # f_transition
    f_trans = (1 + (R / r_t) ** beta) ** (-gamma / beta)
    # Sigma(R)
    rho_R = (rho_in * f_trans) + rho_outer

    return rho_R

# --------------------------------------------------------


def sigma_dk14(rho_s: float, rho_o: float,
               r_s: float, r_t: float,  # , r_out:float
               alpha: float, beta: float, gamma: float,
               s_e: float, R: ArrayLike, zmax: float = 40):

    dk14_partial = partial(
        dk14_numerical,
        rho_s=rho_s, rho_o=rho_o,  # R |ArrayLike,
        r_s=r_s, r_t=r_t,
        alpha=alpha, beta=beta, gamma=gamma,
        s_e=s_e,
    )

    def dk14_integrand(z):
        return dk14_partial(R=np.sqrt(R**2 + z**2))

    sigma_r_dk14 = 2 * quad_vec(dk14_integrand, 0, zmax)[0]

    return sigma_r_dk14

# --------------------------------------------------------


def delta_sigma_dk14_numerical(R: ArrayLike, R_dense: ArrayLike,
                               sigma_r_dense: ArrayLike) -> Union[ArrayLike, float]:

    if R_dense.shape != sigma_r_dense.shape:
        raise ValueError(
            f"operands could not be broadcast together with shapes {R_dense.shape} and {sigma_r_dense.shape}")

    r_sigma_r_dense = R_dense * sigma_r_dense
    R_Sigma_R = CubicSpline(R_dense, r_sigma_r_dense)
    sigma_r = R_Sigma_R(R) / R

    integral = np.fromiter(
        (quad(R_Sigma_R, 0, r_i)[0] for r_i in R),
        count=-1,
        dtype=float,
    )

    return (2 * integral / R**2) - sigma_r

# --------------------------------------------------------


def deltasigma_dk14_direct(rho_s: float, rho_o: float,
                           r_s: float, r_t: float,  # , r_out:float
                           alpha: float, beta: float, gamma: float,
                           s_e: float, R: ArrayLike):

    R_dense = np.logspace(np.log10(R[0])-2,  np.log10(R[-1]), 20*len(R))
    sigma_rdense_dk14 = sigma_dk14(
        rho_s, rho_o, r_s, r_t, alpha, beta, gamma, s_e, R_dense
    )
    dsigmadk14 = delta_sigma_dk14_numerical(R, R_dense, sigma_rdense_dk14)

    return dsigmadk14

# --------------------------------------------------------
