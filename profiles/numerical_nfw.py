from __future__ import annotations
from typing import Union
import numpy as np
from functools import partial
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline
from scipy.integrate import quad_vec, quad


def nfw_numerical(R: Union[float, ArrayLike], rho_s: float, r_s: float) -> Union[ArrayLike, float]:
    x = R / r_s
    nfw = rho_s / (x * (1 + x)**2)  # NFW profile
    return nfw


def delta_sigma_numerical(R: Union[float, ArrayLike], rho_s: float,
                          r_s: float, hrange: tuple[float] = (-40, 40)) -> Union[ArrayLike, float]:

    nfw_partial = partial(nfw_numerical, rho_s=rho_s, r_s=r_s)
    R_dense = np.logspace(np.log10(R[0])-2,  np.log10(R[-1]), 20*len(R))

    def nfw_integral_dense(h):
        return nfw_partial(np.sqrt(R_dense**2+h**2))

    sigma_r_dense = quad_vec(nfw_integral_dense, *hrange)[0]
    r_sigma_r_dense = R_dense * sigma_r_dense

    R_Sigma_R = CubicSpline(R_dense, r_sigma_r_dense)

    sigma_r = R_Sigma_R(R) / R  # Sigma(R)

    integral = np.fromiter(
        (quad(R_Sigma_R, 0, r_i)[0] for r_i in R),
        count=-1,
        dtype=float,
    )

    return (2 * integral / R**2) - sigma_r
