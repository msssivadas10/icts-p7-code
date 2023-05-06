from __future__ import annotations
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from functools import partial


# For the density function for all R
def analytic_surface_density(A: float, x: Union[ArrayLike, float],
                             conditions: tuple[ArrayLike]) -> ArrayLike:

    surface_mass_density = np.zeros(len(x))

    x_lt_1, x_eq_1, x_gt_1 = conditions

    x1 = x[x_lt_1]
    x2 = x[x_eq_1]
    x3 = x[x_gt_1]

    # x < 1
    surface_mass_density[x_lt_1] = (A / (x1**2 - 1)) * \
        (
        1 - (2 / np.sqrt(1-x1**2)) *
        np.arctanh(np.sqrt((1-x1)/(1+x1)))
    )
    # x == 1
    surface_mass_density[x_eq_1] = A/3

    # x > 1
    surface_mass_density[x_gt_1] = (A / (x3**2 - 1)) * \
        (
        1 - (2 / np.sqrt(x3**2-1)) *
        np.arctan(np.sqrt((x3-1)/(1+x3)))
    )

    return surface_mass_density

# ---------------------------------------------------------------------------------
# For the average density within R


def analytic_avg_surface_density(A: float, x: Union[ArrayLike, float],
                                 conditions: tuple[ArrayLike]) -> ArrayLike:

    surface_mass_density = np.zeros(len(x))

    x_lt_1, x_eq_1, x_gt_1 = conditions

    x1 = x[x_lt_1]
    x2 = x[x_eq_1]
    x3 = x[x_gt_1]

    # x < 1
    surface_mass_density[x_lt_1] = (2*A/(x1**2)) * \
        (
        np.log(x1/2) +
        (2/np.sqrt(1 - x1**2)) * np.arctanh(np.sqrt((1-x1)/(1+x1)))
    )

    # x == 1
    surface_mass_density[x_eq_1] = 2 * (1 - np.log(2))

    # x > 1
    surface_mass_density[x_gt_1] = (2*A/(x3**2)) * \
        (
        np.log(x3/2) +
        (2/np.sqrt(x3**2 - 1)) * np.arctan(np.sqrt((x3-1)/(1+x3)))
    )

    return surface_mass_density

# ---------------------------------------------------------------------------------

# $\Sigma_{NFW} (R)$: Analytical Surface density


def analytic_density_nfw(R: Union[ArrayLike, float], r_s: float, rho_s: float) -> Union[ArrayLike, float]:

    R_was_float = False
    if isinstance(R, float):
        R = np.array([R])
        R_was_float = True  # for allowing float usage

    x: ArrayLike = R/r_s
    A: float = 2*r_s*rho_s

    # conditions
    x_lt_1 = np.where(x < 1)
    x_eq_1 = np.where(np.abs(x-1) < 1e-3)
    x_gt_1 = np.where(x > 1)

    surface_mass_density = analytic_surface_density(
        A, x, conditions=(x_lt_1, x_eq_1, x_gt_1))

    if R_was_float:
        return float(surface_mass_density[0])
    else:
        return surface_mass_density


# ---------------------------------------------------------------------------------

# $\bar{\Sigma}_{NFW} (<R)$: Avg. Analytical Surface density inside $R$

def avg_analytic_density_nfw_lt_R(R: Union[ArrayLike, float], r_s: float, rho_s: float) -> Union[ArrayLike, float]:

    R_was_float = False
    if isinstance(R, float):
        R = np.array([R])
        R_was_float = True  # for allowing float usage

    x: ArrayLike = R/r_s
    A: float = 2*r_s*rho_s

    # conditions
    x_lt_1 = np.where(x < 1)
    x_eq_1 = np.where(np.abs(x-1) < 1e-3)
    x_gt_1 = np.where(x > 1)

    surface_mass_density_lt_R = analytic_avg_surface_density(
        A, x, conditions=(x_lt_1, x_eq_1, x_gt_1))

    if R_was_float:
        return float(surface_mass_density_lt_R[0])
    else:
        return surface_mass_density_lt_R

# ---------------------------------------------------------------------------------
# $\Delta\Sigma(R)$


def delta_surface_density(R: ArrayLike, r_s: float, rho_s: float) -> ArrayLike:
    x: ArrayLike = R/r_s
    A: float = 2*r_s*rho_s

    # conditions
    x_lt_1 = np.where(x < 1)
    x_eq_1 = np.where(np.abs(x-1) < 1e-3)
    x_gt_1 = np.where(x > 1)

    surface_mass_density = analytic_surface_density(
        A, x, conditions=(x_lt_1, x_eq_1, x_gt_1))
    surface_mass_density_lt_R = analytic_avg_surface_density(
        A, x, conditions=(x_lt_1, x_eq_1, x_gt_1))

    return surface_mass_density_lt_R - surface_mass_density


def _test_plots():

    r = np.logspace(-3, 2, 100)
    r_s, rho_s = 0.7, 10

    delta_sigma = delta_surface_density(r, r_s, rho_s)
    avg_sigma_lt_R = avg_analytic_density_nfw_lt_R(r, r_s, rho_s)
    sigma_R = analytic_density_nfw(r, r_s, rho_s)

    plt.title(rf"$\rho_s = {rho_s}; r_s = {r_s} $")

    plt.loglog()
    plt.plot(r, avg_sigma_lt_R, ls="--", lw=0.7, label=r"$\bar{\Sigma}(<R)$")
    plt.plot(r, sigma_R, ls="--", lw=0.7, label=r"$\Sigma(R)$")
    plt.plot(r, avg_sigma_lt_R - sigma_R, ls="--", lw=0.7,
             label=r"$\bar{\Sigma}(<R) - \Sigma(R)$", zorder=5)

    plt.plot(r, delta_sigma, ls="-", lw=2,
             label=r"$\Delta\Sigma(R)$")  # main delta_sigma

    plt.xlim(7e-4, r.max())
    # plt.ylabel()
    plt.xlabel("$R$")
    # plt.axvline(r_s, label="$R = r_s$", color="r", lw=0.5)
    plt.grid(which='major', alpha=0.4,)
    plt.grid(which='minor', alpha=0.4, ls='--')
    plt.legend(frameon=False)
