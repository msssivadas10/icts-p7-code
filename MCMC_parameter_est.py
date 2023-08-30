# Analytical/Numerical $\Delta\Sigma(R)$ model & MCMC
# Since SciServer is on Python 3.8 in "SciServer Essentials 2.0",
# need to import typing from __future__ for type hints like list[int], etc.,
# uncomment the next line before running it there
# Note that type hints may not work for Python < 3.7
# For Python > 3.9 importing from __future__ not needed
# from __future__ import annotations

import sys
from typing import Union

import astropy.constants as const
import astropy.cosmology as ac
import astropy.units as cu
import numpy as np
import pandas as pd
import yaml
from emcee import EnsembleSampler
from emcee.backends import HDFBackend
from numpy.typing import NDArray
from schwimmbad import MultiPool

from profiles.analytic_nfw import delta_surface_density
from profiles.numerical_dk import deltasigma_dk14_direct

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

cosmo = ac.FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)


def get_data(fname: str):
    """
    Takes the file with 'DeltaSigma', 'R_center', 'Jackknife_region'
    and 
    """
    data = pd.read_csv(
        fname,
        delim_whitespace=True,
        usecols=[4, 6, 12],  # may need to be edited for the new file
        names=['DeltaSigma', 'R_center', 'Jackknife_region'],
        skiprows=1,  # may need to be edited for the new file
    ).dropna()
    # .dropna() since there was a NaN row being read in
    # at the end of the file; skiprows since first row was text

    # Calculate the number of data points in each jackknife region
    n_pts_per_jn = len(data) // len(data["Jackknife_region"].unique())

    # Calculate the mean DeltaSigma for each R bin
    deltaSigma_mean = data.groupby('R_center')['DeltaSigma'].mean()

    # Split the data into Jackknife regions using Pandas
    n_jackknife = len(data["Jackknife_region"].unique())
    data_list_each_jn = [
        data[data['Jackknife_region'] == i] for i in range(n_jackknife)
    ]

    # Initialize the covariance matrix using NumPy
    cov = np.zeros((n_pts_per_jn, n_pts_per_jn))

    # Calculate the covariance matrix using NumPy's dot product function
    for data_i in data_list_each_jn:
        deltaSigma = data_i['DeltaSigma'].values
        cov += np.outer(deltaSigma - deltaSigma_mean,
                        deltaSigma - deltaSigma_mean)

    # Calculate the final covariance matrix and correlation matrix using NumPy
    cov = ((n_jackknife - 1) / n_jackknife) * cov
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    r_sigma_data = data.groupby(['R_center'])[
        ['R_center', 'DeltaSigma']].mean()

    return r_sigma_data, cov, corr
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------


def random_dk14_data():
    """
    """
    r_data = np.logspace(-2, 1, 20)
    r_s, rho_s = 0.7, 10
    rng = np.random.default_rng()
    noise = 0.2 * rng.lognormal(size=r_data.size)
    # data    = deltasigma_dk14_direct(*args_for_sigma_rdata) + noise

    return r_data, r_s, rho_s, noise  # data


def initial_args_for_dk14(r_data: NDArray):
    args_for_sigma = [  # to initialize
        np.exp(1.1),
        np.exp(0.349),
        np.exp(-0.32),
        np.exp(-0.082),
        np.exp(-0.95),
        np.exp(0.762),
        np.exp(0.66),
        1.601,
    ]

    return tuple(args_for_sigma + [r_data])

# -----------------------------------------------------------------
# Posterior distribution


def lnprob_dk14(
        var: Union[tuple, list, NDArray],
        r_data: NDArray,
        data: NDArray,
        inv_cov: NDArray
):
    """
    Compute the log posterior probability for the DK14 model.

    Parameters:
    -----------
    var : tuple
        Tuple containing the 8 model parameters
        (rho_s, rho_o, r_s, r_t, alpha, beta, gamma, s_e).
    r_data : array-like
        Array of radius values.
    data : array-like
        Array of Delta Sigma values.
    inv_cov : array-like
        Inverse covariance matrix for the data.

    Returns:
    --------
    lnpost : float
        The log posterior probability of the model.
    """

    rho_s, rho_o, r_s, r_t, alpha, beta, gamma, s_e = var

    C1 = (10 >= rho_s >= 0)
    C2 = (10 >= rho_o >= 0)
    C3 = (5 >= r_t > 0.1)
    C4 = (5 >= r_s > 0.1)
    C5 = (10 >= s_e >= 1)
    C6 = (alpha > 0)
    C7 = (beta > 0)
    C8 = (gamma > 0)

    if not (C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8):
        return -np.inf

    logPalpha = -(np.log(alpha) - np.log(0.19)) / (2 * (0.6**2))
    logPbeta = -(np.log(beta) - np.log(4.0)) / (2 * (0.2**2))
    logPgamma = -(np.log(gamma) - np.log(6.0)) / (2 * (0.2**2))
    lnprior = logPalpha + logPbeta + logPgamma

    dsd = deltasigma_dk14_direct(*var, r_data)

    lnlike = -((data - dsd).T @ inv_cov @ (data - dsd))
    lnpost = lnlike + lnprior

    if np.isnan(lnpost):
        return -np.inf

    return lnpost

# -----------------------------------------------------------------


def mu(c200):
    return np.log(1 + c200) - c200/(1+c200)

# -----------------------------------------------------------------


def xm_c2rs_rhos(m200, c200):

    omega_M = cosmo.Om0  # 0.3 # set cosmology and constants
    G = const.G.to(cu.km**2 * cu.Mpc / cu.Msun / cu.s**(2)).value
    # 4.2994e-9 # set cosmology and constants
    rho_bar = omega_M * (3e4 / (8 * np.pi * G))
    r200 = (m200 / (4/3 * np.pi * 200 * rho_bar))**(1/3)
    r_s = r200/c200
    rho_s = m200/(4*np.pi*r_s**3 * mu(c200))

    return r_s, rho_s
# -----------------------------------------------------------------

# Posterior distribution


def lnprob_nfwa(var, r_data, data, inv_cov):

    xm200, c200 = var
    m200 = 10**xm200
    r_s, rho_s = xm_c2rs_rhos(m200, c200)

    if ((c200 < 2) | (xm200 < 10)):
        return -np.inf
    else:
        lnprior = 0

    dsd = delta_surface_density(r_data, r_s, rho_s/1e12)
    lnlike = -((data - dsd).T @  inv_cov @ (data - dsd))
    lnpost = lnlike + lnprior

    if np.isnan(lnpost):
        return -np.inf

    return lnpost

#########################################################
#########################################################


def main(data_fname: str):

    r_sigma_data, cov, corr = get_data(data_fname=data_fname)
    R_bins = r_sigma_data['R_center'].to_numpy()
    data = r_sigma_data['DeltaSigma'].to_numpy()
    inv_cov = np.linalg.inv(cov)

    # ------------------------------------------------------
    args_for_sigma_rdata = initial_args_for_dk14(R_bins)

    lnprob = lnprob_dk14    # callable
    # initiate the random walkers
    # ------------------------------------------------------

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    nwalkers, ndim = 64, 8
    pos = args_for_sigma_rdata[:-1] + 1e-2 * np.random.randn(nwalkers, ndim)

    iterations = 50_000  # SET THE NUMBER OF ITERATIONS HERE

    fname = "sampler_dk14.h5"  # add NUMBER of ITERATIONS in the NAME if needed
    backend = HDFBackend(fname)
    # ----------------------------------------------------------
    # ----------------------------------------------------------

    # run the chain
    with MultiPool() as pool:
        if not pool.is_master():
            pool.wait()
            print("Not on the master process")
            sys.exit(0)

        sampler = EnsembleSampler(
            nwalkers, ndim,
            lnprob,
            args=(
                R_bins,
                data,
                inv_cov
            ),
            backend=backend,
            pool=pool,
        )

        sampler.run_mcmc(pos, iterations, progress=True)
################################################################


if __name__ == '__main__':

    config_fname = 'config.yml'
    with open(config_fname, 'r') as file:
        inputs = yaml.safe_load(file)

    main(data_fname=inputs["files"]["output"])
