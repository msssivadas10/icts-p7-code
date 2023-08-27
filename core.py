#!/usr/bin/python3
# @author: m. s. sūryan śivadās
# @file:
#
# code skelton: almost complete?
#

import sys
import time

import h5py  # to read hdf5 files

# from os import read
import numpy as np
import pandas as pd
import yaml  # to parse yaml config files
from astropy.cosmology import FlatLambdaCDM  # cosmology model
from scipy.interpolate import CubicSpline  # for interpolations

# for calculating delta-sigma
from src.calc_tngt_shear import (  # calculate_dsigma_increments,
    calculate_dsigma_increments_vector,
    get_lens_constants,
)
from src.kdtreecode import BallTree  # for nearest neighbours

# for loading the catalogs
from src.reading_data_shape_redshift_catalog import (
    reading_data_sources,
    reading_lens_params,
)

try:
    from mpi4py import MPI
except ImportError:
    print("MPI not found")

# define the function to run pipline.
# input: config filename


def read_config_file(config_fname):
    with open(config_fname, "r") as file:
        config = yaml.safe_load(file)
    return config


def run_pipeline(config_fname):
    try:
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    except ModuleNotFoundError:
        rank = 0
        size = 1

    #
    # reading the config file
    #
    sys.stderr.write(f"Rank({rank}): Reading the config file...\n")
    config = read_config_file(config_fname)
    #
    # creating a flat-lcdm model using the given cosmology parameters.
    #
    cosmo = FlatLambdaCDM(
        H0=config["cosmology"]["H0"], Om0=config["cosmology"]["OmegaMatter"]
    )  # defaults: Tcmb = 0K, Ob0 = None

    # redshift range
    z_min = config["z_min"]
    z_max = config["z_max"]

    #
    # calculate and interpolate the comoving distances for future use
    #
    z = np.linspace(
        max(z_min - 0.1, 0.0), min(z_max + 0.1, 10.0), config["z_bins_intrp"]
    )
    xz = cosmo.comoving_distance(z)

    # spline object for comoving distance calculation
    # can be accessed like a function
    comoving_distance = CubicSpline(z, xz)

    # parameters for binning neighbour distance
    r_min = config["r_min"]  # lower distance bound
    r_max = config["r_max"]  # upper distance bound
    r_bins = config["r_bins"]  # number of bins

    #
    # find the angular seperation for nn search
    #
    # comoving distance corresponding to minimum redshift
    cmdist = comoving_distance(z_min)
    theta_max = r_max / cmdist  # maximum angular seperation

    #
    # read files: lenses catalog -> ra, dec, redshift
    #
    sys.stderr.write(f"Rank({rank}): Reading the lens catalog...\n")
    lens_fname = config["files"]["lens_file"]  # lenses filename

    # read jackknife indices of the lenses
    jackknife_idx = pd.read_csv(config["files"]["jackknife"])[
        "jack_idx"].values

    # read the lens data into a pandas.DataFrame object, having features including
    # coadd_object_id, ra, dec, zredmagic and lum_z
    # NOTE: ra and dec must be in radians
    lenses = pd.DataFrame(
        reading_lens_params(
            lens_fname,
            jackknife_idx,
            z_min,
            z_max,
            config["frac_bright"],
            config["z_bins_selec"],
        )
    )
    # lenses = lenses.T.dropna().T # dropping the nan

    # precalculate the lens constants and comoving dist
    # lens constants & # comoving distances to the lenses
    lenses["const"], lenses["cdist"] = get_lens_constants(
        lenses, comoving_distance
    )

    #
    # create a ball-tree using the lens positions for efficent neighbor search
    #
    sys.stderr.write(f"Rank({rank}): Creating the lens tree...\n")
    lens_bt = BallTree(
        data=lenses[["dec", "ra"]].to_numpy(),
        # leaf_size = 20,
        # metric    = 'haversine' # metric on a spherical surface
    )

    #
    # read files: source catalog -> ra, dec, redshift etc
    sys.stderr.write(f"Rank({rank}): Creating source file objects...\n")
    # source shape data
    srcs_file = h5py.File(config["files"]["src_shape_file"], "r")
    srcz_file = h5py.File(
        config["files"]["src_redshift_file"], "r"
    )  # source redshifts

    # size of the sources
    src_size = srcs_file["catalog"]["unsheared"]["e_1"].shape[0]
    chunk_size = config["chunk_size"]  # size of each sub catalog

    #
    # read the catalogs and calculate the delta-sigma values
    #
    dsigma_num = np.zeros(r_bins - 1)
    denom = np.zeros(r_bins - 1)
    dsigma_num_cross = np.zeros(r_bins - 1)

    dsigmaalt_num = np.zeros(r_bins - 1)
    dsigmaalt_num_cross = np.zeros(r_bins - 1)

    num_pairs = np.zeros(r_bins - 1)

    # calculate the bin edges TODO
    r_edges = np.logspace(
        np.log10(r_min), np.log10(r_max), r_bins
    )  # log space bin edges

    # nnDB   = [] # a database for the holding the source chunks and the neighbour data
    # dsigma = [] # to store the delta-sigma values (TODO: check this)
    sys.stderr.write(f"Rank({rank}): Starting mainloop...\n")
    for i in range(src_size // chunk_size + 1):
        if i % size != rank:
            continue

        # load a subset of sources
        start = i * chunk_size
        stop = start + chunk_size
        sys.stderr.write(f"Loading sources from {start} to {stop}...\n")
        src_i = pd.DataFrame(reading_data_sources(
            srcs_file, srcz_file, start, stop))
        src_i = src_i.T.dropna().T  # dropping the nan
        src_i["cdist_mean"] = comoving_distance(
            src_i["zmean_sof"]
        )  # using mean redshift
        src_i["cdist_mc"] = comoving_distance(
            src_i["zmc_sof"])  # using mc redshift

        # find the nearest neighbours using the maximum radius
        sys.stderr.write(f"Rank({rank}): Searching for neighbours...\n")
        __t0 = time.time()
        # nnid, dist = lens_bt.query_radius( src_i[['dec', 'ra']].to_numpy(),
        #                                   theta_max,
        #                                   return_distance = True
        #                                )

        nnid = lens_bt.query_radius(src_i[["dec", "ra"]].to_numpy(), theta_max)
        sys.stderr.write(
            f"Rank({rank}): Completed in {time.time() - __t0:,} sec\n")

        #
        # calculating the average delta-sigma value
        #
        # jackknife mean and error TODO
        sys.stderr.write(f"Rank({rank}): Calculating increments...\n")
        __t0 = time.time()

        (
            delta_num,
            delta_num_cross,
            delta_den,
            deltaalt_num,
            deltaalt_num_cross,
            delta_npairs,
        ) = calculate_dsigma_increments_vector(
            src_i, lenses, nnid, r_edges
        )  # vectorized
        sys.stderr.write(f"Completed in {time.time() - __t0:,} sec\n")

        dsigma_num = dsigma_num + delta_num
        dsigma_num_cross = dsigma_num_cross + delta_num_cross
        denom = denom + delta_den

        dsigmaalt_num = dsigmaalt_num + deltaalt_num
        dsigmaalt_num_cross = dsigmaalt_num_cross + deltaalt_num_cross

        num_pairs += delta_npairs

        #
        # calculate delta-sigma and gamma-cross and write to file
        #
        sys.stderr.write(f"Rank({rank}): Calculating delta-sigma...\n")
        dsigma = dsigma_num / denom
        dsigma_cross = dsigma_num_cross / denom

        dsigmaalt = dsigmaalt_num / denom
        dsigmaalt_cross = dsigmaalt_num_cross / denom

        sys.stderr.write(f"Rank({rank}): Writing the output file...\n")
        pd.DataFrame(
            {
                # bin centers (linear)
                "r_center": 0.5 * (r_edges[1:] + r_edges[:-1]),
                "dsigma": dsigma,
                "dsigmaalt": dsigmaalt,
                "num_pairs": num_pairs,
                "dsigma_cross": dsigma_cross,
                "dsigma_num": dsigma_num,
                "dsigma_num_cross": dsigma_num_cross,
                "dsigmaalt_cross": dsigmaalt_cross,
                "dsigmaalt_num": dsigmaalt_num,
                "dsigmaalt_num_cross": dsigmaalt_num_cross,
                "denom": denom,
            }
        ).to_csv(
            f"{config[ 'files' ][ 'output' ]}.{rank:03d}",
            index=False,  # do not write the indices to the file
        )

    # ! REMOVE THIS FOR FULL LOOP
    # break

    sys.stderr.write(f"Rank({rank}): End of mainloop...\n")

    sys.stderr.write(f"Rank({rank}): The end...\n")
    return


#
# main function: parse the arguments and run the pipline
#


def mainloop():
    import argparse

    # creating the argument parser
    parser = argparse.ArgumentParser(
        description="Density profile calculations using weak lensing"
    )
    parser.add_argument(
        "config", metavar="file", type=str, nargs="?", help="path to the yaml config"
    )

    # parsing the arguments. if a correct path to a config file given, run the pipeline
    args = parser.parse_args()
    if args.config:
        run_pipeline(args.config)

    return


if __name__ == "__main__":
    mainloop()
