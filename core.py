#!/usr/bin/python3
# @author: m. s. sūryan śivadās
# @file:
#
# code skelton: almost complete?
#

import numpy as np
import pandas as pd
import sys
import time
import h5py  # to read hdf5 files
import yaml  # to parse yaml config files
from astropy.cosmology import FlatLambdaCDM  # cosmology model
from scipy.interpolate import CubicSpline   # for interpolations
from kdtreecode import BallTree             # for nearest neighbours
# for loading the catalogs
from reading_data_shape_redshift_catalog import (
    reading_lens_params,
    reading_data_sources
)
# for calculating delta-sigma
from calc_tngt_shear import (
    get_lens_constants,
    # calculate_dsigma_increments,
    calculate_dsigma_increments_vector
)

try:
    from mpi4py import MPI
except ImportError:
    print("MPI not found")

# define the function to run pipline.
# input: config filename


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
    with open(config_fname, 'r') as file:
        inputs = yaml.safe_load(file)

    #
    # creating a flat-lcdm model using the given cosmology parameters.
    #
    cm = FlatLambdaCDM(H0=inputs['cosmology']['H0'],
                       Om0=inputs['cosmology']['OmegaMatter']
                       )  # defaults: Tcmb = 0K, Ob0 = None

    # redshift range
    z_min = inputs['z_min']
    z_max = inputs['z_max']

    #
    # calculate and interpolate the comoving distances for future use
    #
    z = np.linspace(max(z_min - 0.1, 0.), min(z_max +
                    0.1, 10.), inputs['z_bins_intrp'])
    xz = cm.comoving_distance(z)

    # spline object for comoving distance calculation
    comoving_distance = CubicSpline(z, xz)  # can be accesses like a function

    # parameters for binning neighbour distance
    r_min = inputs['r_min']  # lower distance bound
    r_max = inputs['r_max']  # upper distance bound
    r_bins = inputs['r_bins']  # number of bins

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
    lens_fname = inputs['files']['lens_file']  # lenses filename

    # read jackknife indices of the lenses
    jackknife_idx = pd.read_csv(
        inputs['files']['jackknife']
    )["jack_idx"].values

    # read the lens data into a pandas.DataFrame object, having features including
    # coadd_object_id, ra, dec, zredmagic and lum_z
    # NOTE: ra and dec must be in radians
    lenses = pd.DataFrame(reading_lens_params(
        lens_fname,
        jackknife_idx,
        z_min, z_max,
        inputs['frac_bright'], inputs['z_bins_selec']
    ))
    # lenses = lenses.T.dropna().T # dropping the nan

    # precalculate the lens constants and comoving dist
    lconst, lcmdist = get_lens_constants(lenses, comoving_distance)
    lenses['const'] = lconst  # lens constants
    lenses['cdist'] = lcmdist  # comoving distances to the lenses

    #
    # create a ball-tree using the lens positions for efficent neighbor search
    #
    sys.stderr.write(f"Rank({rank}): Creating the lens tree...\n")
    lens_bt = BallTree(data=lenses[['dec', 'ra']].to_numpy(),
                       # leaf_size = 20,
                       # metric    = 'haversine' # metric on a spherical surface
                       )
    # lens_bt = BallTree( lenses[['dec', 'ra']].to_numpy(),
    #                )

    #
    # read files: source catalog -> ra, dec, redshift etc
    #
    sys.stderr.write(f"Rank({rank}): Creating source file objects...\n")
    # source shape data
    srcs_file = h5py.File(inputs['files']['src_shape_file'], 'r')
    srcz_file = h5py.File(
        inputs['files']['src_redshift_file'], 'r')  # source redshifts

    # size of the sources
    src_size = srcs_file["catalog"]["unsheared"]["e_1"].shape[0]
    chunk_size = inputs['chunk_size']   # size of each sub catalog

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
    r_edges = np.logspace(np.log10(r_min), np.log10(
        r_max), r_bins)  # log space bin edges

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
        src_i['cdist_mean'] = comoving_distance(
            src_i['zmean_sof'])  # using mean redshift
        src_i['cdist_mc'] = comoving_distance(
            src_i['zmc_sof'])   # using mc redshift

        # find the nearest neighbours using the maximum radius
        sys.stderr.write(f"Rank({rank}): Searching for neighbours...\n")
        __t0 = time.time()
        # nnid, dist = lens_bt.query_radius( src_i[['dec', 'ra']].to_numpy(),
        #                                   theta_max,
        #                                   return_distance = True
        #                                )

        nnid = lens_bt.query_radius(src_i[['dec', 'ra']].to_numpy(), theta_max)
        sys.stderr.write(
            f"Rank({rank}): Completed in {time.time() - __t0:,} sec\n")

        #
        # calculating the average delta-sigma value
        #
        # jackknife mean and error TODO
        sys.stderr.write(f"Rank({rank}): Calculating increments...\n")
        __t0 = time.time()

        delta_num, delta_num_cross, delta_den, deltaalt_num, \
            deltaalt_num_cross, delta_npairs = calculate_dsigma_increments_vector(
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
            {'r_center': 0.5*(r_edges[1:] + r_edges[:-1]),  # bin centers (linear)
             'dsigma': dsigma,
             'dsigmaalt': dsigmaalt,
             'num_pairs': num_pairs,
             'dsigma_cross': dsigma_cross,
             'dsigma_num': dsigma_num,
             'dsigma_num_cross': dsigma_num_cross,
             'dsigmaalt_cross': dsigmaalt_cross,
             'dsigmaalt_num': dsigmaalt_num,
             'dsigmaalt_num_cross': dsigmaalt_num_cross,
             'denom': denom,
             }).to_csv(f"{inputs[ 'files' ][ 'output' ]}.{rank:03d}",
                       index=False,  # do not write the indices to the file
                       )

        # break # for testing, stop after first iteration
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
        description="Density profile calculations using weak lensing")
    parser.add_argument('config',
                        metavar='file',
                        type=str,
                        nargs='?',
                        help='path to the configuration (yaml) file'
                        )

    # parsing the arguments. if a correct path to a config file given, run the pipeline
    args = parser.parse_args()
    if args.config:
        run_pipeline(args.config)

    return


if __name__ == '__main__':
    mainloop()
