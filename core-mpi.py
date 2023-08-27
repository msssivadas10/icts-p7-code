#!/usr/bin/python3
# @author: m. s. sūryan śivadās
# @file:
#
# parallel code skelton: work in progress
#

import sys
import time
from enum import Enum

import h5py  # to read hdf5 files
import numpy as np
import pandas as pd
import yaml  # to parse yaml config files
from astropy.cosmology import FlatLambdaCDM  # cosmology model
from mpi4py import MPI
from scipy.interpolate import CubicSpline  # for interpolations

# for calculating delta-sigma
from src.calc_tngt_shear import calculate_dsigma_increments, get_lens_constants
# from sklearn.neighbors import BallTree
from src.kdtreecode import BallTree  # for nearest neighbours
# for loading the catalogs
from src.reading_data_shape_redshift_catalog import (
    reading_data_sources,
    reading_lens_params
)

# define the function to run pipline.
# input: config filename


class MPITag(Enum):
    NUM = 14
    NUMCROSS = 15
    DENOM = 16


def run_pipeline(config_fname):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()  # rank of the process

    #
    # reading the config file
    #
    sys.stderr.write("Reading the config file...\n")
    with open(config_fname, "r") as file:
        inputs = yaml.safe_load(file)

    #
    # creating a flat-lcdm model using the given cosmology parameters.
    #
    cm = FlatLambdaCDM(
        H0=inputs["cosmology"]["H0"], Om0=inputs["cosmology"]["OmegaMatter"]
    )  # defaults: Tcmb = 0K, Ob0 = None

    # redshift range
    z_min = inputs["z_min"]
    z_max = inputs["z_max"]

    #
    # calculate and interpolate the comoving distances for future use
    #
    z = np.linspace(max(z_min - 0.1, 0.0),
                    min(z_max + 0.1, 10.0), inputs["z_bins"])
    xz = cm.comoving_distance(z)

    # spline object for comoving distance calculation
    comoving_distance = CubicSpline(z, xz)  # can be accesses like a function

    # parameters for binning neighbour distance
    r_min = inputs["r_min"]  # lower distance bound
    r_max = inputs["r_max"]  # upper distance bound
    r_bins = inputs["r_bins"]  # number of bins

    #
    # find the angular seperation for nn search
    #
    # comoving distance corresponding to minimum redshift
    cmdist = comoving_distance(z_min)
    theta_max = r_max / cmdist  # maximum angular seperation

    #
    # read files: lenses catalog -> ra, dec, redshift
    #
    sys.stderr.write("Reading the lens catalog...\n")
    lens_fname = inputs["files"]["lens_file"]  # lenses filename

    # read the lens data into a pandas.DataFrame object, having features including
    # coadd_object_id, ra, dec, zredmagic and lum_z
    # NOTE: ra and dec must be in radians
    lenses = pd.DataFrame(
        reading_lens_params(lens_fname, z_min, z_max, inputs["frac_bright"])
    )
    lenses = lenses.dropna()  # dropping the nan

    # precalculate the lens constants and comoving dist
    lconst, lcmdist = get_lens_constants(lenses, comoving_distance)
    lenses["const"] = lconst  # lens constants
    lenses["cdist"] = lcmdist  # comoving distances to the lenses

    #
    # create a ball-tree using the lens positions for efficent neighbor search
    #
    sys.stderr.write("Creating the lens tree...\n")
    lens_bt = BallTree(
        data=lenses[["dec", "ra"]].to_numpy(),
        # leaf_size = 20,
        # metric    = 'haversine' # metric on a spherical surface
    )

    #
    # read files: source catalog -> ra, dec, redshift etc
    #
    sys.stderr.write("Creating source file objects...\n")
    # source shape data
    srcs_file = h5py.File(inputs["files"]["src_shape_file"], "r")
    srcz_file = h5py.File(
        inputs["files"]["src_redshift_file"], "r")  # source redshifts

    # size of the sources
    src_size = srcs_file["catalog"]["unsheared"]["e_1"].shape[0]
    chunk_size = inputs["chunk_size"]  # size of each sub catalog

    #
    # dividing the job among the processes
    #
    count = src_size // size
    rem = src_size % size

    if rank < rem:
        chunk_start = rank * (count + 1)
        chunk_stop = chunk_start + count + 1
    else:
        chunk_start = rank * count + rem
        chunk_stop = chunk_start + count

    #
    # read the catalogs and calculate the delta-sigma values
    #
    dsigma_num = np.zeros(r_bins - 1)
    denom = np.zeros(r_bins - 1)
    dsigma_num_cross = np.zeros(r_bins - 1)

    # calculate the bin edges TODO
    r_edges = np.logspace(
        np.log10(r_min), np.log10(r_max), r_bins
    )  # log space bin edges

    # nnDB   = [] # a database for the holding the source chunks and the neighbour data
    # dsigma = [] # to store the delta-sigma values (TODO: check this)
    sys.stderr.write("Starting mainloop...\n")
    for i in range(chunk_start, chunk_stop):
        # load a subset of sources
        start = i * chunk_size
        stop = start + chunk_size
        sys.stderr.write(f"Loading sources from {start} to {stop}...\n")
        src_i = pd.DataFrame(reading_data_sources(
            srcs_file, srcz_file, start, stop
        ))
        src_i = src_i.dropna()  # dropping the nan
        src_i["cdist_mean"] = comoving_distance(
            src_i["zmean_sof"]
        )  # using mean redshift
        src_i["cdist_mc"] = comoving_distance(
            src_i["zmc_sof"]
        )  # using mc redshift

        # find the nearest neighbours using the maximum radius
        sys.stderr.write("Searching for neighbours...\n")
        __t0 = time.time()
        # nnid, dist = lens_bt.query_radius(
        #   src_i[['dec', 'ra']].to_numpy(),
        #   theta_max,
        #   return_distance = True
        # )

        nnid = lens_bt.query_radius(src_i[["dec", "ra"]].to_numpy(), theta_max)
        sys.stderr.write(f"Completed in {time.time() - __t0:,} sec\n")

        # NOTE 1: `nnid` and `dist` are arrays of arrays so that, each sub-array
        # correspond to neighbours of a specific source. i.e., `i`-th sub-array will
        # match to the `i`-th source in the sources dataset
        #
        # NOTE 2: combining `nnid` and `dist` for a specific source (specified by index)
        # into a 2d array with col-1 => index or id of the lenses and col-2 => distance.
        # if the source has the index `j` in the source dataset, then corresponding
        # neighbours will be in at index `j` in the list
        # nn_i = list( map(lambda __o: np.stack([__o], 1), zip( nnid, dist )) )
        # # join the 2 arrays along col
        # nnDB.append([ src_i, nn_i ])

        #
        # calculating the average delta-sigma value
        #
        # jackknife mean and error TODO
        sys.stderr.write("Calculating increments...\n")
        __t0 = time.time()
        # delta_num, delta_num_cross, delta_den = calculate_dsigma_increments(
        # src_i, lenses, nnid, dist, r_edges
        # )
        delta_num, delta_num_cross, delta_den = calculate_dsigma_increments(
            src_i, lenses, nnid, r_edges
        )
        sys.stderr.write(f"Completed in {time.time() - __t0:,} sec\n")

        dsigma_num = dsigma_num + delta_num
        dsigma_num_cross = dsigma_num_cross + delta_num_cross
        denom = denom + delta_den

        # ! REMOVE THIS FOR TESTING LOOP
        # break  # for testing, stop after first iteration

    sys.stderr.write("End of mainloop...\n")

    #
    # receive data from other processes and combine (provided using np.array)
    #
    if rank > 0:
        # send data from every other process, except 0
        # FIXME is the `tag` can be any id
        comm.Send(dsigma_num, dest=0, tag=MPITag.NUM)
        comm.Send(dsigma_num_cross, dest=0, tag=MPITag.NUMCROSS)
        comm.Send(denom, dest=0, tag=MPITag.DENOM)
    else:
        # recieve from all others at 0
        for __proc in range(1, size):
            _dsigma_num = np.empty(r_bins - 1)
            comm.Recv(_dsigma_num, source=__proc, tag=MPITag.NUM)

            _dsigma_num_cross = np.empty(r_bins - 1)
            comm.Recv(_dsigma_num_cross, source=__proc, tag=MPITag.NUMCROSS)

            _denom = np.empty(r_bins - 1)
            comm.Recv(_denom, source=__proc, tag=MPITag.DENOM)

            dsigma_num = dsigma_num + _dsigma_num
            dsigma_num_cross = dsigma_num_cross + _dsigma_num_cross
            denom = denom + _denom

        #
        # calculate delta-sigma and gamma-cross and write to file
        #
        # sys.stderr.write("Calculating delta-sigma...\n")
        dsigma = dsigma_num / denom
        dsigma_cross = dsigma_num_cross / denom

        sys.stderr.write("Writing the output file...\n")
        pd.DataFrame(
            {
                # bin centers (linear)
                "r_center": 0.5 * (r_edges[1:] + r_edges[:-1]),
                "dsigma": dsigma,
                "dsigma_cross": dsigma_cross,
            }
        ).to_csv(
            inputs["files"]["output"],  # output filename
            index=False,  # do not write the indices to the file
        )

        sys.stderr.write("The end...\n")


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
        "config",
        metavar="file",
        type=str,
        nargs="?",
        help="path to the configuration (yaml) file",
    )

    # parsing the arguments. if a correct path to a config file given, run the pipeline
    args = parser.parse_args()
    if args.config:
        run_pipeline(args.config)

    return


if __name__ == "__main__":
    mainloop()
