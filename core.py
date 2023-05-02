#!/usr/bin/python3
#
# code skelton: work in progress!
#

import numpy as np, pandas as pd
import astropy.cosmology as ac
import sys, argparse
from sklearn.neighbors import BallTree
from mpi4py import MPI
from typing import Any 

#3
# load specific modules
#
import reading_data_shape_redshift_catalog as rf # for file reading




#
# define the run pipeline
#       input: config
#
def  run_pipeline(config_filename):

    # read config file

    # cosmology model
    cm = ac.FlatLambdaCDM(H0 = ..., Om0 = ...)

    # NOTE: read files: lenses catalog -> ra, dec, redshift
    # read the lens data into a pandas.DataFrame object, having features including 
    # coadd_object_id, ra, dec, zredmagic and lum_z
    lenses = read_lens_catalog() 

    # create a tree from the lenses dataset
    bt = create_tree()

    # read source files in a loop
    source_size = ...
    chunk_size  = ... # from the config file

    dsigma_num = np.zeros(n_bins)
    dsigma_den = np.zeros(n_bins)

    sources = [] # store the source data
    dsigmas = [] # store the delta-sigma

    start  = 0
    for ii in range(source_size // chunk_size + 1):
        start = ii * chunk_size
        stop  = start + chunk_size

        source_chunk = read_sources_catalog()
        sources.append( source_chunk )

        # find the nearest neighouring sources for each lens and setup as a dataset
        nn_id, dist = get_neighbours()

        # average the delta-sigma
        dsigma_num_inc, dsigma_den_inc = calculate_dsigma_increments()
        dsigma_num += dsigma_num_inc
        dsigma_den += dsigma_den_inc

        dsigma = dsigma_num / dsigma_den # Delta Sigma
        dsigmas.append( dsigma )


if __name__ ==  '__main__':
    ...

    # read the config filename and pass it to the pipe line
