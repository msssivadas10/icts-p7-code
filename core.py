#!/usr/bin/python3
# @author: m. s. sūryan śivadās 
# @file: 
#
# code skelton: work in progress
#

import numpy as np, pandas as pd
import sys, time
import h5py # to read hdf5 files
import yaml # to parse yaml config files
from astropy.cosmology import FlatLambdaCDM # cosmology model
from scipy.interpolate import CubicSpline   # for interpolations
#from sklearn.neighbors import BallTree      # for nearest neighbours
from kdtreecode import BallTree
from reading_data_shape_redshift_catalog import reading_lens_params, reading_data_sources # for loading the catalogs
from calc_tngt_shear import get_lens_constants, calculate_dsigma_increments # for calculating delta-sigma

# define the function to run pipline.
# input: config filename  
def run_pipeline(config_fname):

    # 
    # reading the config file
    #
    sys.stderr.write("Reading the config file...\n")
    with open( config_fname, 'r' ) as file:
        inputs = yaml.safe_load( file )

    #
    # creating a flat-lcdm model using the given cosmology parameters. 
    # 
    cm = FlatLambdaCDM( H0  = inputs[ 'cosmology' ][ 'H0' ], 
                        Om0 = inputs[ 'cosmology' ][ 'OmegaMatter' ] 
                    ) # defaults: Tcmb = 0K, Ob0 = None
    
    # redshift range
    z_min = inputs[ 'z_min' ]
    z_max = inputs[ 'z_max' ]

    #
    # calculate and interpolate the comoving distances for future use
    #
    z  = np.linspace( max( z_min - 0.1, 0. ), min( z_max + 0.1, 10. ), inputs[ 'z_bins' ] ) 
    xz = cm.comoving_distance( z ) 
    
    # spline object for comoving distance calculation
    comoving_distance = CubicSpline( z, xz ) # can be accesses like a function

    # parameters for binning neighbour distance
    r_min  = inputs[ 'r_min' ]  # lower distance bound
    r_max  = inputs[ 'r_max' ]  # upper distance bound
    r_bins = inputs[ 'r_bins' ] # number of bins

    #
    # find the angular seperation for nn search
    #
    cmdist = comoving_distance( z_min ) # comoving distance corresponding to minimum redshift
    theta_max = r_max / cmdist          # maximum angular seperation

    # 
    # read files: lenses catalog -> ra, dec, redshift
    #
    sys.stderr.write("Reading the lens catalog...\n")
    lens_fname = inputs[ 'files' ][ 'lens_file' ] # lenses filename
    
    # read the lens data into a pandas.DataFrame object, having features including 
    # coadd_object_id, ra, dec, zredmagic and lum_z 
    # NOTE: ra and dec must be in radians 
    lenses = pd.DataFrame( reading_lens_params( lens_fname, z_min, z_max, inputs[ 'frac_bright' ] ) )
    
    lconst, lcmdist = get_lens_constants( lenses, comoving_distance ) # precalculate the lens constants and comoving dist
    lenses['const'] = lconst  # lens constants
    lenses['cdist'] = lcmdist # comoving distances to the lenses

    #
    # create a ball-tree using the lens positions for efficent neighbor search
    #
    sys.stderr.write("Creating the lens tree...\n")
    #lens_bt = BallTree( data      = lenses[['dec', 'ra']].to_numpy(), 
    #                    # leaf_size = 20, 
    #                    # metric    = 'haversine' # metric on a spherical surface
    #                )
    lens_bt = BallTree( lenses[['dec', 'ra']].to_numpy(), 
                    )

    #
    # read files: source catalog -> ra, dec, redshift etc
    #
    sys.stderr.write("Creating source file objects...\n")
    srcs_file = h5py.File( inputs[ 'files' ][ 'src_shape_file' ] , 'r' ) # source shape data
    srcz_file = h5py.File( inputs[ 'files' ][ 'src_redshift_file' ], 'r' ) # source redshifts

    src_size   = srcs_file["catalog"]["unsheared"]["e_1"].shape[0] # size of the sources
    chunk_size = inputs[ 'chunk_size' ]   # size of each sub catalog

    #
    # read the catalogs and calculate the delta-sigma values 
    #
    dsigma_num      = np.zeros( r_bins )
    denom           = np.zeros( r_bins )
    dsigma_num_cross = np.zeros( r_bins )

    # calculate the bin edges TODO
    r_edges = np.logspace( np.log10( r_min ), np.log10( r_max ), r_bins ) # log space bin edges

    # nnDB   = [] # a database for the holding the source chunks and the neighbour data
    # dsigma = [] # to store the delta-sigma values (TODO: check this)
    sys.stderr.write("Starting mainloop...\n")
    for i in range( src_size // chunk_size + 1 ):

        # load a subset of sources 
        start = i * chunk_size
        stop  = start + chunk_size
        sys.stderr.write(f"Loading sources from {start} to {stop}...\n")
        src_i = pd.DataFrame( reading_data_sources( srcs_file, srcz_file, start, stop ) )
        src_i['cdist_mean'] = comoving_distance( src_i['zmean_sof'] ) # using mean redshift
        src_i['cdist_mc']   = comoving_distance( src_i['zmc_sof'] )   # using mc redshift

        # find the nearest neighbours using the maximum radius
        sys.stderr.write("Searching for neighbours...\n")
        __t0 = time.time()
        nnid, dist = lens_bt.query_radius( src_i[['dec', 'ra']].to_numpy(), 
                                           theta_max, 
                                           return_distance = True 
                                        )
        sys.stderr.write(f"Completed in {time.time() - __t0:,} sec\n")
        
        # NOTE 1: `nnid` and `dist` are arrays of arrays so that, each sub-array 
        # correspond to neighbours of a specific source. i.e., `i`-th sub-array will 
        # match to the `i`-th source in the sources dataset 
        #
        # NOTE 2: combining `nnid` and `dist` for a specific source (specified by index) 
        # into a 2d array with col-1 => index or id of the lenses and col-2 => distance.
        # if the source has the index `j` in the source dataset, then corresponding 
        # neighbours will be in at index `j` in the list
        # nn_i = list( map(lambda __o: np.stack([__o], 1), zip( nnid, dist )) ) # join the 2 arrays along col
        # nnDB.append([ src_i, nn_i ])

        # 
        # calculating the average delta-sigma value
        #
        # jackknife mean and error TODO
        sys.stderr.write("Calculating increments...\n")
        __t0 = time.time()
        delta_num, delta_num_cross, delta_den = calculate_dsigma_increments( src_i, lenses, nnid, dist, r_edges )
        sys.stderr.write(f"Completed in {time.time() - __t0:,} sec\n")
        
        dsigma_num      = dsigma_num + delta_num
        dsigma_num_cross = dsigma_num_cross + delta_num_cross
        denom           = denom + delta_den

        break # for testing, stop after first iteration
    
    sys.stderr.write("End of mainloop...\n")

    #
    # calculate delta-sigma and gamma-cross and write to file
    #
    sys.stderr.write("Calculating delta-sigma...\n")
    dsigma      = dsigma_num / denom
    dsigma_cross = dsigma_num_cross / denom
    
    sys.stderr.write("Writing the output file...\n")
    pd.DataFrame(
                    { 'r_center'   : 0.5*(r_bins[1:] + r_bins[:-1]), # bin centers (linear)
                      'dsigma'     : dsigma,
                      'dsigma_cross': dsigma_cross, 
                }).to_csv( inputs[ 'files' ][ 'output' ], # output filename
                           index = False,                 # do not write the indices to the file
                        )
    
    sys.stderr.write("The end...\n")
    return

# 
# main function: parse the arguments and run the pipline
#
def mainloop():

    import argparse

    # creating the argument parser
    parser = argparse.ArgumentParser(description = "Density profile calculations using weak lensing")
    parser.add_argument( 'config', 
                         metavar = 'file', 
                         type    = str, 
                         nargs   = '?', 
                         help    = 'path to the configuration (yaml) file' 
                    )
    
    # parsing the arguments. if a correct path to a config file given, run the pipeline
    args = parser.parse_args()
    if args.config:
        run_pipeline( args.config )

    return

if __name__ == '__main__':
    mainloop()

