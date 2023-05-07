#!/usr/bin/env python
# coding: utf-8
# @author: AnirbanC


###########################################################################################################################

# This file calcuates the DeltaSigma (Surface Mass Density in units of M_sun/Mpc^2) for different radial bins(log scale).
####  It returns numerators and denominators which can be used to calculate Delta Sigma

###########################################################################################################################


import numpy as np
from tqdm import tqdm
from kdtreecode import get_xxyyzz_simple
from scipy.stats import binned_statistic #, binned_statistic_2d

def get_distance(lra, ldec, sra, sdec):                      #Function finds anguar distances of objects(lenses)[lra,ldec] and a source at [sra,sdec]
    xxl, yyl, zzl = get_xxyyzz_simple(lra,ldec)
    xxs, yys, zzs = get_xxyyzz_simple(sra,sdec)
    dist = xxl*xxs+yyl*yys+zzl*zzs
    dist[dist>1] = 1.0
    return np.arccos(dist)
    

PI = np.pi
G  = 4.299E-09   # gravitational constant in (km/sec)^2 Mpc/Msun
C  = 299792.4580 # speed of light in km/sec 

# Function finds lens constants used in the calculation of deltaSigma. lens constants are values that only depend on lenses in the Surface Mass Density estimator. Check Shin et al, sec 3 for more details


def get_lens_constants(lenses,c_dist) :                       
    
    halo_z=lenses["zredmagic"]
    halo_x=c_dist(halo_z)
    return 4*PI*G/(C**2)*(1+halo_z)*halo_x,halo_x


# Function calculates the DeltaSigma

###   src is the source data frame passed on to by core.py
###   lenses is the source data frame passed on to by core.py
###   nnid is a list(on sources) containing arrays which contain ids of all lenses linked to each source

def calculate_dsigma_increments (src,lenses,nnid,binedges) :
    
    jk_patches=max(lenses["jack_idx"])
    
    bin_min=binedges[0]
    bin_max=binedges[-1]
    
# initializing matrices    
    
    num_tan = np.zeros((len(binedges)-1),jk_patches)
    num_cross = np.zeros((len(binedges)-1),jk_patches)
    den = np.zeros((len(binedges)-1),jk_patches)

    numalt_tan = np.zeros((len(binedges)-1),jk_patches)
    numalt_cross = np.zeros((len(binedges)-1),jk_patches)

    numpairs = np.zeros((len(binedges)-1),jk_patches)
    

    ra= src["ra"]
    dec= src["dec"]
    z_mean= src["zmean_sof"]
    cdist_mean=src["cdist_mean"]
    z_mc=src["zmc_sof"]
    cdist_mc=src["cdist_mc"]
    e1= src["e_1"]
    e2=src["e_2"]
    R11 = src["R11"]
    R22 = src["R22"]  
    R=0.5*(R11+R22)
    w=src["weight"]
    for i in tqdm(range(len(ra))) :
        lens_id = np.array(nnid[i])
        nn_lens = lenses.iloc[lens_id]
        lens_jkIDX = np.array(nn_lens["jack_idx"])
        lens_ra= np.array(nn_lens["ra"])
        lens_dec= np.array(nn_lens["dec"])
        lens_z= np.array(nn_lens["zredmagic"])
        lens_constant = np.array(nn_lens["const"])
        lens_cdist= np.array(nn_lens["cdist"])
        lens_theta = get_distance(lens_ra, lens_dec, ra[i], dec[i])
         
#    Choose lenses only beind the source
        where = lens_z<z_mean[i]
        lens_id = lens_id[where]
        lens_jkIDX=lens_jkIDX[where]
        lens_theta = lens_theta[where]
        lens_ra= lens_ra[where]
        lens_dec= lens_dec[where]
        lens_z= lens_z[where]
        lens_constant = lens_constant[where]
        lens_cdist= lens_cdist[where]
                              

# calculate radial distances from theta
        
        lens_radDist = lens_theta * lens_cdist

        
# Choose only lenses less than bin_max
        
        where = (lens_radDist < bin_max) & (lens_radDist > bin_min)
        lens_id = lens_id[where]
        lens_jkIDX=lens_jkIDX[where]
        lens_theta = lens_theta[where]
        lens_ra= lens_ra[where]
        lens_dec= lens_dec[where]
        lens_z= lens_z[where]
        lens_constant = lens_constant[where]
        lens_cdist= lens_cdist[where]
        lens_radDist = lens_radDist[where]
        theta_abs = np.abs(np.sin(lens_theta))
        index = np.digitize(lens_radDist,binedges) -1

        
        for j in range ( len(lens_id)) :
            
            sin_phi2 = (-np.sin(lens_dec[j])*np.cos(dec[i]) + np.sin(dec[i])*np.cos(lens_dec[j])*np.cos(lens_ra[j]-ra[i]))/theta_abs[j]
            cos_phi2 = np.sin(ra[i]-lens_ra[j])*np.cos(lens_dec[j])/theta_abs[j]
            cos_phi2_sq = cos_phi2**2
            # sin_phi2 = np.sqrt( 1 - cos_phi2_sq )

            cos_2phi2 = 2*cos_phi2_sq - 1
            sin_2phi2 = 2*sin_phi2*cos_phi2

            e_tan = -e1[i]*cos_2phi2 - e2[i]*sin_2phi2
            e_cross = e1[i]*sin_2phi2 - e2[i]*cos_2phi2

            yij = lens_constant[j]*w[i]*(1-lens_cdist[j]/cdist_mean[i])
            
            num_tan[index[j]]+=yij*e_tan
            num_cross[index[j]]+=yij*e_cross

            den[index[j]]+=lens_constant[j]*yij*(1-lens_cdist[j]/cdist_mc[i])*R[i]

            ealt_tan = -e1[i]*cos_2phi2 + e2[i]*sin_2phi2
            ealt_cross = e1[i]*sin_2phi2 + e2[i]*cos_2phi2
            numalt_tan[index[j]]+=yij*ealt_tan
            numalt_cross[index[j]]+=yij*ealt_cross
            numpairs[index[j]] += 1
         
            
            num_tan[index[j]][lens_jkIDX[j]]-=yij*e_tan
            num_cross[index[j]][lens_jkIDX[j]]-=yij*e_cross
            den[index[j]][lens_jkIDX[j]]-=lens_constant[j]*yij*(1-lens_cdist[j]/cdist_mc[i])*R[i]

            ealt_tan = -e1[i]*cos_2phi2 + e2[i]*sin_2phi2
            ealt_cross = e1[i]*sin_2phi2 + e2[i]*cos_2phi2
            
            numalt_tan[index[j]][lens_jkIDX[j]]-=yij*ealt_tan
            numalt_cross[index[j]][lens_jkIDX[j]]-=yij*ealt_cross
            numpairs[index[j]][lens_jkIDX[j]]-= 1
                              
    return num_tan, num_cross, den, numalt_tan, numalt_cross
            
# vectorized vaersion of `calculate_dsigma_increments`
def calculate_dsigma_increments_vector_l(src, lenses, nnid, binedges, z_diff = 0.3):

    # calculation for single source
    def _calaculate_for_lens(ra_l, dec_l, z_l, const_l, cdist_l, src_id):

        # get nearest sources
        src_id = np.array( src_id )
        nn_src = src.iloc[ src_id ]

        # source parameters
        (
            ra_s, dec_s, z_mean_s, cdist_mean_s, z_mc_s, cdist_mc_s, e1_s, e2_s, w_s
        )   = nn_src[['ra', 'dec', 'zmean_sof', 'cdist_mean', 'zmc_sof', 'cdist_mc', 'e_1', 'e_2', 'weight']].to_numpy().T
        R_s = 0.5 * ( nn_src['R11'] + nn_src['R22'] ).to_numpy()

        # chose only sources behind the lenses, with min distance
        mask = ( z_l + z_diff < z_mean_s )
        ra_s, dec_s, z_mean_s, cdist_mean_s, z_mc_s, cdist_mc_s, e1_s, e2_s, w_s, R_s = ra_s[ mask ], dec_s[ mask ], z_mean_s[ mask ], cdist_mean_s[ mask ], z_mc_s[ mask ], cdist_mc_s[ mask ], e1_s[ mask ], e2_s[ mask ], w_s[ mask ], R_s[ mask ]

        if not len( ra_s ):
            nbins = len( binedges ) - 1
            return np.zeros( nbins ), np.zeros( nbins ), np.zeros( nbins ), np.zeros( nbins ), np.zeros( nbins ), np.zeros( nbins )
        
        theta = get_distance( ra_l, dec_l, ra_s, dec_s ) # angular distance in between in radian
        rad_l   = theta * cdist_l                        # radial distance in Mpc

        abs_sin_t = np.abs( np.sin( theta ) )

        sin_p2  = ( -np.sin( dec_l ) * np.cos( dec_s ) 
                        + np.sin( dec_s ) * np.cos( dec_l ) * np.cos( ra_l - ra_s ) ) / abs_sin_t

        cos_p2  = np.sin( ra_s - ra_l ) * np.cos( dec_s ) / abs_sin_t
        cos2_p2 = cos_p2**2
        cos_2p2, sin_2p2 = 2.*cos2_p2 - 1., 2.*sin_p2*cos_p2 # double angle values

        f1 = w_s * const_l * ( 1. - cdist_l / cdist_mean_s ) * R_s # a factor appearing in multiple times 
        f2 = const_l * ( 1. - cdist_l / cdist_mc_s ) * R_s         # a factor apperaing in denominator

        e_tan = -e1_s * cos_2p2 + e2_s * sin_2p2 # tangential ellipsicity
        e_crs =  e1_s * sin_2p2 + e2_s * cos_2p2 # cross direction ellipsicity

        # NOTE: alternative version with sign of e2 reversed
        # e_tan_alt = -e1_s * cos_2p2 - e2_s * sin_2p2 # tangential ellipsicity
        # e_crs_alt =  e1_s * sin_2p2 - e2_s * cos_2p2 # cross direction ellipsicity

        # using binned_statistics to bin the values with increments as weights
        # weights = 
        bstats = binned_statistic( rad_l, 
                                   values = [ f1 * e_tan,              # numerator for tangential part
                                              f1 * e_crs,              # numerator for cross part
                                            #   f1 * e_tan_alt,          # numerator for tangential part
                                            #   f1 * e_crs_alt,          # numerator for cross part
                                              f1 * f2,                 # denominator for both part
                                              np.ones( len( rad_l ) ), # pair counting
                                            ],
                                   bins = binedges,
                                   statistic = 'sum',
                            )
        # num_tan, num_crs, num_tan_alt, num_crs_alt, den_all, npairs = bstats.statistic
        num_tan, num_crs, den_all, npairs = bstats.statistic

        # bstats1 = binned_statistic_2d( rad_l, 
        #                              jack_idx_l,        # jacknife index of the patch                
        #                              values = weights,
        #                              bins = binedges,
        #                              statistic = 'sum',
        #                     )
        # num_tan1, num_crs1, num_tan_alt1, num_crs_alt1, den_all1, npairs1 = bstats1.statistic

        # # jackknife samples are given by removing the values corresponding to an object falling
        # # in that jaknife region: 
        # num_tan       = num_tan     - num_tan1     
        # num_crs       = num_crs     - num_crs1     
        # den_all       = den_all     - den_all1     
        # num_tan_alt   = num_tan_alt - num_tan_alt1     
        # num_crs_alt   = num_crs_alt - num_crs_alt1     
        # npairs        = npairs      - npairs1      
        
        # return num_tan, num_crs, den_all, num_tan_alt, num_crs_alt, npairs
        return num_tan, num_crs, den_all, npairs
    
    # calculate the values for all sources
    ra, dec, z, const, cdist, jkid = lenses[['ra', 'dec', 'zredmagic', 'const', 'cdist', 'jacknife_idx']].to_numpy().T

    ################################################################################
    # USING FOR LOOP
    ################################################################################ 
    nbins = len( binedges ) - 1
    njack = max( lenses['jacknife_idx'] ) + 1 # no. jaknife patches, assuming index start from 0
    (
        num_tan, num_crs, den_all#, num_tan_alt, num_crs_alt
    )      = np.zeros( nbins, njack ), np.zeros( nbins, njack ), np.zeros( nbins, njack )#, np.zeros( nbins, njack ), np.zeros( nbins, njack )
    npairs = np.zeros( nbins, njack )


    jkids = np.arange( njack ) # jackknife indices
    for l in tqdm( range( lenses.shape[0] ) ):
        (
            num_tan_s, num_crs_s, den_all_s, num_tan_alt_s, num_crs_alt_s, npairs_s
        ) = _calaculate_for_lens( ra[l], dec[l], z[l], const[l], cdist[l], nnid[l] )

        # in jackknife sampling, add the weights to all patches, except its patch
        mask = ( jkids != jkid[l] )

        num_tan[ mask ]     += num_tan_s[:]
        num_crs[ mask ]     += num_crs_s[:]
        # num_tan_alt[ mask ] += num_tan_alt_s[:]
        # num_crs_alt[ mask ] += num_crs_alt_s[:]
        den_all[ mask ]     += den_all_s[:]
        npairs[ mask ]      += npairs_s[:]

    # return num_tan, num_crs, den_all, num_tan_alt, num_crs_alt, npairs
    return num_tan, num_crs, den_all, npairs



