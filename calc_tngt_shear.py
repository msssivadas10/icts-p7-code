#!/usr/bin/env python
# coding: utf-8
# @author: AnirbanC

import numpy as np
from tqdm import tqdm
from kdtreecode import get_xxyyzz_simple

def get_distance(lra, ldec, sra, sdec):
    xxl, yyl, zzl = get_xxyyzz_simple(lra,ldec)
    xxs, yys, zzs = get_xxyyzz_simple(sra,sdec)
    dist = xxl*xxs+yyl*yys+zzl*zzs
    dist[dist>1] = 1.0
    return np.arccos(dist)
    

PI = np.pi
G  = 4.299E-09   # gravitational constant in (km/sec)^2 Mpc/Msun
C  = 299792.4580 # speed of light in km/sec 

def get_lens_constants(lenses,c_dist) :
    
    halo_z=lenses["zredmagic"]
    halo_x=c_dist(halo_z)
    return 4*PI*G/(C**2)*(1+halo_z)*halo_x,halo_x


def calculate_dsigma_increments (src,lenses,nnid,binedges) :
    
    bin_min=binedges[0]
    bin_max=binedges[-1]
    
    num_tan = np.zeros(len(binedges)-1)
    num_cross = np.zeros(len(binedges)-1)
    den = np.zeros(len(binedges)-1)

    numalt_tan = np.zeros(len(binedges)-1)
    numalt_cross = np.zeros(len(binedges)-1)
    
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
        lens_ra= np.array(nn_lens["ra"])
        lens_dec= np.array(nn_lens["dec"])
        lens_z= np.array(nn_lens["zredmagic"])
        lens_constant = np.array(nn_lens["const"])
        lens_cdist= np.array(nn_lens["cdist"])
        lens_theta = get_distance(lens_ra, lens_dec, ra[i], dec[i])
         
#    Choose lenses only beind the source
        where = lens_z<z_mean[i]
        lens_id = lens_id[where]
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
        lens_theta = lens_theta[where]
        lens_ra= lens_ra[where]
        lens_dec= lens_dec[where]
        lens_z= lens_z[where]
        lens_constant = lens_constant[where]
        lens_cdist= lens_cdist[where]
        lens_radDist = lens_radDist[where]
        theta_abs = np.abs(np.sin(lens_theta))
        index = np.digitize(np.log10(lens_radDist),binedges)

        
        for j in range ( len(lens_id)) :
            
            sin_phi2 = (-1*np.sin(lens_dec[j])*np.cos(dec[i]) + np.sin(dec[i])*np.cos(lens_dec[j])*np.cos(lens_ra[j]-ra[i]))/theta_abs[j]
            cos_phi2 = np.sin(ra[i]-lens_ra[j])*np.cos(lens_dec[j])/theta_abs[j]

            cos_2phi2 = 2*cos_phi2*cos_phi2 - 1
            sin_2phi2 = 2*sin_phi2*cos_phi2
            e_tan = -1*e1[i]*cos_2phi2 - e2[i]*sin_2phi2
            e_cross = e1[i]*sin_2phi2 - e2[i]*cos_2phi2
            num_tan[index]+=lens_constant[j]*w[i]*(1-lens_cdist[j]/cdist_mean[i])*e_tan
            num_cross[index]+=lens_constant[j]*w[i]*(1-lens_cdist[j]/cdist_mean[i])*e_cross
            den[index]+=(lens_constant[j]**2)*w[j]*(1-lens_cdist[j]/cdist_mean[i])*(1-lens_cdist[j]/cdist_mc[i])*R[i]

            ealt_tan = -1*e1[i]*cos_2phi2 + e2[i]*sin_2phi2
            ealt_cross = e1[i]*sin_2phi2 + e2[i]*cos_2phi2
            numalt_tan[index]+=lens_constant[j]*w[i]*(1-lens_cdist[j]/cdist_mean[i])*ealt_tan
            numalt_cross[index]+=lens_constant[j]*w[i]*(1-lens_cdist[j]/cdist_mean[i])*ealt_cross

    return num_tan, num_cross, den, numalt_tan, numalt_cross
            
 
