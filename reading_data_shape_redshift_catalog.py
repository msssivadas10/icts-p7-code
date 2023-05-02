#!/usr/bin/env python
# @uthur: Mukesh K Singh
# Date: 02 May 2023


import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy import units as u
import astropy.coordinates as coord

from matplotlib import rc, rcParams
rc_params = {'axes.labelsize': 20,
             'axes.titlesize': 20,
             'axes.linewidth':2,
             'font.size': 20,
             'lines.linewidth' : 2.5,
             'legend.fontsize': 15,
             'xtick.labelsize': 20,
             'ytick.labelsize': 20
            }
rcParams.update(rc_params)

def reading_shape_data(file_data, params, shear_flag="unsheared", start=0, end=1000000):
    """
    file_data: open file data to the shape catalog
    params:   name of the parameters to be read
    shear_flag: `unsheared` or `sheared_1m`, `sheared_1p`, `sheared_2p`, `sheared_2m` (default is sheared)
    start:    index to start from (int)
    end:      index to end at (int)
    """
    data = {}
    for param_key in params:
        data[param_key] = file_data["catalog"][shear_flag][param_key][start:end]
        
    return pd.DataFrame(data)
    
def reading_DNF_redshift(file_data, params, shear_flag="unsheared", start=0, end=1000000):
    """
    file_data: open file data to the shape catalog
    params:   name of the parameters to be read
    shear_flag: `unsheared` or `sheared_1m`, `sheared_1p`, `sheared_2p`, `sheared_2m` (default is sheared)
    start:    index to start from (int)
    end:      index to end at (int)
    """
    data = {}
    for param_key in params:
        data[param_key] = file_data["catalog"][shear_flag][param_key][start:end]
        
    return pd.DataFrame(data)

def reading_lens_params(filename, z_min=0.01, z_max=4, frac=0.01):
    
    lens_params = ['coadd_object_id', 'ra', 'dec', 'zredmagic', 'lum_z']
    
    data = fits.getdata(filename)
    header = fits.getheader(filename)
    
    # selecting a fraction 
    idx = np.array(np.random.random(size=data["lum_z"].size)<frac) \
            & np.array(data["zredmagic"]>z_min) \
            & np.array(data["zredmagic"]<z_max)
    
    data_dict = {}
    for key in lens_params:
        data_dict[key] = data[key][idx]
    return data_dict


def reading_data_sources(start, end):
    """
    start:
    end: 
    """
    # The parameters needed to computing differential surface density
    shape_params = ['coadd_object_id', 'ra', 'dec', 'e_1', 'e_2',\
                    'snr', 'weight', 'flags', 'size_ratio', 'T', \
                    'R11', 'R12', 'R21', 'R22']
    redshift_params = ['coadd_object_id', 'zmc_sof', 'zmean_sof']
    
    data1 = reading_shape_data(file_data=shape_file_data, params=shape_params, start=start, end=end)
    data2 = reading_shape_data(file_data=redshift_file_data, params=redshift_params, start=start, end=end)
    data1["zmc_sof"] = data2["zmc_sof"]
    data1["zmean_sof"] = data2["zmean_sof"]
    
    # Metacal selection
    data = data1
    flags = 0
    snr_th_lower = 10
    snr_th_upper = 1000
    size_ratio = 0.5
    T = 10

    idx = np.array(data["flags"] == flags) & np.array(data["snr"] > snr_th_lower) & \
    np.array(data["snr"] < snr_th_upper) & np.array(data["size_ratio"] > size_ratio) & \
    np.array(data["T"] < T)
    
    data_selected = dict()
    for key in list(data.keys()):
        data_selected[key] = data[key].values[idx]
        
    return data_selected

file = "/home/idies/workspace/Temporary/surhudm/scratch/DES/DESY3_metacal_v03-004.h5"
file_z = "/home/idies/workspace/Temporary/surhudm/scratch/DES/DESY3_GOLD_2_2.1_DNF.h5"
shape_file_data = h5py.File(file, "r")
redshift_file_data = h5py.File(file_z, "r")

# t0 = time.time()
# Nsize = shape_file_data["catalog"]["unsheared"]["e_1"].shape[0]
# chunksize = 50000000
# for ii in range(0, Nsize//chunksize + 1):
#     start = ii*chunksize
#     end = (ii+1)*chunksize
#     data_selected = reading_data_sources(start=start, end=end) 
#     break
# tf = time.time()
# print("Time taken: %f seconds!"%(tf-t0))


fname = "/home/idies/workspace/Temporary/surhudm/scratch/DES/y3a2_gold2.2.1_redmagic_highdens.fits"
lens_data = reading_lens_params(filename=fname)


