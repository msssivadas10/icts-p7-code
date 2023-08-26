#!/usr/bin/env python
# @uthur: Mukesh K Singh
# Date: 02 May 2023


import numpy as np
import pandas as pd
from astropy.io import fits


def reading_shape_data(file_data, params, shear_flag="unsheared", start=0, end=1000000):
    """
    file_data: open file data to the shape catalog
    params:   name of the parameters to be read
    shear_flag: `unsheared` or `sheared_1m`, `sheared_1p`,
                `sheared_2p`, `sheared_2m` (default is sheared)
    start:    index to start from (int)
    end:      index to end at (int)
    """
    data = {}
    for param_key in params:
        data[param_key] = file_data["catalog"][shear_flag][param_key][start:end]

    return pd.DataFrame(data)


def reading_DNF_redshift(
        file_data, params, shear_flag="unsheared", start=0, end=1000000
):
    """
    file_data: open file data to the shape catalog
    params:   name of the parameters to be read
    shear_flag: `unsheared` or `sheared_1m`, `sheared_1p`,
                `sheared_2p`, `sheared_2m` (default is sheared)
    start:    index to start from (int)
    end:      index to end at (int)
    """
    data = {}
    for param_key in params:
        data[param_key] = file_data["catalog"][shear_flag][param_key][start:end]

    return pd.DataFrame(data)


def reading_lens_params(
        filename, jacknife_idx, z_min=0.01, z_max=4, frac=0.01, zbins=5
):

    lens_params = ['coadd_object_id', 'ra', 'dec', 'zredmagic', 'lum_z']

    data = fits.getdata(filename)
    # header = fits.getheader(filename)

    # selecting a fraction
    zdiff = (z_max - z_min)/zbins
    idx = data["ra"] != data["ra"]

    for ii in range(zbins):
        iidx = (data["zredmagic"] > z_min+ii *
                zdiff) & (data["zredmagic"] <= z_min+(ii+1)*zdiff)
        lum_z = data["lum_z"][iidx]
        lumcut = np.percentile(lum_z, 100*(1.-frac))
        idx = idx | (iidx & (data["lum_z"] >= lumcut))

    data_dict = {}
    for key in lens_params:
        data_dict[key] = data[key][idx].byteswap().newbyteorder()
    data_dict["jacknife_idx"] = jacknife_idx[idx]

    data_dict["ra"] = data_dict["ra"]*np.pi/180.
    data_dict["dec"] = data_dict["dec"]*np.pi/180.

    return data_dict

# NOTE: function now accept the file objects as inputs (not global vars)


def reading_data_sources(shape_file_data, redshift_file_data, start, end):
    """
    start:
    end: 
    """
    # The parameters needed to computing differential surface density
    shape_params = [
        'coadd_object_id', 'ra', 'dec', 'e_1', 'e_2',
        'snr', 'weight', 'flags', 'size_ratio', 'T',
        'R11', 'R12', 'R21', 'R22'
    ]
    redshift_params = ['coadd_object_id', 'zmc_sof', 'zmean_sof']

    data1 = reading_shape_data(
        file_data=shape_file_data, params=shape_params,
        start=start, end=end
    )
    data2 = reading_shape_data(
        file_data=redshift_file_data, params=redshift_params,
        start=start, end=end
    )

    data1["zmc_sof"] = data2["zmc_sof"]
    data1["zmean_sof"] = data2["zmean_sof"]

    # Metacal selection
    data = data1
    flags = 0
    snr_th_lower = 10
    snr_th_upper = 1000
    size_ratio = 0.5
    T = 10

    idx = (
        np.array(data["flags"] == flags) &
        np.array(data["snr"] > snr_th_lower) &
        np.array(data["snr"] < snr_th_upper) &
        np.array(data["size_ratio"] > size_ratio) &
        np.array(data["T"] < T)
    )

    data_selected = dict()
    for key in list(data.keys()):
        data_selected[key] = data[key].values[idx]

    data_selected["ra"] = data_selected["ra"]*np.pi/180.
    data_selected["dec"] = data_selected["dec"]*np.pi/180.

    return data_selected


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
