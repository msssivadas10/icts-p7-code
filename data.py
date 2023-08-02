#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from astropy.io import fits as pyfits
from tqdm import tqdm
import pandas

'''
Get the jackknife region for each galaxy by reading the jackknife regions file
'''
def getregions(regfile,ra,dec):
    ra = (ra+90.)%360.0
    jackreg = np.zeros(ra.size, dtype=int)
    # Read the file
    listramin,listramax,listdecmin,listdecmax=np.loadtxt(regfile,unpack=1)
    for i in tqdm(range(ra.size)):
        for j in range(listramin.size):
            if(ra[i]<listramin[j]):
                continue
            if(ra[i]>=listramax[j]):
                continue
            if(dec[i]<listdecmin[j]):
                continue
            if(dec[i]>=listdecmax[j]):
                continue
            jackreg[i]=j
            break
    return jackreg

if __name__ == "__main__":
    # Now let us get all the jackknife regions
    regfile = "DES-regions.list"

    hdulist = pyfits.open("y3a2_gold2.2.1_redmagic_highdens.fits")
    data = hdulist[1].data
    ra = data["ra"]
    dec = data["dec"]

    jackreg = getregions(regfile, ra, dec)
    d = {}
    d["jack_idx"] = jackreg
    pandas.DataFrame(data=d).to_csv("Jackknife.dat")


# In[ ]:




