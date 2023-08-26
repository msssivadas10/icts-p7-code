#!/usr/bin/env python
# coding: utf-8

import numpy as np
from astropy.io import fits as pyfits
# from pylab import *
from pylab import plt


def makeregions(ra, dec, rabin, decbin, buffer=1):
    # Sort by ra and then dec
    idx = np.lexsort((dec, ra))
    ra = ra[idx]
    dec = dec[idx]
    listramin = np.array([], dtype='float64')
    listramax = np.array([], dtype='float64')
    listdecmin = np.array([], dtype='float64')
    listdecmax = np.array([], dtype='float64')

    for i in np.arange(rabin):
        ramin = ra[int(ra.size*(i*1.0)/rabin)]
        if (i == 0 and buffer):
            ramin = ramin-1.0e-3
        if (i == rabin-1):
            ramax = ra[-1]
        else:
            ramax = ra[int(ra.size*((i+1)*1.0)/rabin)]

        idx = (ra > ramin) & (ra < ramax)
        xra = ra[idx]
        xdec = dec[idx]

        idx = np.lexsort((xra, xdec))
        xra = xra[idx]
        xdec = xdec[idx]

        for j in np.arange(decbin):
            # Now sort into declination ranges
            decmin = xdec[int(xdec.size*(j*1.0)/decbin)]
            if (j == 0 and buffer):
                decmin = decmin-1.0E-3
            if (j == decbin-1):
                decmax = xdec[-1]
            else:
                decmax = xdec[int(xdec.size*((j+1)*1.0)/decbin)]
            listramin = np.append(listramin, ramin)
            listramax = np.append(listramax, ramax)
            listdecmin = np.append(listdecmin, decmin)
            listdecmax = np.append(listdecmax, decmax)
    return listramin, listramax, listdecmin, listdecmax


divide = 1
if (divide):
    hdulist = pyfits.open("y3a2_gold2.2.1_redmagic_highdens_randoms.fits")
    data = hdulist[1].data
    rra = data.field("ra")
    rdec = data.field("dec")

    # Essential change here for contiguous south region
    rra = (rra+(90.)) % (360.)

    # Divide into 3 ra bins and 4 dec bins for each, fiducial
    rabin = 10
    decbin = 10
    listramin, listramax, listdecmin, listdecmax = makeregions(
        rra, rdec, rabin, decbin)
    np.savetxt("DES-regions.list",
               np.transpose([listramin, listramax, listdecmin, listdecmax]))
'''
    hdulist=pyfits.open("y3a2_gold2.2.1_redmagic_highdens_randoms.fits");
    data=hdulist[1].data
    rra=data.field("ra");
    rdec=data.field("dec");

    # Essential change here for contiguous south region
    rra=(rra+(60.))%(360.);

    # Divide into 6 ra bins and 6 dec bins for each
    rabin=12
    decbin=12
    listramin,listramax,listdecmin,listdecmax=makeregions(rra,rdec,rabin,decbin);
    np.savetxt("N-regions.list",np.transpose([listramin,listramax,listdecmin,listdecmax]));
'''


def _test():
    # Let us plot each of the section
    sec = np.array(["DES"])
    ax = plt.subplot(1, 1, 1)
    ax.set_xlim([0, 360])
    ax.set_ylim([-90, 90])
    for ss in np.arange(sec.size):
        # ax=subplot(2,2,ss+1);
        listramin, listramax, listdecmin, listdecmax = np.loadtxt(
            sec[ss]+"-regions.list", unpack=1)
        for ii in range(listramin.size):
            ax.plot([listramin[ii], listramin[ii]], [
                    listdecmin[ii], listdecmax[ii]], color='r')
            ax.plot([listramin[ii], listramax[ii]], [
                    listdecmin[ii], listdecmin[ii]], color='r')
            ax.plot([listramin[ii], listramax[ii]], [
                    listdecmax[ii], listdecmax[ii]], color='r')
            ax.plot([listramax[ii], listramax[ii]], [
                    listdecmin[ii], listdecmax[ii]], color='r')

    plt.tight_layout()
    plt.savefig("Figure_chunk.pdf")


if __name__ == '__main__':
    _test()
