from scipy.spatial import KDTree
import numpy as np

def get_xxyyzz(data):
    dec, ra = data.T
    ra = ra*np.pi/180.
    dec = dec*np.pi/180.
    xx = np.cos(dec) * np.cos(ra)
    yy = np.cos(dec) * np.sin(ra)
    zz = np.sin(dec)

    return xx, yy, zz

class BallTree:

    def __init__(self, data=None):
        self.xx, self.yy, self.zz = get_xxyyzz(data)

        self.tree = KDTree(np.array([self.xx, self.yy, self.zz]).T)


    def query_radius(self, data_src, theta_max, return_distance):
    
        xx, yy, zz = get_xxyyzz(data_src)
        idx = self.tree.query_ball_point(np.array([xx, yy, zz]).T, r=theta_max)
        dist = idx * 1
        if return_distance:
            for ii in range(idx.shape[0]):
                # Get_distances
                dist[ii] = np.arccos(max(0, min(xx[ii]*xx[idx[ii]] + yy[ii]*yy[idx[ii]] + zz[ii]*zz[idx[ii]], 1)))

            return idx, dist

        return idx
