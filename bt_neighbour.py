# import libraries needed
import numpy as np
from sklearn.neighbors import BallTree
# from sklearn.metrics.pairwise import haversine_distances # no need!
import astropy.cosmology as ac
from typing import Any 


def get_neighbour_indices(z: float, cat: Any, cm: ac.Cosmology, center_point: Any, b: float = 10.) -> Any:
    r"""
    Searach for the nearest neighbours of a query point within some angular radius, 
    given a galaxy catalog, redshift and cosmology model. 

    Parameters
    ----------
    z: float
        Value of the reshift: must be a float greater than -1.
    cat: array_like
        Galaxy catalog. This should only have two features: `dec` and `ra` in that 
        order, given in radians, represnting the galaxy position on the sky sphere 
        corresponding to the redshift.
    cm: astropy.cosmology.Cosmology
        A cosmology model object.
    center_point: array_like
        Point whose nearest neighbours are to be found.

    Returns
    -------
    nnids: array_like
        Indices of the nearest neighbours on the catalog. This will be an array of 
        arrays, each corresponds to neighbours of each center point.

    Example
    -------
    TODO


    """

    # Checks:
    # NOTE: for compatibility with sklearn's `query_radius` function, which accepts 
    # multiple center points, argument `center_point` must be a 2d array, allowing 
    # multiple queries.
    assert isinstance(z, (int, float)), 'redshift must be a number'
    assert z + 1. > 0., 'redshift must be greater than -1' 
    assert np.ndim( cat ) == 2, 'catalog must be a 2D array'
    assert np.size( cat, 1 ) == 2, 'catalog must have exactly 2 freatures'
    assert isinstance( cm, ac.Cosmology ), '`cm` must be a astropy `Cosmology` object' 
    assert np.ndim( center_point ) == 2, '`center_point` must be a 2D array'
    assert np.size( center_point, 1 ) == 2, '`center_point` must have exactly 2 freatures'

    # calculate the value of the comoving distance corresponding to `z` in Mpc
    # astropy returns value as `Quantity` object and extract the value from it!
    x = cm.comoving_distance( z ).value

    # compute the angular distance in radian, corresponding to `x` radius
    theta = b / x

    # build the ball tree using the given angular coordinates. the first feature 
    # should be the `dec` and the other is `ra`, both in radians
    # 
    # NOTE: using the conversion ra == tongitude and dec == latitude
    bt = BallTree( cat, leaf_size = 2, metric = 'haversine' )

    # search for nearest neighbors in `theta` distance from the center point
    nnids = bt.query_radius( center_point, theta )

    return nnids


# test:
def _test():

    # using a redshift closer to 0 to get a large angle
    z = 0.005

    # using 1000 random points on the sphere as galaxies
    cat = np.random.uniform( [-np.pi/2., 0.], [np.pi/2., 2*np.pi], size = [100000, 2] ) 

    # using a flat lcdm model cosmology
    cm  = ac.FlatLambdaCDM(H0 = 70.0, Om0 = 0.3, Tcmb0 = 2.725, Ob0 = 0.05)

    # find the neighbours around random point
    p = np.random.uniform( [-np.pi/2., 0.], [np.pi/2., 2*np.pi], size = [2, 2] )
    j = get_neighbour_indices( z = z, cat = cat, cm = cm, center_point = p )

    
    # visualize to check
    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot( cat[:,1], cat[:,0], 'o', ms = 3, color = 'blue' )  # plot the catalog

    n_points = len( p )
    for i in range( n_points ):
        val = (i + 1) / (n_points+1)
        plt.plot( p[i,1],      p[i,0],      'o', ms = 5, color = (val, 0., 0.) ) # test points
        plt.plot( cat[j[i],1], cat[j[i],0], 'o', ms = 5, color = (0., val, 0.) ) # neighbors

    plt.show()

    return

if __name__ == '__main__':
    _test()