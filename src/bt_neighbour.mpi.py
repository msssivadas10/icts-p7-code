# import numpy as np
# import astropy.cosmology as ac
from mpi4py import MPI
# from sklearn.neighbors import BallTree
# from typing import Any

comm = MPI.COMM_WORLD
size = comm.Get_size()  # get number of processes
rank = comm.Get_rank()  # get the process rank

print(rank)
