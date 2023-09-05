from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()  # get number of processes
rank = comm.Get_rank()  # get the process rank

print(rank)
