
from mpi4py import MPI
import numpy as np
import time
import math
import matplotlib.pyplot as plt
plt.ion() 

from P3_serial import data_transformer

def parallel_trans(data, n_phi, comm, image_size, p_root= 0):

 	rank = comm.Get_rank()
  	size = comm.Get_size()

	data = comm.scatter(np.array_split(data,size), root = 0)


	sample_size = 6144
	#image_size = 512
	
	result = np.zeros((image_size,image_size), dtype=np.float64)
	
	Transformer = data_transformer(sample_size, image_size)

 	for k in xrange(0,n_phi/size):
		# Compute the angle of this slice
		phi = -(k +  rank*n_phi/size) * math.pi / n_phi
		# Accumulate the back-projection
		result += Transformer.transform(data[k,:], phi)

	final = comm.reduce(result, root = 0)

 	return final


if __name__ == '__main__':
	comm = MPI.COMM_WORLD
  	rank = comm.Get_rank()
  	size = comm.Get_size()
	
	n_phi       = 2048   # The number of Tomographic projections
	sample_size = 6144   # The number of samples in each projection
 	image_size = 2048

	data = np.fromfile(file='TomoData.bin', dtype=np.float64)
	data = data.reshape(n_phi, sample_size)

	comm.barrier()
	p_start = MPI.Wtime()
	p_dot = parallel_trans(data, n_phi, comm, image_size)
	comm.barrier()
	p_stop = MPI.Wtime()

	p_time = p_stop - p_start

	print 'image size: ' + str(image_size)
	print 'parallel time: ' + str(p_time)
	print '# of processors: ' + str(size)

	# Allocate space for the tomographic image
	

	#result = parallel_trans(data, n_phi, comm)

	# Plot/Save the final result
	if rank == 0:
		plt.imsave('P3b2.png', p_dot, cmap='bone')

