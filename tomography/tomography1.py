from mpi4py import MPI
import numpy as np
import time
import math
import matplotlib.pyplot as plt
plt.ion() 

from P3_serial import data_transformer

def parallel_trans(data, n_phi, comm, p_root= 0):

 	rank = comm.Get_rank()
  	size = comm.Get_size()

  	#send rows to be transformed from the root to the different processes
 	if rank == 0:
		split_data = np.array_split(data,size)
		for p in range(1, size):
			comm.send(split_data[p], dest = p)
			#print str(rank) + 'sending'
		#comm.barrier()
	else: 
		#print str(rank) + ' receiving'
		data = comm.recv(source=0)


	
	sample_size = 6144
	image_size = 512
	
	result = np.zeros((image_size,image_size), dtype=np.float64)
	
	Transformer = data_transformer(sample_size, image_size)

 	for k in xrange(0,n_phi/size):
		# Compute the angle of this slice
		phi = -(k +  rank*n_phi/size) * math.pi / n_phi
		# Accumulate the back-projection
		result += Transformer.transform(data[k,:], phi)
	
	#send transformed data back to root
	if rank == 0:
		for p in range(1,size):
			temp = comm.recv(source = p)
			result += temp
	else:
		comm.send(result, dest = 0)


	#final = comm.reduce(result, root = 0)

 	return result


if __name__ == '__main__':
	comm = MPI.COMM_WORLD
  	rank = comm.Get_rank()
	
	n_phi       = 2048   # The number of Tomographic projections
	sample_size = 6144   # The number of samples in each projection
 	

	data = np.fromfile(file='TomoData.bin', dtype=np.float64)
	data = data.reshape(n_phi, sample_size)

	comm.barrier()
	p_start = MPI.Wtime()
	p_dot = parallel_trans(data, n_phi, comm)
	comm.barrier()
	p_stop = MPI.Wtime()

	# Allocate space for the tomographic image
	image_size = 512

	result = parallel_trans(data, n_phi, comm)

	if rank == 0:
		#plt.imsave('TomographicReconstruction.png', result, cmap='bone')
		plt.imsave('P3a.png', result, cmap='bone')
	#raw_input("Any key to exit...")

