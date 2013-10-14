
from mpi4py import MPI
import numpy as np
import time
import math
import matplotlib.pyplot as plt
plt.ion() 

from P4_serial import *

def parallel_man(comm):

	rank = comm.Get_rank()
	size = comm.Get_size()

	sample_size = 6144
	image_size = 512
	minX,  maxX   = -2.1, 0.7
	minY,  maxY   = -1.25, 1.25
	width, height = 2**10, 2**10

	C = np.zeros([height/size,width], dtype=np.uint16)
	
	y = np.linspace(minY, maxY, height)
	#print "Line %d with y = %f" % (i, y)

	block = comm.scatter(np.array_split(y, size), root = 0)
	
	for row, y in enumerate(block):
		for column, x in enumerate(np.linspace(minX, maxX, width)):
			C[row, column] = mandelbrot(x,y)
			#print "added " + str(C[row, column]) +   " to row: " + str(row) + " column: " + str(column)
		

	#image = np.zeros([height,width], dtype=np.uint16)
	C = comm.gather(C, root = 0)
	final = np.zeros([height,width], dtype=np.uint16)
	for n, block  in enumerate(C):
		for m, row in enumerate(block):
			final[m + n*(1024/size)] = row
			print "finalized row: " + str(m + n*(1024/size))

	return final


if __name__ == '__main__':
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	

	#comm.barrier()
	p_start = MPI.Wtime()
	p_man = parallel_man(comm)
	#comm.barrier()
	p_stop = MPI.Wtime()

	# Allocate space for the tomographic image
	image_size = 512

	#result = parallel_trans(data, n_phi, comm)

	# Plot/Save the final result
	#if rank == 0:
	plt.imsave('Joe.png', p_man, cmap='spectral')
	plt.imshow(p_man, aspect='equal', cmap='spectral')

	plt.show()

