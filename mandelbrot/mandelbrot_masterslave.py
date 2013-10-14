import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from P4_serial import *

def slave(comm):
    jobs = True
    while jobs:
        status = MPI.Status()
        y = comm.recv(source=0, tag = MPI.ANY_TAG, status = status) #, status=status)
        
        tag = status.Get_tag()
        if tag >= 1024:
            jobs = False
        
        print 'slave received row: ' + str(tag)
    
        #x = np.linspace(minX, maxX, width)
        temp = []
        for x in np.linspace(minX, maxX, width):
            temp.append(mandelbrot(x,y))


        comm.send(temp, dest = 0, tag = tag)
        print 'slave sent row: ' + str(tag) #+ " values: " + str(temp)
    return

def master(comm):
    image = np.zeros([height,width], dtype=np.uint16)
    count = 0
    size = comm.Get_size()

    #send as many rows as possible to processes 
    y = np.linspace(minY, maxY, height)
    for p in range(1,size):
        comm.send(y[count], dest = p, tag = count)
        print 'master sent row: ' + str(count)
        count += 1

    
    #send next row to available slaves until there are no more rows left
    while count < 1024:
        
        status = MPI.Status()
        
        t = comm.recv(source= MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status) #, status=status)
        n = status.Get_tag()
        print 'master received row: ' + str(n)
        source = status.Get_source()
        
        image[n] = t
        
        comm.send(y[count], dest = source, tag = count)
        print 'master sent row: ' + str(count)
        count += 1

    return image


if __name__ == '__main__':
    # Get MPI data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


    if rank == 0:
        start_time = MPI.Wtime()
        C = master(comm)
        end_time = MPI.Wtime()
        print "Time: %f secs" % (end_time - start_time)
        plt.imsave('Mandelbrot.png', C, cmap='spectral')
        plt.imshow(C, aspect='equal', cmap='spectral')
        plt.show()
    else:
        slave(comm)
