from mpi4py import MPI
import numpy as np
import time


def get_big_arrays():
  '''Generate two big random arrays.'''
  N = 12582912       # A big number, the size of the arrays.
  np.random.seed(0)  # Set the random seed
  return np.random.random(N), np.random.random(N)


def serial_dot(a, b):
  '''The dot-product of the arrays -- slow implementation using for-loop.'''
  result = 0
  for k in xrange(0, len(a)):
    result += a[k]*b[k]
  return result


if __name__ == '__main__':
  #Get big arrays
  a, b = get_big_arrays()

  # Compute the dot product in serial
  start_time = time.time()
  result = serial_dot(a, b)
  end_time = time.time()

  print "a*b = %f in %f seconds" % (result, end_time - start_time)
