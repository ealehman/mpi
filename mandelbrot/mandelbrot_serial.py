import numpy as np
import matplotlib.pyplot as plt
import time

def mandelbrot(x, y):
  '''Compute a Mandelbrot pixel -- Unoptimized'''
  z = c = complex(x,y)
  it, maxit = 0, 511
  while abs(z) < 2 and it < maxit:
    z = z*z + c
    it += 1
  return it

# Global variables, can be used by any process
minX,  maxX   = -2.1, 0.7
minY,  maxY   = -1.25, 1.25
width, height = 2**10, 2**10

if __name__ == '__main__':
  C = np.zeros([height,width], dtype=np.uint16)

  start_time = time.time()
  for i,y in enumerate(np.linspace(minY, maxY, height)):
    print "Line %d with y = %f" % (i, y)
    for j,x in enumerate(np.linspace(minX, maxX, width)):
      C[i,j] = mandelbrot(x,y)
  end_time = time.time()

  print "Time: %f secs" % (end_time - start_time)
  plt.imsave('Mandelbrot.png', C, cmap='spectral')
  plt.imshow(C, aspect='equal', cmap='spectral')
  plt.show()
