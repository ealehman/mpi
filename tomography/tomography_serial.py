import numpy as np
import matplotlib.pyplot as plt
import math
plt.ion()        # Allow interactive updates to the plots
from mpi4py import MPI

class data_transformer:
  '''A class to transform a line of attenuated data into a back-projected image.
  Construct on the number of data points in a line of data and the number of
  pixels in the resulting square image. This precomputes the
  back-projection operator.
  Once constructed, call the transform method on a line of attenuated data and
  the angle that data represents to retrieve the back-projected image.'''
  def __init__(self, sample_size, image_size):
    '''Perform the required precomputation for the back-projection step.'''
    [self.X,self.Y] = np.meshgrid(np.linspace(-1,1,image_size),
                                  np.linspace(-1,1,image_size))
    self.proj_domain = np.linspace(-1,1,sample_size)
    self.f_scale = abs(np.fft.fftshift(np.linspace(-1,1,sample_size+1)[0:-1]))

  def transform(self, data, phi):
    '''Transform a data line taken at an angle phi to its back-projected image.
    Input: data, an array of sample_size values.
    Output: an image_size x image_size array -- the back-projected image'''
    # Compute the Fourier filtered data
    filtered_data = np.fft.ifft(np.fft.fft(data) * self.f_scale).real
    # Interpolate the data to the rotated image domain
    result = np.interp(self.X*np.cos(phi) + self.Y*np.sin(phi),
                       self.proj_domain, filtered_data)
    return result


if __name__ == '__main__':
  # Metadata
  n_phi       = 2048   # The number of Tomographic projections
  sample_size = 6144   # The number of samples in each projection

  # Read the projective data from file
  data = np.fromfile(file='TomoData.bin', dtype=np.float64)
  data = data.reshape(n_phi, sample_size)
  
  """

  # Plot the raw data
  plt.figure(1);
  plt.imshow(data, cmap='bone');
  """

  # Allocate space for the tomographic image
  image_size = 2048

  start = MPI.Wtime()
  result = np.zeros((image_size,image_size), dtype=np.float64)

  # Precompute a data_transformer
  Transformer = data_transformer(sample_size, image_size)

  # For each row of the data
  for k in xrange(0,n_phi):
    # Compute the angle of this slice
    phi = -k * math.pi / n_phi
    # Accumulate the back-projection
    result += Transformer.transform(data[k,:], phi)

  stop = MPI.Wtime()
  s_time = stop - start

  print 'image size ' + str(image_size) + ' serial time: ' + str(s_time) + ' seconds'
  
  """
    # Update a plot every so often to show progress
    print k, phi
    if k % 50 == 0:
      plt.figure(2)
      plt.imshow(result, cmap='bone')
      plt.draw()
  """
  # Plot/Save the final result
  #plt.figure(2)
  #plt.imshow(result, cmap='bone')
  #plt.draw()
  plt.imsave('TomographicReconstruction.png', result, cmap='bone')
  #raw_input("Any key to exit...")
