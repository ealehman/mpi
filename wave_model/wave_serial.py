import numpy as np
#from Plotter3DCS205 import MeshPlotter3D, MeshPlotter3DParallel
import time

def initial_conditions(DTDX, X, Y):
  '''Construct the grid points and set the initial conditions.
  X[i,j] and Y[i,j] are the 2D coordinates of u[i,j]'''
  assert X.shape == Y.shape

  um = np.zeros(X.shape)     # u^{n-1}  "u minus"
  u  = np.zeros(X.shape)     # u^{n}    "u"
  up = np.zeros(X.shape)     # u^{n+1}  "u plus"
  # Define Ix and Iy so that 1:Ix and 1:Iy define the interior points
  Ix = u.shape[0] - 1
  Iy = u.shape[1] - 1
  # Set the interior points: Initial condition is Gaussian
  u[1:Ix,1:Iy] = np.exp(-50 * (X[1:Ix,1:Iy]**2 + Y[1:Ix,1:Iy]**2))
  # Set the ghost points to the boundary conditions
  set_ghost_points(u)
  # Set the initial time derivative to zero by running backwards
  apply_stencil(DTDX, um, u, up)
  um *= 0.5
  # Done initializing up, u, and um
  return up, u, um

def apply_stencil(DTDX, up, u, um):
  '''Apply the computational stencil to compute u^{n+1} -- "up".
  Assumes the ghost points exist and are set to the correct values.'''

  # Define Ix and Iy so that 1:Ix and 1:Iy define the interior points
  Ix = u.shape[0] - 1
  Iy = u.shape[1] - 1
  # Update interior grid points with vectorized stencil
  up[1:Ix,1:Iy] = ((2-4*DTDX)*u[1:Ix,1:Iy] - um[1:Ix,1:Iy]
                   + DTDX*(u[0:Ix-1,1:Iy  ] +
                           u[2:Ix+1,1:Iy  ] +
                           u[1:Ix  ,0:Iy-1] +
                           u[1:Ix  ,2:Iy+1]))

  # The above is a vectorized operation for the simple for-loops:
  #for i in range(1,Ix):
  #  for j in range(1,Iy):
  #    up[i,j] = ((2-4*DTDX)*u[i,j] - um[i,j]
  #               + DTDX*(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]))

def set_ghost_points(u):
  '''Set the ghost points.
  In serial, the only ghost points are the boundaries.
  In parallel, each process will have ghost points:
      some will need data from neighboring processes,
      others will use these boundary conditions.'''
  # Define Nx and Ny so that Nx+1 and Ny+1 are the ghost points
  Nx = u.shape[0] - 2
  #print 'Nx: ' + str(Nx)

  Ny = u.shape[1] - 2
  #print 'Ny: ' + str(Ny)
  # Update ghost points with boundary condition
  u[0,:]    = u[2,:];       # u_{0,j}    = u_{2,j}      x = 0
  u[Nx+1,:] = u[Nx-1,:];    # u_{Nx+1,j} = u_{Nx-1,j}   x = 1
  u[:,0]    = u[:,2];       # u_{i,0}    = u_{i,2}      y = 0
  u[:,Ny+1] = u[:,Ny-1];    # u_{i,Ny+1} = u_{i,Ny-1}   y = 1


if __name__ == '__main__':
  start = time.time()
  # Global constants
  xMin, xMax = 0.0, 1.0     # Domain boundaries
  yMin, yMax = 0.0, 1.0     # Domain boundaries
  Nx = 1024                   # Number of total grid points in x
  Ny = Nx                   # Number of total grid points in y
  dx = (xMax-xMin)/(Nx-1)   # Grid spacing, Delta x
  dy = (yMax-yMin)/(Ny-1)   # Grid spacing, Delta y
  dt = 0.4 * dx             # Time step (Magic factor of 0.4)
  T = 5                     # Time end
  DTDX = (dt*dt) / (dx*dx)  # Precomputed CFL scalar

  # The global indices: I[i,j] and J[i,j] are indices of u[i,j]
  [I,J] = np.mgrid[-1:Nx+1, -1:Ny+1]
  # Convenience so u[1:Ix,1:Iy] are all interior points
  Ix, Iy = Nx+1, Ny+1

  # Set the initial conditions
  up, u, um = initial_conditions(DTDX, I*dx-0.5, J*dy)

  #print 'DTDX: ' + str(DTDX) + 'up: ' + str(up) + 'u: ' + str(u) + 'um: ' + str(um)
  # Setup the serial plotter -- one plot per process
  #plotter = MeshPlotter3D()
  # Setup the parallel plotter -- one plot gathered from all processes
  #plotter = MeshPlotter3DParallel()

  for k,t in enumerate(np.arange(0,T,dt)):
    # Compute u^{n+1} with the computational stencil
    apply_stencil(DTDX, up, u, um)

    # Set the ghost points on u^{n+1}
    set_ghost_points(up)

    # Swap references for the next step
    # u^{n-1} <- u^{n}
    # u^{n}   <- u^{n+1}
    # u^{n+1} <- u^{n-1} to be overwritten in next step
    um, u, up = u, up, um
    # print 'DTDX: ' + str(DTDX) + 'up: ' + str(up) + 'u: ' + str(u) + 'um: ' + str(um)
    # Output and draw Occasionally
    #print "Step: %d  Time: %f" % (k,t)
    #if k % 5 == 0:
      #plotter.draw_now(I[1:Ix,1:Iy], J[1:Ix,1:Iy], u[1:Ix,1:Iy])
  stop = time.time()
  duration = stop - start
  print duration
  #plotter.save_now(I[1:Ix,1:Iy], J[1:Ix,1:Iy], u[1:Ix,1:Iy], "FinalWave.png")
