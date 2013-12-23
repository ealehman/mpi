import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpi4py import MPI
import sys

from Plotter3DCS205 import MeshPlotter3D, MeshPlotter3DParallel
plt.ion()

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
	up[1:Ix,1:Iy] = ((2-4*DTDX)*u[1:Ix,1:Iy] - um[1:Ix,1:Iy]+ DTDX*(u[0:Ix-1,1:Iy] + u[2:Ix+1,1:Iy] +u[1:Ix,0:Iy-1] + u[1:Ix,2:Iy+1]))

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
	Ny = u.shape[1] - 2

	buf_x = np.empty(u.shape[1])
	buf_y = np.empty(u.shape[0])
	
	# Update ghost points with boundary condition
	if p_row == 0:
		u[0,:]    = u[2,:]      # u_{0,j}    = u_{2,j}      x = 0

	elif p_row == Py - 1:
		u[Nx+1,:] = u[Nx-1,:]    # u_{Nx+1,j} = u_{Nx-1,j}   x = 1
	else:
		if p_row%2 == 0:
			comm_col.Isend(u[1,:], dest = p_row - 1)
			req = comm_col.Irecv(buf_x, source = p_row - 1)


			status = MPI.Status()
			req.Wait(status)
			assert status.source == p_row -1
			u[0,:] = buf_x

		else:
			comm_col.Isend(u[Nx,:], dest = p_row + 1)
			req = comm_col.Irecv(buf_x, source = p_row + 1)


			status = MPI.Status()
			req.Wait(status)
			assert status.source == p_row +1
			u[Nx + 1,:] = buf_x

	if p_col == 0:
		u[:,0]    = u[:,2]       # u_{i,0}    = u_{i,2}      y = 0
	elif p_col == Px - 1: 
		u[:,Ny+1] = u[:,Ny-1]    # u_{i,Ny+1} = u_{i,Ny-1}   y = 1
	else:
		#sendrecv rightmost col to the leftmost(first) ghost column of p_col + 1
		if p_col%2 ==0:
			comm_row.Isend(np.ascontiguousarray(u[:, 1]), dest = p_col - 1)
			req = comm_row.Irecv(buf_y, source = p_col - 1)

			status = MPI.Status()
			req.Wait(status)
			
			assert status.source == p_col -1
			u[:,0] = buf_y

		else:
			comm_row.Isend(np.ascontiguousarray(u[:, Ny]), dest = p_col + 1)
			req = comm_row.Irecv(buf_y, source = p_col + 1)

			status = MPI.Status()
			req.Wait(status)
			
			assert status.source == p_col +1
			u[:,Ny + 1] = buf_y
			
	if p_row%2 == 0:
		comm_col.Isend(u[Nx,:], dest = p_row + 1)
		req = comm_col.Irecv(buf_x, source = p_row + 1)


		status = MPI.Status()
		req.Wait(status)
			
		assert status.source == p_row +1
		u[Nx + 1,:] = buf_x
			
	else:
		comm_col.Isend(u[1,:], dest = p_row - 1)
		req = comm_col.Irecv(buf_x, source = p_row - 1)

		status = MPI.Status()
		req.Wait(status)
			
		assert status.source == p_row - 1
		u[0,:] = buf_x

	if p_col%2 == 0:	
		comm_row.Isend(np.ascontiguousarray(u[:, Ny]), dest = p_col + 1)
		req = comm_row.Irecv(buf_y, source = p_col + 1)

		status = MPI.Status()
		req.Wait(status)
		
		assert status.source == p_col +1
		u[:,Ny + 1] = buf_y

	else:

		comm_row.Isend(np.ascontiguousarray(u[:, 1]), dest = p_col - 1)
		req = comm_row.Irecv(buf_y, source = p_col - 1)

		status = MPI.Status()
		req.Wait(status)
		
		assert status.source == p_col -1
		u[:,0] = buf_y
			
if __name__ == '__main__':

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	# Get Px and Py from command line
	try:
		Px = int(sys.argv[1])
		Py = int(sys.argv[2])
	except:
		print 'Usage: mpiexec -n (Px*Py) python Plotter3DCS205.py Px Py'
		sys.exit()

	# Sanity check
	assert Px*Py == MPI.COMM_WORLD.Get_size()
	
	# Create row and column communicators
	comm_col  = comm.Split(rank%Px)
	comm_row  = comm.Split(rank/Px)
	
	# Get the row and column indices for this process
	p_row     = comm_col.Get_rank()
	p_col     = comm_row.Get_rank()
	print p_row
	print p_col

	# Global constants
	xMin, xMax = 0.0, 1.0     # Domain boundaries
	yMin, yMax = 0.0, 1.0     # Domain boundaries
	Nx = 256                   # Number of total grid points in x
	Ny = Nx                   # Number of total grid points in y
	dx = (xMax-xMin)/(Nx-1)   # Grid spacing, Delta x
	dy = (yMax-yMin)/(Ny-1)   # Grid spacing, Delta y
	dt = 0.4 * dx             # Time step (Magic factor of 0.4)
	T = 5                     # Time end
	DTDX = (dt*dt) / (dx*dx)  # Precomputed CFL scalar

	# Local constants
	Nx_local = Nx/Px          # Number of local grid points in x
	Ny_local = Ny/Py          # Number of local grid points in y

	# The global indices: I[i,j] and J[i,j] are indices of u[i,j]
	[I,J] = np.mgrid[(Ny_local*p_row-1):(Ny_local*(p_row+1) + 1),(Nx_local*p_col- 1):(Nx_local*(p_col+1) +1)]

	# Plot data using parallel plotter -- Gather the data and create one plot
	plotter = MeshPlotter3DParallel()

	# Plot data using a serial plotter -- Create one plot for each process
	#plotter = MeshPlotter3D()


	up, u, um = initial_conditions(DTDX, I*dx -0.5, J*dy)

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
	
		# Print out the step and simulation time
		if rank == 0:
			print "Step: %d  Time: %f" % (k,t)
		# All processes draw the image. Comment when non-interactive.
		if k % 5 == 0:
			print k
			plotter.draw_now(I, J, u)


	# Save an image of the final data
	plotter.save_now(I, J, u, "OscillatorB.png")
