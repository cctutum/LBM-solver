# LBM-solver using Numpy
#======================== 

# IMport libraries
import matplotlib.pyplot as plt
import numpy as np

# Simulations Parameters and Constants
Nx = 400 # resolution x-dir
Ny = 100 # resolution y-dir
rho0 = 100 # average density
tau = 0.6 # collision timescale
Nt = 100 # number of timesteps

# Lattice speeds / weights
NL = 9 # D2Q9 (9 lattice directions)
idxs = np.arange(NL) # Lattice indices
cxs = np.array([0, 0, 1, 1,  1,  0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1,  0,  1])
weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36]) # sums to 1

# Initial Conditions - flow to the right with some perturbations
np.random.seed(42)
F = np.ones((Ny, Nx, NL)) # fluid distribution function
F += 0.01 * np.random.randn(Ny, Nx, NL) # add small noise
X, Y = np.meshgrid(range(Nx), range(Ny))
F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4)) # flow perturbation in +x direction
rho = np.sum(F, 2) # F.shape=(100,400,9) --> rho.shape(100,400)
for i in idxs:
    F[:, :, i] *= rho0 / rho
    
# Cylinder boundary
# X, Y = np.meshgrid(range(Nx), range(Ny))
cylinder = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/4)**2

# Prep figure
fig = plt.figure(figsize=(4,2), dpi=80)

# Simulation Main Loop
# for it in range(Nt):
    
        

