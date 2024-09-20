# LBM-solver using Numpy
#========================

"""
The Lattice Boltzmann Method (LBM) is a computational fluid dynamics (CFD) 
technique used to simulate fluid flows. It operates on a mesoscopic scale, 
modeling the fluid as a collection of particles distributed on a lattice. 
These particles undergo streaming and collision processes, which approximate 
the macroscopic behavior of the fluid without directly solving the 
Navier-Stokes equations.

During each time step, particles stream along their velocity directions to 
neighboring nodes, followed by a collision step that relaxes the distribution 
function towards equilibrium.

Explanation
Initialization: The distribution functions are initialized uniformly with 
weights corresponding to the D2Q9 model.
Equilibrium Function: Computes the equilibrium distribution based on local 
density and velocity.
Collision Step: Adjusts the distribution functions towards equilibrium using 
the relaxation time τ.
Streaming Step: Shifts the distribution functions according to their velocities.
""" 

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import utils

# Simulations Parameters and Constants
Nx = 400 # resolution x-dir
Ny = 100 # resolution y-dir
rho0 = 100 # average density
tau = 0.6 # collision timescale
Nt = 100 # number of timesteps
plotRealTime = True
saveImages = True

if saveImages:
    # Delete previous images before saving new ones
    path_figures = os.path.join(os.getcwd(), "figures") 
    utils.clear_folder_contents(path_figures)
        

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
for it in range(Nt):
    # print time step
    print(it)
    
    # Drift (Stream?)
    for i, cx, cy in zip(idxs, cxs, cys):
        F[:,:,i] = np.roll( F[:,:,i], cx, axis=1)
        F[:,:,i] = np.roll( F[:,:,i], cy, axis=0)
        
    # Set reflective boundaries
    bndryF = F[cylinder,:]
    bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]
    
    # Calculate fluid variables
    rho = np.sum(F, 2)
    ux  = np.sum(F * cxs, 2) / rho
    uy  = np.sum(F * cys, 2) / rho
    
    # Apply Collision
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux + cy*uy) + 
                                9 * (cx*ux + cy*uy)**2 / 2 - 
                                3 * (ux**2 + uy**2) / 2 )
        
    F += -(1.0/tau) * (F - Feq)
		
    # Apply boundary 
    F[cylinder,:] = bndryF
		
    # plot in real time - color 1/2 particles blue, other half red
    if (plotRealTime and (it % 10) == 0) or (it == Nt-1):
        plt.cla()
        plt.title(label=f"Vorticity - Timestep={it}", fontsize=8)
        ux[cylinder] = 0
        uy[cylinder] = 0
        vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        vorticity[cylinder] = np.nan
        vorticity = np.ma.array(vorticity, mask=cylinder)
        plt.imshow(vorticity, cmap='bwr')
        plt.imshow(~cylinder, cmap='gray', alpha=0.3)
        plt.clim(-.1, .1)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)	
        ax.set_aspect('equal')	
        if saveImages:
            plt.savefig(f'{path_figures}/LBM_numpy_timestep_{it:04d}.png', dpi=120)
        plt.pause(0.001)
    
# Save figure
plt.savefig(f'LBM_numpy_timestep_{it:04d}.png', dpi=120)
plt.show()

