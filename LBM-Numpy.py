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


import numpy as np
import matplotlib.pyplot as plt
import os
import utils

plot_every = 100 # iterations

def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def main():
    Nx, Ny = 400, 100 # Grid resolution
    tau = 0.53 # Kinnematic viscosity or time scale
    Nt = 6000
    plotRealTime = True
    saveImages = True
    
    # Delete previous images before saving new ones
    if saveImages:
        path_figures = os.path.join(os.getcwd(), "figures") 
        os.makedirs(path_figures, exist_ok=True)
        utils.clear_folder_contents(path_figures)
    
    # Lattice speeds and weights
    NL = 9 # D2Q9 (9 lattice directions)
    cxs = np.array([0, 0, 1, 1,  1,  0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1,  0,  1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36]) # sums to 1
    
    # Initial Conditions (ICs)
    F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.3 # We assume there is a flow in the +x-direction (direction=3)
    
    cylinder = np.full((Ny, Nx), False)
    
    for y in range(0, Ny):
        for x in range(0, Nx):
            if distance(Nx//4, Ny//4, x, y) < 8: # radius of the cylinder is 13
                cylinder[y, x] = True
            elif distance(Nx//4, 2*Ny//4, x, y) < 8:
                cylinder[y, x] = True
            elif distance(Nx//4, 3*Ny//4, x, y) < 8:
                cylinder[y, x] = True
                
    # Main time loop
    for t in range(Nt+1):
        print(t)
        
        # Apply BCs at left and right walls to absorb waves
        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]
        
        # Streaming step
        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)
            
        # Apply Reflective BCs
        bcF = F[cylinder, :]
        bdF = bcF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]] # Reflect (opposite directions)
        
        # FLuid variables
        rho = np.sum(F, 2) # density
        ux = np.sum(F * cxs, 2) / rho # momentum
        uy = np.sum(F * cys, 2) / rho
        
        # Apply BCs?
        F[cylinder, :] = bdF
        ux[cylinder] = 0
        uy[cylinder] = 0
        
        # Collision step
        Feq = np.zeros_like(F)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:, :, i] = rho * w * (
                1 + 3*(cx*ux + cy*uy) + 9*(cx*ux + cy*uy)**2/2 - 3*(ux**2 + uy**2)/2)        
        
        F += -(1/tau) * (F-Feq)
        
        if (plotRealTime and (t % plot_every == 0)):
            
            # Field variable: Velocity
            velocity = np.sqrt(ux**2+uy**2)
            
            # Field variable: Curl of Velocity vector
            # To plot regions of high and low vorticity
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
            dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfydx - dfxdy
            
            # Plot Field variable
            # if shape of BC-boolean map (e.g., cylinder) does not match field shape,
            # we need to trim the BC-boolean map
            field = curl
            if field.shape != cylinder.shape:
                cylinder_trimmed = cylinder[1:-1, 1:-1]
            else:
                cylinder_trimmed = cylinder
                
            # Create a masked array
            masked_field = np.ma.masked_array(field, mask=cylinder_trimmed)
            
            # Create a custom colormap
            cmap = plt.get_cmap('bwr').copy()
            cmap.set_bad('black', 1.0)
            
            plt.imshow(masked_field, cmap=cmap, interpolation='nearest') 
            # Note on colormap: blue for negative, red for positive values
            # plt.colorbar()
            plt.title(label=f"Vorticity field - Timestep={t}", fontsize=8)
            
            if saveImages:
                plt.savefig(f'{path_figures}/curl_LBM_numpy_timestep_{t:04d}.png', dpi=120)
            
            plt.pause(0.01)
            plt.cla()
        
    if saveImages:
        utils.create_gif_with_PIL(path_figures, "videos/Vorticity.gif")
        
    
if __name__ == "__main__":
    main()































