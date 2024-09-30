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


def equilibrium(F, cxs, cys, NL, weights, rho, ux, uy):
    Feq = np.zeros_like(F)
    for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
        Feq[:, :, i] = rho * w * (
            1 + 3*(cx*ux + cy*uy) + 9*(cx*ux + cy*uy)**2/2 - 3*(ux**2 + uy**2)/2)  
    return Feq


def compute_field(field_var, ux, uy, boundary):
    if field_var == "Velocity":
        # Field variable: Velocity
        field = np.sqrt(ux**2+uy**2)
        # boundary array stays same
    elif field_var == "Vorticity":
        # Field variable: Vorticity (Curl of Velocity vector)
        # TODO: Check the implementation!!
        # field = calculate_vorticity_withManualGrad(ux, uy, 1, 1)
        # boundary = boundary[1:-1, 1:-1] # boundary array is trimmed
        field = calculate_vorticity(ux, uy, 1, 1)
    return field, boundary


def calculate_vorticity_withManualGrad(velocity_x, velocity_y, dx, dy):
    # dx, dy = 1, 1 
    # TODO: Implement more general grid
    dvy_dx = velocity_x[2:, 1:-1] - velocity_x[:-2, 1:-1]
    dvx_dy = velocity_y[1:-1, 2:] - velocity_y[1:-1, :-2]
    return dvy_dx - dvx_dy


def calculate_vorticity(velocity_x, velocity_y, dx, dy):
    # This implementation looks more elegant!
    dvx_dy = np.gradient(velocity_x, dy, axis=0)
    dvy_dx = np.gradient(velocity_y, dx, axis=1)
    return dvy_dx - dvx_dy


def plot_field(field_var, ux, uy, boundary, t, saveImages, path_figures):
    field, cylinder_trimmed = compute_field(field_var, ux, uy, boundary)
        
    # Create a masked array
    masked_field = np.ma.masked_array(field, mask=cylinder_trimmed)
    
    # Create a custom colormap
    cmap = plt.get_cmap('bwr').copy()
    cmap.set_bad('black', 1.0)
    
    plt.imshow(masked_field, cmap=cmap, interpolation='nearest') 
    # Note on colormap: blue for negative, red for positive values
    # plt.colorbar()
    if field_var == "Velocity":
        label = "Velocity field"
    elif field_var == "Vorticity":
        label = "Vorticity field"
    plt.title(label=f"{label} - Timestep={t}", fontsize=8)
    
    if saveImages:
        plt.savefig(f'{path_figures}/LBM_numpy_timestep_{t:04d}.png', dpi=120)
    
    plt.pause(0.01)
    plt.cla()
    
    
def create_cylinder_mask(Ny, Nx, obstacles):
    y, x = np.ogrid[:Ny, :Nx]
    cylinder = np.full((Ny, Nx), False)
    
    for obs in obstacles:
        center_x, center_y = obs['pos-x'], obs['pos-y']
        radius = obs['radius']
        
        # Create a mask for this obstacle
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Update the cylinder (mask) array
        cylinder |= mask # (in-place OR) equivalent to "cylinder = cylinder | mask"
        # but we avoid creating a new array for each obstacle
        # It updates "cylinder" in-place with new "mask" array
        
    return cylinder


def create_cylinder_mask_naive(Ny, Nx, obstacles):
    cylinder = np.full((Ny, Nx), False)
    
    for y in range(0, Ny):
        for x in range(0, Nx):
            for obs in obstacles: 
                if distance(obs['pos-x'], obs['pos-y'], x, y) < obs['radius']:
                    cylinder[y, x] = True
                    
    return cylinder


def main():
    Nx, Ny = 400, 100 # Grid resolution
    tau = 0.53 # Kinnematic viscosity or time scale
    Nt = 6000
    plotRealTime = True
    saveImages = True
    saveVTK = True
    obstacles = [{"pos-x": Nx//4, "pos-y": Ny//4, "radius": 8},
                  {"pos-x": Nx//4, "pos-y": 2*Ny//4, "radius": 8},
                  {"pos-x": Nx//4, "pos-y": 3*Ny//4, "radius": 8}]
    
    # Delete previous images before saving new ones
    if saveImages:
        path_figures = os.path.join(os.getcwd() + "/..", "figures") 
        os.makedirs(path_figures, exist_ok=True)
        utils.clear_folder_contents(path_figures)
        
    if saveVTK:
        path_vtk = os.path.join(os.getcwd() + "/..", "vtk") 
        os.makedirs(path_vtk, exist_ok=True)
        utils.clear_folder_contents(path_vtk)
    
    # Lattice speeds and weights
    NL = 9 # D2Q9 (9 lattice directions)
    cxs = np.array([0, 0, 1, 1,  1,  0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1,  0,  1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36]) # sums to 1
    
    # Initial Conditions (ICs)
    # F: Discrete velocities
    F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL) 
    # np.ones() or np.zeros()??
    F[:, :, 3] = 2.3 # We assume there is a flow in the +x-direction (direction=3)
    
    # Apply BCs (obstacles)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    cylinder = create_cylinder_mask(Ny, Nx, obstacles)
                
    # Main time loop
    for t in range(Nt+1):
        print(t)
        
        # Apply BCs at left and right walls to absorb waves
        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]
        
        # Streaming step (Sharing velocities between neighbors)
        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)
            
        # Apply Reflective BCs
        bcF = F[cylinder, :]
        bdF = bcF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]] # Reflect (opposite directions)
        
        # Fluid variables
        rho = np.sum(F, 2) # density
        ux = np.sum(F * cxs, 2) / rho # momentum
        uy = np.sum(F * cys, 2) / rho
        
        # Apply BCs?
        F[cylinder, :] = bdF
        ux[cylinder] = 0
        uy[cylinder] = 0
        
        # Collision step (BGK Equation)
        Feq = equilibrium(F, cxs, cys, NL, weights, rho, ux, uy)
        F += -(1/tau) * (F-Feq)
        
        if (plotRealTime and (t % plot_every == 0)):
            field_var = "Vorticity"
            plot_field(field_var, ux, uy, cylinder, t, saveImages, path_figures)
            if saveVTK:
                utils.write_VTK(path_vtk, f"Velocity_{t:04d}", 
                                X, Y, ux, uy)
        
    if saveImages:
        utils.create_gif_with_PIL(path_figures, f"../videos/{field_var}.gif")
        
    
if __name__ == "__main__":
    main()































