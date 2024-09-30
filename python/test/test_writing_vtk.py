import numpy as np
from pyevtk.hl import gridToVTK
import pickle

#=============================================
with open('data.pickle', 'rb') as file:
    data = pickle.load(file)
filename = "simdata"
XX = data["X"]
YY = data["Y"]
velocity_xx = data["velocity_x"]
velocity_yy = data["velocity_y"]
#=============================================

# Set up the grid
nx, ny = 400, 100
x = np.linspace(0, 1, nx)
y = np.linspace(0, 0.25, ny)
X, Y = np.meshgrid(x, y)

# Ensure X and Y have shape (ny, nx)
assert X.shape == (ny, nx), f"X shape is {X.shape}, expected {(ny, nx)}"
assert Y.shape == (ny, nx), f"Y shape is {Y.shape}, expected {(ny, nx)}"

# Generate random velocity fields with correct shape
np.random.seed(42)  # for reproducibility
velocity_x = np.random.rand(ny, nx) - 0.5
velocity_y = np.random.rand(ny, nx) - 0.5

# Verify velocity field shapes
assert velocity_x.shape == (ny, nx), f"velocity_x shape is {velocity_x.shape}, expected {(ny, nx)}"
assert velocity_y.shape == (ny, nx), f"velocity_y shape is {velocity_y.shape}, expected {(ny, nx)}"

def write_vtk(filename, x, y, velocity_x, velocity_y):
    # Extend 2D data to 3D
    z = np.zeros((y.shape[0], x.shape[1], 1))
    x3d = x[:,:,np.newaxis]
    y3d = y[:,:,np.newaxis]
    
    # Extend velocity components to 3D
    vx = velocity_x[:,:,np.newaxis]
    vy = velocity_y[:,:,np.newaxis]
    vz = np.zeros_like(vx)
    
    with open(f"../vtk/{filename}.pickle", 'wb') as file:
        data = {
            "filename": filename,
            "x3d": x3d,
            "y3d": y3d,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz
            }
        pickle.dump(data, file)
    
    # Write VTK file
    gridToVTK("../vtk/"+filename, x3d, y3d, z, pointData={"velocity": (vx, vy, vz)})

# Use the function to write the VTK file
write_vtk("syntheticdata", X, Y, velocity_x, velocity_y)
write_vtk(filename, XX.astype('float64'), YY.astype('float64'), velocity_xx, velocity_yy)



