from PIL import Image
import glob
import os
import shutil
import numpy as np
from pyevtk.hl import gridToVTK

def create_gif_with_PIL(path, gif_filename):
    
    # Get list of image files
    image_files = sorted(glob.glob(os.path.join(path, '*.png'))) # give pattern (".png")
    
    # Open images
    images = [Image.open(file) for file in image_files]
    
    # Create animation as GIF
    images[0].save(gif_filename, save_all=True, append_images=images[1:],
                   duration=200, loop=0)
    # 'save_all=True' tells it to include all frames
    # 'append_images' specifies the rest of the frames (excluding the first one)
    # 'duration' sets how long each frame is displayed (in milliseconds)
    # 'loop=0' makes the GIF loop indefinitely
    print(f"{gif_filename} is saved in {path}")
    
    
def clear_folder_contents(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # List all files in the folder
        files = os.listdir(folder_path)
        
        if files:  # If the folder is not empty
            print(f"The folder '{folder_path}' contains files. Deleting them...")
            for file in files:
                file_path = os.path.join(folder_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)  # Delete file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Delete directory and its contents
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            print("All files have been deleted.")
        else:
            print(f"The folder '{folder_path}' is empty.")
    else:
        print(f"The folder '{folder_path}' does not exist.")
        
        
def write_VTK(path, filename, x, y, velocity_x, velocity_y):
    # 2D data needs to be converted to 3D for VTK format
    ny, nx = x.shape
    z = np.zeros((ny, nx, 1))
    x3d = x[:, :, np.newaxis]
    y3d = y[:, :, np.newaxis]
    
    # Extend velocity components to 3D
    vx = velocity_x[:, :, np.newaxis]
    vy = velocity_y[:, :, np.newaxis]
    vz = np.zeros_like(vx)
    
    # Write VTK file
    gridToVTK(os.path.join(path, filename), 
              x3d, y3d, z, 
              pointData={"Velocity": (vx, vy, vz)})
    
    
    
    