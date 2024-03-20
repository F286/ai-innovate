import matplotlib.pyplot as plt
from .sdf_object import SDFObject
import numpy as np

def visualize_sdf(*sdf_objects: 'SDFObject'):
    """
    Visualizes a variable number of SDFObjects.
    
    Args:
        *sdf_objects: A variable number of SDFObject instances to be visualized.
    """
    num_objects = len(sdf_objects)
    fig, axs = plt.subplots(1, num_objects, figsize=(5 * num_objects, 5))
    
    if num_objects == 1:  # If there's only one object, axs is not a list but a single AxesSubplot
        axs = [axs]
    
    for ax, sdf_object in zip(axs, sdf_objects):
        # Assuming sdf_object.sdf_data is a numpy array
        # Use np.squeeze to remove single-dimensional entries from the shape
        data = np.squeeze(sdf_object.sdf_data)
        
        # Automatically determine the min and max values for the SDF for proper visualization
        data_min, data_max = data.min(), data.max()
        
        # Visualize the SDF with its respective min and max values for normalization
        im = ax.imshow(data, cmap='RdBu', vmin=data_min, vmax=data_max)
        ax.set_title(f'{sdf_object.name}')  # Ideally, we'd use variable names, but Python doesn't support that directly
        
        # Add a colorbar to the subplot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.show()  # Non-blocking show
