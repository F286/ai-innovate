import matplotlib.pyplot as plt
from .sdf_object import SDFObject
import numpy as np

import io
import matplotlib.pyplot as plt
from .sdf_object import SDFObject
import numpy as np

def visualize_sdf(*sdf_objects: 'SDFObject', buffer: io.BytesIO = None, layout: str = 'square'):
    """
    Visualizes a variable number of SDFObjects. Can optionally save the visualization to a buffer.
    The layout parameter allows for a square layout of the visualizations.
    
    Args:
        *sdf_objects: A variable number of SDFObject instances to be visualized.
        buffer: An optional BytesIO buffer to save the visualization to. If None, the plot will be shown.
        layout: The layout of the visualizations. 'square' for a square layout, otherwise linear.
    """
    num_objects = len(sdf_objects)
    if layout == 'square':
        # Calculate the size of the grid needed to display the objects in a square (or nearly square) layout
        grid_size = int(np.ceil(np.sqrt(num_objects)))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))
    else:
        fig, axs = plt.subplots(1, num_objects, figsize=(5 * num_objects, 5))
    
    axs = np.array(axs).reshape(-1)  # Ensure axs is always a flat array, even for a single subplot
    
    for ax, sdf_object in zip(axs[:num_objects], sdf_objects):  # Only iterate through the number of sdf_objects
        data = np.squeeze(sdf_object.sdf_data)  # Assuming sdf_object.sdf_data is a numpy array
        data_min, data_max = data.min(), data.max()  # Determine min and max values for normalization
        
        im = ax.imshow(data, cmap='RdBu', vmin=data_min, vmax=data_max)
        ax.set_title(f'{sdf_object.name}')
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    for ax in axs[num_objects:]:  # Hide any unused subplots
        ax.axis('off')
    
    if buffer is not None:
        plt.savefig(buffer, format='png')  # Save the plot to the provided buffer
        plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()  # Display the plot if no buffer is provided