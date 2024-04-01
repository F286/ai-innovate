import matplotlib.pyplot as plt
from .sdf_object import SDFObject
import numpy as np
import io

def visualize_sdf(*sdf_objects: 'SDFObject', buffer: io.BytesIO = None, depths: np.ndarray = np.linspace(-1.0, 1.0, 6)):
    """
    Visualizes slices of 3D SDFObjects at specified depths with contours. Can optionally save the visualization to a buffer.
    
    Args:
        *sdf_objects: A variable number of 3D SDFObject instances to be visualized.
        buffer: An optional BytesIO buffer to save the visualization to. If None, the plot will be shown.
        depths: An array of depths at which to slice the 3D SDF for visualization.
    """
    for sdf_object in sdf_objects:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z (Depth + SDF)')
        ax.set_title(f'Refined 3D SDF Visualization Across Depths: {sdf_object.name}')

        # Assuming sdf_object.sdf_data is a 3D numpy array
        data = sdf_object.sdf_data
        grid_size = data.shape
        x_fine = np.linspace(0, grid_size[2]-1, 50)
        y_fine = np.linspace(0, grid_size[1]-1, 50)
        
        # Plot slices at different z depths
        for z in depths:
            z_index = int((z + 1) / 2 * (grid_size[0] - 1))  # Convert from [-1, 1] to [0, grid_size[0]-1]
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
            Z_fine = data[z_index, :, :]  # Slice the data at this z index
            
            # Plot surface with improved resolution and adjusted transparency
            surf = ax.plot_surface(X_fine, Y_fine, np.full_like(X_fine, z_index), rstride=1, cstride=1, alpha=0.3, facecolors=plt.cm.coolwarm(Z_fine / np.max(Z_fine)), linewidth=0, antialiased=False)
            
            # Highlight zero crossing with contours
            ax.contour(X_fine, Y_fine, np.full_like(X_fine, z_index), levels=[z_index], colors='k', linestyles='solid', linewidths=0.5)

        ax.view_init(elev=30, azim=120)

        if buffer is not None:
            plt.savefig(buffer, format='png')  # Save the plot to the provided buffer
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()  # Display the plot if no buffer is provided