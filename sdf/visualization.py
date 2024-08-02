import matplotlib.pyplot as plt
from .sdf_object import SDFObject
import numpy as np
import io

import matplotlib.pyplot as plt
import numpy as np
import io

def visualize_sdf(*sdf_objects: 'SDFObject', buffer: io.BytesIO = None, depths: np.ndarray = np.linspace(0, 1, 4)):
    num_depths = len(depths)
    num_objects = len(sdf_objects)
    # Adjust the subplot layout to accommodate all SDF objects, adding extra space for the colorbar
    fig, axs = plt.subplots(num_objects, num_depths, figsize=(4*num_depths, 3.5*num_objects + 1), squeeze=False)
    for obj_index, sdf_object in enumerate(sdf_objects):
        # Check if the sdf_data has a batch dimension and assert it's 1 if present
        if sdf_object.sdf_data.ndim == 4:
            assert sdf_object.sdf_data.shape[0] == 1, "Batch size must be 1 for visualization."
            # Remove the batch dimension for visualization
            sdf_data_no_batch = sdf_object.sdf_data.squeeze(0)
        else:
            sdf_data_no_batch = sdf_object.sdf_data

        # Determine global min and max across all specified depths for uniform color scaling
        global_min = np.inf
        global_max = -np.inf
        for z in depths:
            z_index = int((z + 1) / 2 * (sdf_data_no_batch.shape[0] - 1))
            Z_fine = sdf_data_no_batch[z_index, :, :]
            global_min = min(global_min, Z_fine.min())
            global_max = max(global_max, Z_fine.max())

        # Set the title for each row corresponding to an SDF object
        axs[obj_index, 0].set_ylabel(f'{sdf_object.name}', rotation=0, size='large', labelpad=60)

        images = []
        for i, z in enumerate(depths):
            z_index = int((z + 1) / 2 * (sdf_data_no_batch.shape[0] - 1))
            ax = axs[obj_index, i]
            ax.set_title(f'Depth: {z:.2f}')
            Z_fine = sdf_data_no_batch[z_index, :, :]
            
            # Assert to verify the shape of Z_fine is valid for rendering
            assert Z_fine.ndim == 2, f"Expected Z_fine to be a 2D array, got {Z_fine.ndim}D array instead."
            
            # Apply the global min and max for color scaling
            img = ax.imshow(Z_fine, cmap='coolwarm', interpolation='bilinear', aspect='auto', extent=[0, Z_fine.shape[1], 0, Z_fine.shape[0]], vmin=global_min, vmax=global_max)
            images.append(img)
            
            ax.set_xlabel('X axis')
            # Only set Y axis label for the first column
            if i == 0:
                ax.set_ylabel('Y axis')

    # Create a single colorbar for the entire figure based on the scale of images
    # Adjust the placement of the colorbar to prevent overlap with the bottom images
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # x, y, width, height
    fig.colorbar(images[0], cax=cbar_ax, orientation='horizontal', aspect=50)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust the bottom margin to accommodate the colorbar

    if buffer is not None:
        plt.savefig(buffer, format='png')
        plt.close(fig)
    else:
        plt.show()
