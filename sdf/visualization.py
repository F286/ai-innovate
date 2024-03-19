import matplotlib.pyplot as plt
from .sdf_object import SDFObject
import numpy as np

def visualize_sdf(sdf_object: SDFObject):
    """Visualizes the Signed Distance Field (SDF) contained within an SDFObject."""
    plt.imshow(sdf_object.sdf_data, cmap='RdBu')
    plt.colorbar()
    plt.title('Signed Distance Field Visualization')
    plt.show()

def visualize_comparison(original_sdf: 'SDFObject', predicted_sdf: 'SDFObject'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Assuming original_sdf.sdf_data and predicted_sdf.sdf_data are numpy arrays
    # Use np.squeeze to remove single-dimensional entries from the shape
    original_data = np.squeeze(original_sdf.sdf_data)
    predicted_data = np.squeeze(predicted_sdf.sdf_data)
    
    # # Check if the original or predicted SDF data contains all the same values
    # if np.all(original_data == original_data[0,0]):
    #     print("Warning: Original SDF data contains all the same values.")
    # if np.all(predicted_data == predicted_data[0,0]):
    #     print("Warning: Predicted SDF data contains all the same values.")
    
    # Automatically determine the min and max values for original and predicted SDFs for proper visualization
    original_min, original_max = original_data.min(), original_data.max()
    predicted_min, predicted_max = predicted_data.min(), predicted_data.max()
    
    # Visualize the original and predicted SDFs with their respective min and max values for normalization
    im1 = axs[0].imshow(original_data, cmap='RdBu', vmin=original_min, vmax=original_max)
    axs[0].set_title('Original SDF')
    fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)  # Add a colorbar to the first subplot
    
    im2 = axs[1].imshow(predicted_data, cmap='RdBu', vmin=predicted_min, vmax=predicted_max)
    axs[1].set_title('Predicted SDF')
    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)  # Add a colorbar to the second subplot
    
    plt.show()