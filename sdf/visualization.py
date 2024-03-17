import matplotlib.pyplot as plt
from sdf_object import SDFObject
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
    
    axs[0].imshow(original_data, cmap='RdBu')
    axs[0].set_title('Original SDF')
    axs[1].imshow(predicted_data, cmap='RdBu')
    axs[1].set_title('Predicted SDF')
    
    plt.show()