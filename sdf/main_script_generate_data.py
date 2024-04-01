import numpy as np
from .shape_generator import ShapeGenerator3D  # Updated import
from .sdf_generator import SDFGenerator  # This might need adjustments for 3D
from .save_load import save_sdf, load_sdf  # Ensure these are compatible with 3D data
from .visualization import visualize_sdf
import os

np.random.seed(42)  # Set seed for consistent randomness

def generate_and_save_random_volume(volume_name: str):
    """
    Generates a random 3D volume with a variety of shapes at fully random locations within the volume,
    with varying sizes, etc., and saves it to the specified path.
    
    Args:
    volume_name (str): The full path including the name of the volume to save.
    """
    shape_gen = ShapeGenerator3D(grid_size=(100, 100, 100))  # Updated for 3D
    num_shapes = np.random.randint(4, 11)  # Randomly decide to generate between 4 to 10 shapes
    combined_dense_grid = np.zeros(shape_gen.grid_size, dtype=np.int8)

    for _ in range(num_shapes):
        shape_type = np.random.choice(['box', 'sphere'])  # Updated shape types for 3D
        location = (np.random.randint(0, 100), np.random.randint(0, 100), np.random.randint(0, 100))
        size = np.random.randint(10, 50)  # Random size

        if shape_type == 'box':
            dense_grid = shape_gen.create_box_3d(location, size)
        elif shape_type == 'sphere':
            dense_grid = shape_gen.create_sphere_3d(location, size)

        combined_dense_grid = np.logical_or(combined_dense_grid, dense_grid).astype(int)

    # Assuming SDFGenerator and save_sdf are updated for 3D
    sdf_object = SDFGenerator.generate_sdf_from_dense_grid(combined_dense_grid)
    save_sdf(volume_name, sdf_object)

def generate_variations_and_save(sdf_folder='sdf/sdf_variations', generate_count: int = 1024):
    os.makedirs(sdf_folder, exist_ok=True)  # Create the sub-folder if it doesn't exist

    for i in range(generate_count):
        volume_name = os.path.join(sdf_folder, f'sdf_variation_{i:04d}.npy')
        generate_and_save_random_volume(volume_name)

# Visualization for 3D data would require a different approach, possibly slicing or 3D rendering
# This part of the script would need significant adjustments for 3D visualization


def visualize_variation(sdf_folder: str, sdf_filename: str):
    """
    Visualizes a specific SDF variation given its filename.
    
    Args:
    sdf_folder (str): The folder where the SDF files are stored.
    sdf_filename (str): The name of the SDF file to visualize.
    """
    filepath = os.path.join(sdf_folder, sdf_filename)
    sdf_object = load_sdf(filepath)
    print(f"Visualizing: {filepath}")
    visualize_sdf(sdf_object)

def visualize_random_variation(sdf_folder='sdf/sdf_variations'):
    """
    Selects a random SDF variation from the specified folder and visualizes it.
    
    Args:
    sdf_folder (str): The folder where the SDF files are stored.
    """
    variation_number = np.random.randint(0, 32)
    sdf_filename = f'sdf_variation_{{:04d}}.npy'.format(variation_number)
    visualize_variation(sdf_folder, sdf_filename)


if __name__ == "__main__":
    
    generate_and_save_random_volume(volume_name=f'sdf/sdf_evaluate/sdf_variation_input.npy')
    visualize_variation(sdf_folder=f'sdf/sdf_evaluate/', sdf_filename=f'sdf_variation_input.npy')

    sdf_folder = 'sdf/sdf_variations'
    generate_variations_and_save(sdf_folder=sdf_folder)
    visualize_random_variation(sdf_folder=sdf_folder)