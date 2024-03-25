import numpy as np
from .shape_generator import ShapeGenerator
from .sdf_generator import SDFGenerator
from .save_load import save_sdf, load_sdf
from .visualization import visualize_sdf
import os

np.random.seed(42)  # Set seed for consistent randomness


def generate_variations_and_save(sdf_folder='sdf/sdf_variations', generate_count: int = 8192):
    os.makedirs(sdf_folder, exist_ok=True)  # Create the sub-folder if it doesn't exist
    shape_gen = ShapeGenerator(grid_size=(128, 128))

    for i in range(generate_count):
        num_boxes = np.random.randint(1, 16)  # Randomly decide to generate 1 to 3 boxes
        combined_dense_grid = np.zeros(shape_gen.grid_size, dtype=np.int8)

        for _ in range(num_boxes):
            top_left = (np.random.randint(10, 60), np.random.randint(10, 60))
            bottom_right = (np.random.randint(top_left[0] + 10, 90), np.random.randint(top_left[1] + 10, 90))
            dense_grid = shape_gen.create_box_dense_grid(top_left, bottom_right)
            combined_dense_grid = np.logical_or(combined_dense_grid, dense_grid).astype(int)

        sdf_object = SDFGenerator.generate_sdf_from_dense_grid(combined_dense_grid)
        # Four-digit numbering for the files, saved in the specified sub-folder
        filepath = os.path.join(sdf_folder, f'sdf_variation_{{:04d}}.npy'.format(i))
        save_sdf(filepath, sdf_object)


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
    # sdf_evaluate_folder = 'sdf/sdf_evaluate'
    # generate_variations_and_save(sdf_folder=sdf_evaluate_folder, generate_count=1)
    # visualize_variation(sdf_folder=sdf_evaluate_folder, sdf_filename=f'sdf_variation_0000.npy')

    sdf_folder = 'sdf/sdf_variations'
    generate_variations_and_save(sdf_folder=sdf_folder)
    visualize_random_variation(sdf_folder=sdf_folder)
