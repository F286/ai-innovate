import numpy as np
from .shape_generator import ShapeGenerator
from .sdf_generator import SDFGenerator
from .save_load import save_sdf, load_sdf
from .visualization import visualize_sdf
import os

np.random.seed(42)  # Set seed for consistent randomness


def generate_and_save_random_image(image_name: str):
    """
    Generates a random image with a variety of shapes at fully random locations within the image,
    with varying sizes, rotations, etc., and saves it to the specified path.
    
    Args:
    image_name (str): The full path including the name of the image to save.
    """
    shape_gen = ShapeGenerator(grid_size=(256, 256))
    num_shapes = np.random.randint(5, 21)  # Randomly decide to generate 5 to 20 shapes
    combined_dense_grid = np.zeros(shape_gen.grid_size, dtype=np.int8)

    for _ in range(num_shapes):
        shape_type = np.random.choice(['box', 'circle', 'triangle'])
        location = (np.random.randint(0, 256), np.random.randint(0, 256))
        size = np.random.randint(10, 50)  # Random size between 10 and 50
        rotation = np.random.randint(0, 360)  # Random rotation between 0 and 360 degrees

        if shape_type == 'box':
            dense_grid = shape_gen.create_box_dense_grid(location, size, rotation)
        elif shape_type == 'circle':
            dense_grid = shape_gen.create_circle_dense_grid(location, size)
        elif shape_type == 'triangle':
            dense_grid = shape_gen.create_triangle_dense_grid(location, size, rotation)

        combined_dense_grid = np.logical_or(combined_dense_grid, dense_grid).astype(int)

    sdf_object = SDFGenerator.generate_sdf_from_dense_grid(combined_dense_grid)
    save_sdf(image_name, sdf_object)

def generate_variations_and_save(sdf_folder='sdf/sdf_variations', generate_count: int = 8192):
    os.makedirs(sdf_folder, exist_ok=True)  # Create the sub-folder if it doesn't exist

    for i in range(generate_count):
        # Four-digit numbering for the files, saved in the specified sub-folder
        image_name = os.path.join(sdf_folder, f'sdf_variation_{{:04d}}.npy'.format(i))
        generate_and_save_random_image(image_name)

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
    generate_and_save_random_image(image_name=f'sdf/sdf_evaluate/sdf_variation_input.npy')

    sdf_folder = 'sdf/sdf_variations'
    generate_variations_and_save(sdf_folder=sdf_folder)
    visualize_random_variation(sdf_folder=sdf_folder)
