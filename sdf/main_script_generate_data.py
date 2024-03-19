import numpy as np
from .shape_generator import ShapeGenerator
from .sdf_generator import SDFGenerator
from .save_load import save_sdf, load_sdf
from .visualization import visualize_sdf
import os

np.random.seed(42)  # Set seed for consistent randomness


def generate_variations_and_save(sdf_folder='sdf/sdf_variations'):
    os.makedirs(sdf_folder, exist_ok=True)  # Create the sub-folder if it doesn't exist
    shape_gen = ShapeGenerator(grid_size=(100, 100))

    for i in range(1024):
        num_boxes = np.random.randint(1, 4)  # Randomly decide to generate 1 to 3 boxes
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


def visualize_random_variation(sdf_folder='sdf/sdf_variations'):
    variation_number = np.random.randint(0, 32)
    filepath = os.path.join(sdf_folder, f'sdf_variation_{{:04d}}.npy'.format(variation_number))
    sdf_object = load_sdf(filepath)
    print(f"Visualizing: {filepath}")
    visualize_sdf(sdf_object)


if __name__ == "__main__":
    sdf_folder = 'sdf/sdf_variations'
    generate_variations_and_save(sdf_folder=sdf_folder)
    visualize_random_variation(sdf_folder=sdf_folder)
