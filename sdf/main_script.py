from shape_generator import ShapeGenerator
from sdf_generator import SDFGenerator
import numpy as np
import os


def save_sdf_example(filename, sdf_object):
    np.save(filename, sdf_object.sdf_data)


if __name__ == "__main__":
    shape_gen = ShapeGenerator(grid_size=(100, 100))
    dense_grid_1 = shape_gen.create_box_dense_grid((20, 20), (40, 40))
    dense_grid_2 = shape_gen.create_box_dense_grid((60, 60), (80, 80))
    combined_dense_grid = np.logical_or(dense_grid_1, dense_grid_2).astype(int)

    sdf_object = SDFGenerator.generate_sdf_from_dense_grid(combined_dense_grid)
    filepath = os.path.join('sdf_examples', 'sdf_example.npy')
    save_sdf_example(filepath, sdf_object)