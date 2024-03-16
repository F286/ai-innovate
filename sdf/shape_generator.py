
import numpy as np

class ShapeGenerator:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def create_box_dense_grid(self, top_left, bottom_right):
        grid = np.zeros(self.grid_size, dtype=np.int8)
        grid[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 1
        return grid
