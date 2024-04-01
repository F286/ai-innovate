import numpy as np
import math

class ShapeGenerator3D:
    def __init__(self, grid_size):
        # grid_size is expected to be a tuple of three dimensions now (depth, height, width)
        self.grid_size = grid_size

    def create_box_3d(self, location, size):
        grid = np.zeros(self.grid_size, dtype=np.uint8)
        # Calculate the start and end points of the box
        start = [max(0, location[i] - size // 2) for i in range(3)]
        end = [min(self.grid_size[i], location[i] + size // 2) for i in range(3)]
        # Fill the box area with ones
        grid[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 1
        return grid

    def create_sphere_3d(self, center, radius):
        grid = np.zeros(self.grid_size, dtype=np.uint8)
        # Generate coordinates for the grid
        z, y, x = np.indices(self.grid_size)
        # Calculate the distance of all points from the center
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        # Fill points within the specified radius
        grid[distance <= radius] = 1
        return grid

# Example usage
if __name__ == "__main__":
    generator = ShapeGenerator3D((100, 100, 100))  # Create a 100x100x100 grid
    box = generator.create_box_3d((50, 50, 50), 20)  # Create a box centered at (50, 50, 50) with size 20
    sphere = generator.create_sphere_3d((50, 50, 50), 20)  # Create a sphere centered at (50, 50, 50) with radius 20