import numpy as np
import math

class ShapeGenerator3D:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def create_box_3d(self, location, size, rotation_angle=0):
        grid = np.zeros(self.grid_size, dtype=np.uint8)
        rotation_angle_rad = np.deg2rad(rotation_angle)
        cos_angle, sin_angle = np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)

        # Define the rotation matrix for the Z-axis
        rotation_matrix = np.array([[cos_angle, -sin_angle, 0],
                                    [sin_angle, cos_angle, 0],
                                    [0, 0, 1]])

        # Generate coordinates for the grid
        z, y, x = np.indices(self.grid_size).astype(float)
        # Shift the grid so that the rotation is around the center of the box
        z -= location[2]
        y -= location[1]
        x -= location[0]

        # Apply rotation
        rotated_coords = np.dot(rotation_matrix, np.array([z.flatten(), y.flatten(), x.flatten()]))
        z_rot, y_rot, x_rot = rotated_coords.reshape(3, *self.grid_size)

        # Calculate the mask for the box with rotation
        mask = (np.abs(x_rot) <= size / 2) & (np.abs(y_rot) <= size / 2) & (np.abs(z_rot) <= size / 2)
        grid[mask] = 1
        return grid

    def create_sphere_3d(self, center, radius):
        grid = np.zeros(self.grid_size, dtype=np.uint8)
        z, y, x = np.indices(self.grid_size)
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        grid[distance <= radius] = 1
        return grid

if __name__ == "__main__":
    generator = ShapeGenerator3D((100, 100, 100))
    box = generator.create_box_3d((50, 50, 50), 20, 45)  # Now with rotation angle of 45 degrees
    sphere = generator.create_sphere_3d((50, 50, 50), 20)
