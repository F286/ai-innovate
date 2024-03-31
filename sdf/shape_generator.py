import numpy as np
import cv2
import math

class ShapeGenerator:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def create_box_dense_grid(self, location, size, rotation):
        grid = np.zeros(self.grid_size, dtype=np.uint8)
        center = location
        half_size = size // 2
        # Calculate rectangle coordinates before rotation
        rect = ((center[0], center[1]), (size, size), rotation)
        # Calculate the points of the rectangle
        box = cv2.boxPoints(rect).astype(np.int0)
        # Draw the rotated rectangle
        cv2.drawContours(grid, [box], 0, (1), thickness=cv2.FILLED)
        return grid

    def create_circle_dense_grid(self, location, size):
        grid = np.zeros(self.grid_size, dtype=np.uint8)
        center = location
        radius = size // 2
        # Draw the circle
        cv2.circle(grid, center, radius, (1), thickness=cv2.FILLED)
        return grid

    def create_triangle_dense_grid(self, location, size, rotation):
        grid = np.zeros(self.grid_size, dtype=np.uint8)
        half_size = size // 2
        # Define triangle vertices relative to center
        vertices = np.array([
            [location[0], location[1] - half_size],
            [location[0] - half_size, location[1] + half_size],
            [location[0] + half_size, location[1] + half_size]
        ], dtype=np.int32)
        
        # Rotate vertices if rotation is applied
        if rotation != 0:
            angle = math.radians(rotation)
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            rotated_vertices = []
            for x, y in vertices:
                x_rotated = cos_angle * (x - location[0]) - sin_angle * (y - location[1]) + location[0]
                y_rotated = sin_angle * (x - location[0]) + cos_angle * (y - location[1]) + location[1]
                rotated_vertices.append([int(x_rotated), int(y_rotated)])
            vertices = np.array(rotated_vertices, dtype=np.int32)

        # Draw the polygon (triangle in this case)
        cv2.fillPoly(grid, [vertices], (1))
        return grid
