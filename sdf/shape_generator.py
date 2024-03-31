
import numpy as np
from typing import Tuple
import math

class ShapeGenerator:
    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_size = grid_size

    def create_box_dense_grid(self, location: Tuple[int, int], size: int, rotation: int) -> np.ndarray:
        grid = np.zeros(self.grid_size, dtype=np.int8)
        center_x, center_y = location
        half_size = size // 2
        angle = math.radians(rotation)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        for x in range(center_x - half_size, center_x + half_size):
            for y in range(center_y - half_size, center_y + half_size):
                # Rotate point around the center
                x_rotated = cos_angle * (x - center_x) - sin_angle * (y - center_y) + center_x
                y_rotated = sin_angle * (x - center_x) + cos_angle * (y - center_y) + center_y
                if 0 <= int(x_rotated) < self.grid_size[0] and 0 <= int(y_rotated) < self.grid_size[1]:
                    grid[int(y_rotated), int(x_rotated)] = 1
        return grid

    def create_circle_dense_grid(self, location: Tuple[int, int], size: int) -> np.ndarray:
        grid = np.zeros(self.grid_size, dtype=np.int8)
        center_x, center_y = location
        radius = size // 2

        for x in range(center_x - radius, center_x + radius):
            for y in range(center_y - radius, center_y + radius):
                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                    if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                        grid[y, x] = 1
        return grid

    def create_triangle_dense_grid(self, location: Tuple[int, int], size: int, rotation: int) -> np.ndarray:
        grid = np.zeros(self.grid_size, dtype=np.int8)
        center_x, center_y = location
        half_size = size // 2
        angle = math.radians(rotation)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        # Define triangle vertices relative to center
        vertices = [
            (center_x, center_y - half_size),  # Top vertex
            (center_x - half_size, center_y + half_size),  # Bottom left
            (center_x + half_size, center_y + half_size)  # Bottom right
        ]

        # Rotate vertices
        rotated_vertices = []
        for vx, vy in vertices:
            x_rotated = cos_angle * (vx - center_x) - sin_angle * (vy - center_y) + center_x
            y_rotated = sin_angle * (vx - center_x) + cos_angle * (vy - center_y) + center_y
            rotated_vertices.append((x_rotated, y_rotated))

        # Fill triangle
        for x in range(center_x - size, center_x + size):
            for y in range(center_y - size, center_y + size):
                if self._point_inside_triangle((x, y), rotated_vertices):
                    if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                        grid[y, x] = 1
        return grid

    def _point_inside_triangle(self, point: Tuple[int, int], vertices: list) -> bool:
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = sign(point, vertices[0], vertices[1]) < 0.0
        b2 = sign(point, vertices[1], vertices[2]) < 0.0
        b3 = sign(point, vertices[2], vertices[0]) < 0.0

        return ((b1 == b2) and (b2 == b3))
