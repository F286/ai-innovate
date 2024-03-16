from scipy.ndimage import distance_transform_edt
import numpy as np
from sdf_object import SDFObject

class SDFGenerator:
    @staticmethod
    def generate_sdf_from_dense_grid(dense_grid):
        outside_distances = distance_transform_edt(1 - dense_grid)
        inside_distances = distance_transform_edt(dense_grid)
        sdf = outside_distances - inside_distances
        return SDFObject(sdf)