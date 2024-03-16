import numpy as np

class SDFObject:
    def __init__(self, sdf_data):
        self.sdf_data = sdf_data

    @staticmethod
    def load(filename):
        sdf_data = np.load(filename)
        return SDFObject(sdf_data)