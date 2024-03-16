import numpy as np
from sdf_object import SDFObject

def save_sdf(filename, sdf_object):
    np.save(filename, sdf_object.sdf_data)

def load_sdf(filename):
    sdf_data = np.load(filename)
    return SDFObject(sdf_data)