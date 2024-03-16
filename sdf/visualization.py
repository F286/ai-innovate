import matplotlib.pyplot as plt
from sdf_object import SDFObject

def visualize_sdf(sdf_object):
    plt.imshow(sdf_object.sdf_data, cmap='RdBu')
    plt.colorbar()
    plt.title('Signed Distance Field Visualization')
    plt.show()