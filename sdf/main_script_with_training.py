
from data_processor import SDFDataset
from torch.utils.data import DataLoader
import torch
import os
import sys

from sdf_model import SDFNet
from train import train_model
# from predict import predict  # Uncomment if prediction functionality is needed

from visualization import visualize_comparison
from sdf_object import SDFObject

def visualize_first_entry(train_dir):
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    first_file = train_files[0]
    sdf_object = SDFObject.load(first_file)
    target_sdf = sdf_object.get_target()
    edge_voxels_sdf = sdf_object.get_edge_voxels()
    
    visualize_comparison(edge_voxels_sdf, target_sdf)


if __name__ == "__main__":
    train_dir = 'sdf/sdf_variations'  # Define the path to your training data
    
    # Visualize the first entry
    visualize_first_entry(train_dir)


    train_model(train_dir)  # Example call to the training function
    # Add more code here for evaluation and possibly for predictions
