from .sdf_dataset import SDFDataset
from torch.utils.data import DataLoader
import torch
import os
import sys
from torch.utils.tensorboard import SummaryWriter

from .train import train_model

from .visualization import visualize_sdf
from .sdf_object import SDFObject
from .evaluate_and_visualize_callback import EvaluateAndVisualizeCallback

# def visualize_first_entry(train_dir):
#     train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
#     first_file = train_files[5]
#     sdf_object = SDFObject.load(first_file)
#     edge_voxels_sdf = sdf_object.get_edge_voxels()
#     target_sdf = sdf_object.get_target()
    
#     visualize_sdf(sdf_object, edge_voxels_sdf, target_sdf)


if __name__ == "__main__":
    train_dir = 'sdf/sdf_variations'  # Define the path to your training data
    input_path = 'sdf/sdf_evaluate/sdf_variation_input.npy'
    
    # Visualize the first entry
    # visualize_first_entry(train_dir)

    # Initialize your callback
    writer = SummaryWriter('sdf/conv_next1')
    callback = EvaluateAndVisualizeCallback(input_path, writer, visualize_every_n_epochs=10)

    trained_model = train_model(train_dir, callback=callback)
    
    # Path to the sample input for evaluation
    # input_path = 'sdf/sdf_evaluate/sample_input.npy'

    # Evaluate the model and visualize the results using the trained model directly
    # evaluate_and_visualize(trained_model, input_path)
