
from data_processor import SDFDataset
from torch.utils.data import DataLoader
import torch
import os
import sys

from sdf_model import SDFNet
from train import train_model
# from predict import predict  # Uncomment if prediction functionality is needed

if __name__ == "__main__":
    train_dir = 'sdf/sdf_variations'  # Define the path to your training data
    train_model(train_dir)  # Example call to the training function
    # Add more code here for evaluation and possibly for predictions
