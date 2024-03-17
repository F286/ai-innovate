
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
    edge_voxels_sdf = sdf_object.get_edge_voxels()
    target_sdf = sdf_object.get_target()
    
    visualize_comparison(edge_voxels_sdf, target_sdf)


def evaluate_and_visualize(model: SDFNet, input_path: str) -> None:
    """
    Evaluates a given input using a trained model and visualizes the prediction.

    Parameters:
    - model: The trained SDFNet model.
    - input_path: Path to the input data for evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Ensure the model is on the correct device
    model.eval()  # Set the model to evaluation mode

    # Load the input data
    sdf_object = SDFObject.load(input_path)
    edge_voxels_input = sdf_object.get_edge_voxels().to(device)

    # Perform prediction
    with torch.no_grad():
        edge_voxels_input = torch.tensor(edge_voxels_input).float().unsqueeze(0)  # Add batch dimension
        predicted_sdf_array = model(edge_voxels_input).squeeze(0).cpu().numpy()  # Remove batch dimension

    # Convert the predicted numpy array back to an SDFObject for visualization
    predicted_sdf_object = SDFObject(predicted_sdf_array)

    # Visualize the original and predicted SDF
    visualize_comparison(sdf_object, predicted_sdf_object)


if __name__ == "__main__":
    train_dir = 'sdf/sdf_variations'  # Define the path to your training data
    
    # Visualize the first entry
    # visualize_first_entry(train_dir)


    # Train the model and receive the trained model directly
    trained_model = train_model(train_dir)
    
    # Path to the sample input for evaluation
    input_path = 'sdf/sdf_evaluate/sample_input.sdf'

    # Evaluate the model and visualize the results using the trained model directly
    evaluate_and_visualize(trained_model, input_path)
