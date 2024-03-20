import torch

from .visualization import visualize_sdf
from .sdf_object import SDFObject
from .sdf_model import SDFNet
from .callbacks import Callback

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
    edge_voxels_input = sdf_object.get_edge_voxels_tensor().to(device)

    # Perform prediction
    with torch.no_grad():
        predicted_sdf_array = model(edge_voxels_input).squeeze(0).cpu().numpy()  # Remove batch dimension

    # Convert the predicted numpy array back to an SDFObject for visualization
    predicted_sdf_object = SDFObject(predicted_sdf_array, "Predicted")

    # Visualize the original and predicted SDF
    visualize_sdf(sdf_object, sdf_object.get_edge_voxels(), sdf_object.get_target(), predicted_sdf_object)


class EvaluateAndVisualizeCallback(Callback):
    """
    Callback for evaluating and visualizing model performance.
    """
    def __init__(self, input_path, visualize_every_n_epochs=10):
        self.input_path = input_path
        self.visualize_every_n_epochs = visualize_every_n_epochs

    def on_epoch_end(self, epoch, model):
        if (epoch + 1) % self.visualize_every_n_epochs == 0:
            # Assuming evaluate_and_visualize is a function that performs evaluation and visualization
            evaluate_and_visualize(model, self.input_path)