import io
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from .visualization import visualize_sdf
from .sdf_object import SDFObject
from .callbacks import Callback
from torch.utils.tensorboard import SummaryWriter
import time  # Import the time module

def evaluate_and_visualize(model, input_path: str, writer: SummaryWriter, epoch: int, loss: float) -> None:
    writer.add_scalar('Loss', loss, epoch)

    # Remember the current device
    original_device = next(model.parameters()).device
    device = "cpu"
    model.to(device)
    model.eval()

    sdf_object = SDFObject.load(input_path)
    edge_voxels_input = sdf_object.get_edge_voxels_tensor().to(device)
    edge_voxels_input = edge_voxels_input.unsqueeze(0)

    # Start timing here
    start_time = time.time()

    with torch.no_grad():
        predicted_sdf_array = model(edge_voxels_input).squeeze(0).cpu().numpy()

    # End timing here
    end_time = time.time()
    total_time = end_time - start_time

    # Log the evaluation time
    print(f"Evaluation time for a single batch: {total_time:.4f} seconds")
    writer.add_scalar('Evaluation Time', total_time, epoch)

    predicted_sdf_object = SDFObject(predicted_sdf_array, "Predicted")

    buffer = io.BytesIO()
    visualize_sdf(sdf_object.get_edge_voxels(), sdf_object.get_target(), predicted_sdf_object, buffer=buffer)
    buffer.seek(0)

    image = ToTensor()(plt.imread(buffer, format='png'))
    writer.add_image('SDF Visualization', image, global_step=epoch)

    # Set the model back to its original device
    model.to(original_device)


class EvaluateAndVisualizeCallback(Callback):
    """
    Callback for evaluating and visualizing model performance.
    """
    def __init__(self, input_path, writer: SummaryWriter, visualize_every_n_epochs=10):
        self.input_path = input_path
        self.writer = writer
        self.visualize_every_n_epochs = visualize_every_n_epochs

    def on_epoch_end(self, epoch, model, loss: float):
        if epoch % self.visualize_every_n_epochs == 0:
            evaluate_and_visualize(model, self.input_path, self.writer, epoch=epoch, loss=loss)