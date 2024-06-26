import io
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from .visualization import visualize_sdf
from .sdf_object import SDFObject
from .callbacks import Callback
from torch.utils.tensorboard import SummaryWriter


def evaluate_and_visualize(model, input_path: str, writer: SummaryWriter, epoch: int, loss: float) -> None:
    
    writer.add_scalar('Loss', loss, epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sdf_object = SDFObject.load(input_path)
    edge_voxels_input = sdf_object.get_edge_voxels_tensor().to(device)
    edge_voxels_input = edge_voxels_input.unsqueeze(0)

    with torch.no_grad():
        predicted_sdf_array = model(edge_voxels_input).squeeze(0).cpu().numpy()

    predicted_sdf_object = SDFObject(predicted_sdf_array, "Predicted")

    # Assuming visualize_sdf can save the plot to a buffer, but modifying it to display images in a square layout
    buffer = io.BytesIO()
    # The number of objects to visualize is 4 (original, edge voxels, target, predicted), so we aim for a 2x2 layout
    visualize_sdf(sdf_object.get_edge_voxels(), sdf_object.get_target(), predicted_sdf_object, buffer=buffer)
    buffer.seek(0)

    # Convert buffer to Tensor
    image = ToTensor()(plt.imread(buffer, format='png'))
    writer.add_image('SDF Visualization', image, global_step=epoch)

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