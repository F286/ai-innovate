
import torch
from sdf_model import SDFNet
import numpy as np

def predict(model_path, edge_voxels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDFNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        edge_voxels = torch.tensor(edge_voxels).float().unsqueeze(0).to(device)  # Add batch dimension
        prediction = model(edge_voxels)
        return prediction.squeeze(0).cpu().numpy()  # Remove batch dimension
