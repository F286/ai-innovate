import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sdf_model import SDFNet
from data_processor import SDFDataset
import os

def train_model(train_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDFNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Assuming the training data is in 'train_dir'
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    train_dataset = SDFDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model.train()
    for epoch in range(10):  # Example: 10 epochs
        for edge_voxels, target in train_loader:
            edge_voxels, target = edge_voxels.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(edge_voxels)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
