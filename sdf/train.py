import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# Assuming Adafactor might be used in some scenarios, keeping the import
from transformers import Adafactor
from .sdf_model_separable_conv import SDFNet
from .sdf_dataset import SDFDataset
from .callbacks import Callback
from torch.cuda.amp import autocast, GradScaler
import os

def train_model(train_dir: str, callback: Callback = None) -> SDFNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDFNet().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = Adafactor(model.parameters())
    criterion = nn.MSELoss()

    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    train_dataset = SDFDataset(train_files)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, prefetch_factor=1, persistent_workers=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, prefetch_factor=64, persistent_workers=True)
    # train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, prefetch_factor=64, persistent_workers=True)

    # Initialize the gradient scaler
    scaler = GradScaler()

    model.train()
    for epoch in range(1000000):
        epoch_loss = 0.0  # Initialize epoch loss
        num_batches = 0  # Initialize batch counter

        for edge_voxels, target in train_loader:
            edge_voxels, target = edge_voxels.to(device), target.to(device)

            optimizer.zero_grad()

            # Run the forward pass under autocast
            with autocast():
                output = model(edge_voxels)
                loss = criterion(output, target)

            # Scale the loss and call backward to create scaled gradients
            scaler.scale(loss).backward()

            # Unscales the gradients and calls or skips optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            epoch_loss += float(loss.item())  # Accumulate loss for the epoch
            num_batches += 1  # Increment batch counter

            print(f"Epoch {epoch}, Batch Loss: {loss.item()}")

        epoch_loss /= num_batches  # Calculate average loss for the epoch

        if callback is not None:
            # Call the callback with the average loss for the epoch
            callback.on_epoch_end(epoch, model, loss=epoch_loss)

        print(f"Epoch {epoch}, Average Loss: {epoch_loss}")

    return model  # Return the trained model