import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Print the current CUDA device
print(f"Current CUDA Device: {torch.cuda.current_device()}")

# Print the name of the current CUDA device
print(f"Current CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Print the number of GPUs available
print(f"Number of GPUs Available: {torch.cuda.device_count()}")
