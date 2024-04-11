import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import sys

def profile_model(model, device='cuda'):
    # Ensure the model is in evaluation mode
    model.eval()

    # Create dummy data to simulate a typical input
    dummy_input = torch.randn(1, 1, 256, 256, 256).to(device)  # Adjust the size as per your model's input

    # Number of forward passes to profile
    num_passes = 10

    # Setup profiler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 record_shapes=True, 
                 profile_memory=True, 
                 with_stack=True) as prof:
        with torch.no_grad():  # Ensure no gradients are computed
            for _ in range(num_passes):
                with record_function("model_forward"):
                    outputs = model(dummy_input)

    # Save profiling results to a file
    with open("profiling_results.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("Profiling complete. Results saved to 'profiling_results.txt'.")

if __name__ == "__main__":
    # Assuming SDFNet is defined in the same directory or is accessible
    from sdf_model_separable_conv import SDFNet  # Adjust the import as per your file structure

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model and move it to the appropriate device
    model = SDFNet().to(device)

    # Profile the model
    profile_model(model, device=device)