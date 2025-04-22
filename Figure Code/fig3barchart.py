import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os

# Define model versions and their corresponding labels
model_versions = ["v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"]
model_labels = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]

# Define base directory
base_dir = "C:/Users/nikhi/LabProject/new"

# Define file identifiers and corresponding frame ranges
file_info = {
    "8742_l4": {"title": "Mouse 1 Trial 1", "frame_range": (1540, 1580)},
    "8742_l5": {"title": "Mouse 1 Trial 2", "frame_range": (1940, 1980)},
    "8743_l3": {"title": "Mouse 2 Trial 1", "frame_range": (2100, 2140)}
}

# Function to extract max change in Y coordinate
def get_max_y_change(file_path, frame_range):
    try:
        with h5py.File(file_path, 'r') as h5_file:
            y_data = np.array(h5_file['tracks'])[0, 1, 2, :]
            
            # Ensure frame range is within bounds
            start_frame, end_frame = frame_range
            if end_frame > len(y_data):
                end_frame = len(y_data)
            
            # Extract relevant frame range
            y_selected = y_data[start_frame:end_frame]
            
            # Compute max change in Y coordinate
            max_change = np.nanmax(y_selected) - np.nanmin(y_selected)
            return max_change
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.nan

# Process each file type
for file_id, info in file_info.items():
    max_y_changes = []
    
    for version in model_versions:
        file_path = os.path.join(base_dir, version, "Predictions", f"{file_id}.mp4.000_{file_id}.analysis.h5")
        max_y_change = get_max_y_change(file_path, info["frame_range"])
        max_y_changes.append(max_y_change)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.bar(model_labels, max_y_changes, color='blue')
    plt.xlabel("Model Version")
    plt.ylabel("Maximum Change in Y Coordinate")
    plt.title(f"{info['title']} - Maximum Change in Y Coordinate")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
