import h5py
import numpy as np
import matplotlib.pyplot as plt
import os  # To extract file name

# File path
filename = r"C:\Users\nikhi\LabProject\brush_predictions_v4_slp\predictions_v4_copy\8743_r1.mp4.000_8743_r1.analysis.h5"

# Extract the base file name (e.g., '8743_r1') without extension
file_id = os.path.basename(filename).split('.')[0]

# Load the HDF5 file and extract data
with h5py.File(filename, "r") as f:
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

# Debugging: Print file details
print("===filename===")
print(filename)
print("===locations data shape===")
print(locations.shape)
print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")

# Function to filter out NaN or Inf values
def filter_valid_data(node_data):
    valid_mask = np.isfinite(node_data)  # Mask for valid data (finite values)
    return node_data[valid_mask]

# Calculate maximum change in distance for each node
node_distances = {}
for i, node_name in enumerate(node_names):
    print(f"Processing node: {node_name}")
    node_data = locations[:, i, :, :]

    # Filter valid data for X and Y coordinates
    x_data = filter_valid_data(node_data[:, 0, 0])
    y_data = filter_valid_data(node_data[:, 1, 0])

    # Ensure there is enough valid data to calculate distance
    if len(x_data) == 0 or len(y_data) == 0:
        print(f"No valid data for {node_name}, skipping...")
        continue

    # Calculate maximum change in distance for X and Y separately
    x_range = np.ptp(x_data)  # Peak-to-peak distance in X
    y_range = np.ptp(y_data)  # Peak-to-peak distance in Y

    # Combine X and Y ranges as a measure of total distance change
    total_distance = np.sqrt(x_range**2 + y_range**2)
    node_distances[node_name] = total_distance

# Plot bar chart
plt.figure(figsize=(10, 6))
plt.bar(node_distances.keys(), node_distances.values(), color='skyblue')
plt.title(f'Maximum Change in Distance per Node ({file_id})')
plt.xlabel('Node')
plt.ylabel('Maximum Change in Distance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
