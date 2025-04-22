import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing H5 files
directory = "C:/Users/nikhi/LabProject/new/v10/Predictions"

# List all H5 files in the directory
h5_files = [f for f in os.listdir(directory) if f.endswith(".h5")]

video_names = []
max_changes_l_h_paw = []
max_changes_r_h_paw = []

for h5_file in h5_files:
    file_path = os.path.join(directory, h5_file)
    with h5py.File(file_path, "r") as h5_data:
        data = h5_data["tracks"][:]
        
        # Extract y-coordinates
        y_l_h_paw = data[0, 1, 2, :]
        y_r_h_paw = data[0, 1, 3, :]
        
        # Compute max change in y-coordinates
        max_change_l_h_paw = np.nanmax(y_l_h_paw) - np.nanmin(y_l_h_paw)
        max_change_r_h_paw = np.nanmax(y_r_h_paw) - np.nanmin(y_r_h_paw)
        
        video_names.append(h5_file.replace(".h5", ""))  # Remove file extension
        max_changes_l_h_paw.append(max_change_l_h_paw)
        max_changes_r_h_paw.append(max_change_r_h_paw)

# Plot bar chart
x = np.arange(len(video_names))  # X locations for the groups
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, max_changes_l_h_paw, width, label="Left Hind Paw", color='blue')
ax.bar(x + width/2, max_changes_r_h_paw, width, label="Right Hind Paw", color='red')

ax.set_xlabel("Video Name")
ax.set_ylabel("Max Change in Y-Coordinate (Height)")
ax.set_title("Max Change in Height for Left and Right Hind Paws")
ax.set_xticks(x)
ax.set_xticklabels(video_names, rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.show()
