import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

# Versions to compare
versions = ["v5", "v6", "v7","v8","v9","v10"]
base_path = "C:/Users/nikhi/LabProject/new"
file_name = "8742_l4.mp4.000_8742_l4.analysis.h5"

video_versions = []
max_changes_l_h_paw = []
max_changes_r_h_paw = []

for version in versions:
    file_path = os.path.join(base_path, version, "Predictions", file_name)
    
    if os.path.exists(file_path):
        with h5py.File(file_path, "r") as h5_data:
            data = h5_data["tracks"][:]
            
            # Extract y-coordinates
            y_l_h_paw = data[0, 0, 2, :]
            y_r_h_paw = data[0, 0, 3, :]
            
            # Compute max change in y-coordinates
            max_change_l_h_paw = np.nanmax(y_l_h_paw) - np.nanmin(y_l_h_paw)
            max_change_r_h_paw = np.nanmax(y_r_h_paw) - np.nanmin(y_r_h_paw)
            
            video_versions.append(version)  # Add version name
            max_changes_l_h_paw.append(max_change_l_h_paw)
            max_changes_r_h_paw.append(max_change_r_h_paw)

# Plot bar chart
x = np.arange(len(video_versions))  # X locations for the groups
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, max_changes_l_h_paw, width, label="Left Hind Paw", color='blue')
ax.bar(x + width/2, max_changes_r_h_paw, width, label="Right Hind Paw", color='red')

ax.set_xlabel("Version")
ax.set_ylabel("Max Change in Y-Coordinate (Height)")
ax.set_title("Max Change in Height for Left and Right Hind Paws Across Versions")
ax.set_xticks(x)
ax.set_xticklabels(video_versions, rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.show()

