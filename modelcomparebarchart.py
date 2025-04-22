import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Base directory pattern
base_dir = "C:/Users/nikhi/LabProject/new"
versions = [f"v{i}" for i in range(5, 14)]  # v5 to v13

# List of all h5 filenames
h5_filenames = [
    "8742_l4.mp4.000_8742_l4.analysis.h5", "8742_l5.mp4.000_8742_l5.analysis.h5",
    "8743_l3.mp4.000_8743_l3.analysis.h5", "8743_r1.mp4.000_8743_r1.analysis.h5",
    "8745_r1.mp4.000_8745_r1.analysis.h5", "8746_r3.mp4.000_8746_r3.analysis.h5",
    "8791_r1.mp4.000_8791_r1.analysis.h5", "8794_r4.mp4.000_8794_r4.analysis.h5",
    "8955_l4.mp4.000_8955_l4.analysis.h5", "8956_l1.mp4.000_8956_l1.analysis.h5"
]

# Corresponding frame ranges 
frame_ranges = [
    (1540, 1580), (1940, 1980), (2100, 2140), (2230, 2280), (2675, 2715),
    (2255, 2285), (1775, 1890), (2185, 2225), (470, 510), (400, 460)
]

# Loop through each h5 file and generate a comparison graph
for filename, (start_frame, end_frame) in zip(h5_filenames, frame_ranges):
    max_height_changes = []
    valid_versions = []

    print(f"Processing {filename} (Frames {start_frame}-{end_frame})")

    for version in versions:
        file_path = f"{base_dir}/{version}/Predictions/{filename}"

        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as f:
                # Extract left hind paw y-coordinates
                try:
                    y_coords = np.array(f['tracks'][0, 1, 2, :])

                    # Extract only values within the frame range
                    if y_coords.shape[0] > end_frame:
                        y_coords = y_coords[start_frame:end_frame]
                    else:
                        print(f"Warning: Frame range exceeds data in {file_path}")
                        continue  # Skip this version if frames are missing

                    # Remove NaN values
                    valid_y_coords = y_coords[~np.isnan(y_coords)]  

                    # Compute max height change
                    if valid_y_coords.size > 0:
                        max_change = np.max(valid_y_coords) - np.min(valid_y_coords)
                    else:
                        max_change = np.nan  # If all values are NaN, store NaN

                    # Store results
                    valid_versions.append(version)
                    max_height_changes.append(max_change)
                except KeyError:
                    print(f"KeyError: 'tracks' dataset not found in {file_path}")

    # Plot results for this specific h5 file
    plt.figure(figsize=(10, 6))
    plt.bar(valid_versions, max_height_changes, color='b', alpha=0.7)
    plt.xlabel("Model Version")
    plt.ylabel("Max Height Change (pixels)")
    plt.title(f"Max Change in Left Hind Paw Height ({start_frame}-{end_frame})\n{filename}")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
