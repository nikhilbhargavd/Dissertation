import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Directory for version v6
base_dir = "C:/Users/nikhi/LabProject/new/v6/Predictions"

# List of all h5 filenames
h5_filenames = [
    "8742_l4.mp4.000_8742_l4.analysis.h5", "8742_l5.mp4.000_8742_l5.analysis.h5",
    "8743_l3.mp4.000_8743_l3.analysis.h5", "8955_l4.mp4.000_8955_l4.analysis.h5",
    "8956_l1.mp4.000_8956_l1.analysis.h5",  # First 5 files (l_h_paw)
    "8743_r1.mp4.000_8743_r1.analysis.h5", "8745_r1.mp4.000_8745_r1.analysis.h5",
    "8746_r3.mp4.000_8746_r3.analysis.h5", "8791_r1.mp4.000_8791_r1.analysis.h5",
    "8794_r4.mp4.000_8794_r4.analysis.h5"   # Last 5 files (r_h_paw)
]

# Corresponding frame ranges for each file
frame_ranges = [
    (1540, 1580),
    (1940, 1980),
    (2100, 2140),
    (470, 510),
    (400, 460),
    (2230, 2280),
    (2675, 2715),
    (2255, 2285),
    (1775, 1890),
    (2185, 2225)
]

# Loop through each h5 file and generate a comparison graph
for idx, (filename, (start_frame, end_frame)) in enumerate(zip(h5_filenames, frame_ranges)):
    file_path = os.path.join(base_dir, filename)
    
    if os.path.exists(file_path):
        plt.figure(figsize=(10, 6))
        plt.title(f"X-Coordinate Over Time: {filename}")
        plt.xlabel("Frames")
        plt.ylabel("X-Coordinate (Width)")
        
        with h5py.File(file_path, "r") as h5_file:
            try:
                # Determine the correct index based on the file's position in the list
                node_index = 2 if idx < 5 else 3  # l_h_paw (2) for first 5, r_h_paw (3) for last 5
                
                # Extract X-coordinates for the given node
                x_coords = np.array(h5_file["tracks"][0, 0, node_index, :])  # Change [0, 1, node_index, :] to [0, 0, node_index, :]
                frames = np.arange(len(x_coords))  # Frame numbers

                # Apply frame range
                frames = frames[start_frame:end_frame]
                x_coords = x_coords[start_frame:end_frame]

                # Remove NaNs
                valid_indices = ~np.isnan(x_coords)
                frames = frames[valid_indices]
                x_coords = x_coords[valid_indices]

                # Plot the data
                plt.plot(frames, x_coords, label="v6")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()
            except KeyError:
                print(f"KeyError: 'tracks' dataset missing in {file_path}")
