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

# Nodes to track (assuming they are indexed from 0 to 11)
nodes = ["nose", "right_ear", "left_ear", "neck", "right_fore_paw", "left_fore_paw", 
         "right_hind_paw", "left_hind_paw", "tail_base", "tail_mid", "tail_tip", "spine"]

# Loop through each h5 file
for filename in h5_filenames:
    for node_idx, node_name in enumerate(nodes):
        for coord, axis_label in zip([0, 1], ['X', 'Y']):  # 0 = X, 1 = Y
            plt.figure(figsize=(10, 6))
            plt.title(f"{axis_label}-Coordinate Over Time: {filename} ({node_name})")
            plt.xlabel("Frames")
            plt.ylabel(f"{axis_label}-Coordinate")
            
            # Loop through versions
            for version in versions:
                file_path = f"{base_dir}/{version}/Predictions/{filename}"
                
                if os.path.exists(file_path):
                    with h5py.File(file_path, "r") as h5_file:
                        try:
                            # Extract X or Y coordinates for the current node
                            coords = np.array(h5_file["tracks"][0, coord, node_idx, :])
                            frames = np.arange(len(coords))  # Frame numbers

                            # Remove NaNs
                            valid_indices = ~np.isnan(coords)
                            frames = frames[valid_indices]
                            coords = coords[valid_indices]

                            # Plot each version on the same graph
                            plt.plot(frames, coords, label=f"{version}")
                        
                        except KeyError:
                            print(f"KeyError: 'tracks' dataset missing in {file_path}")
            
            plt.legend()
            if axis_label == 'Y':
                plt.gca().invert_yaxis()  # Flip Y-axis so higher values are lower
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()
