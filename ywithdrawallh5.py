import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Base directory pattern
base_dir = "C:/Users/nikhi/LabProject/new"
version = "v5"  # Only use one version

# List of all h5 filenames
h5_filenames = [
    "8742_l4.mp4.000_8742_l4.analysis.h5", "8742_l5.mp4.000_8742_l5.analysis.h5",
    "8743_l3.mp4.000_8743_l3.analysis.h5", "8743_r1.mp4.000_8743_r1.analysis.h5",
    "8745_r1.mp4.000_8745_r1.analysis.h5", "8746_r3.mp4.000_8746_r3.analysis.h5",
    "8791_r1.mp4.000_8791_r1.analysis.h5", "8794_r4.mp4.000_8794_r4.analysis.h5",
    "8955_l4.mp4.000_8955_l4.analysis.h5", "8956_l1.mp4.000_8956_l1.analysis.h5"
]

# Separate files into specific groups
left_paw_subset1 = [
    "8742_l4.mp4.000_8742_l4.analysis.h5", "8742_l5.mp4.000_8742_l5.analysis.h5", "8743_l3.mp4.000_8743_l3.analysis.h5"
]
left_paw_subset2 = [
    "8955_l4.mp4.000_8955_l4.analysis.h5", "8956_l1.mp4.000_8956_l1.analysis.h5"
]
specific_r_files = [
    "8743_r1.mp4.000_8743_r1.analysis.h5", "8745_r1.mp4.000_8745_r1.analysis.h5", 
    "8746_r3.mp4.000_8746_r3.analysis.h5", "8791_r1.mp4.000_8791_r1.analysis.h5", 
    "8794_r4.mp4.000_8794_r4.analysis.h5"
]

def plot_h5_files(file_list, title, track_node, frame_range=None):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Y-Coordinate (Height)")
    
    for filename in file_list:
        file_path = f"{base_dir}/{version}/Predictions/{filename}"
        
        if os.path.exists(file_path):
            with h5py.File(file_path, "r") as h5_file:
                try:
                    # Extract Y-coordinates for the specified node
                    y_coords = np.array(h5_file["tracks"][0, 1, track_node, :])
                    frames = np.arange(len(y_coords))  # Frame numbers

                    # Remove NaNs
                    valid_indices = ~np.isnan(y_coords)
                    frames = frames[valid_indices]
                    y_coords = y_coords[valid_indices]

                    # Apply frame range filter if specified
                    if frame_range is not None:
                        range_mask = (frames >= frame_range[0]) & (frames <= frame_range[1])
                        frames = frames[range_mask]
                        y_coords = y_coords[range_mask]

                    # Plot each h5 file on the same graph
                    plt.plot(frames, y_coords, label=f"{filename}")

                except KeyError:
                    print(f"KeyError: 'tracks' dataset missing in {file_path}")
    
    plt.legend()
    plt.gca().invert_yaxis()  # Flip Y-axis so higher values are lower
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Plot left hind paw for subset 1 (Frames 1500-2500)
plot_h5_files(left_paw_subset1, "Y-Coordinate Over Time for Left Hind Paw Subset 1 (v5)", track_node=2, frame_range=(1500, 2500))

# Plot left hind paw for subset 2 (Frames 0-500)
plot_h5_files(left_paw_subset2, "Y-Coordinate Over Time for Left Hind Paw Subset 2 (v5)", track_node=2, frame_range=(0, 500))

# Plot right hind paw for specific right files
plot_h5_files(specific_r_files, "Y-Coordinate Over Time for Specific Right Hind Paw Files (v5)", track_node=3)
