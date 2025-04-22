import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Base directory pattern
base_dir = "C:/Users/nikhi/LabProject/new"
version = "v6"  # Only use one version

# List of all h5 filenames
h5_filenames = [
    "8742_l4.mp4.000_8742_l4.analysis.h5", "8742_l5.mp4.000_8742_l5.analysis.h5",
    "8743_l3.mp4.000_8743_l3.analysis.h5", "8743_r1.mp4.000_8743_r1.analysis.h5",
    "8745_r1.mp4.000_8745_r1.analysis.h5", "8746_r3.mp4.000_8746_r3.analysis.h5",
    "8791_r1.mp4.000_8791_r1.analysis.h5", "8794_r4.mp4.000_8794_r4.analysis.h5",
    "8955_l4.mp4.000_8955_l4.analysis.h5", "8956_l1.mp4.000_8956_l1.analysis.h5"
]

# Separate files into left (_l) and right (_r) groups
l_files = [f for f in h5_filenames if "_l" in f]
r_files = [f for f in h5_filenames if "_r" in f]

def plot_h5_files(file_list, title, node_index):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Y-Coordinate (Height)")
    
    for filename in file_list:
        file_path = f"{base_dir}/{version}/Predictions/{filename}"
        
        if os.path.exists(file_path):
            with h5py.File(file_path, "r") as h5_file:
                try:
                    # Extract Y-coordinates based on whether it's a left or right paw file
                    y_coords = np.array(h5_file["tracks"][0, 1, node_index, :])
                    frames = np.arange(len(y_coords))  # Frame numbers

                    # Remove NaNs
                    valid_indices = ~np.isnan(y_coords)
                    frames = frames[valid_indices]
                    y_coords = y_coords[valid_indices]

                    # Plot each h5 file on the same graph
                    plt.plot(frames, y_coords, label=f"{filename}")

                except KeyError:
                    print(f"KeyError: 'tracks' dataset missing in {file_path}")
    
    plt.legend()
    plt.gca().invert_yaxis()  # Flip Y-axis so higher values are lower
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Plot left paw files using node index 2 (l_h_paw)
plot_h5_files(l_files, "Y-Coordinate Over Time for Left Paw Files (v5)", node_index=2)

# Plot right paw files using node index 3 (r_h_paw)
plot_h5_files(r_files, "Y-Coordinate Over Time for Right Paw Files (v5)", node_index=3)

def plot_8956_l1():
    filename = "8956_l1.mp4.000_8956_l1.analysis.h5"
    file_path = f"{base_dir}/{version}/Predictions/{filename}"

    if os.path.exists(file_path):
        with h5py.File(file_path, "r") as h5_file:
            try:
                # Extract Y-coordinates for l_h_paw (node index 2) for frames 2000-2500
                start_frame, end_frame = 340, 400
                y_coords = np.array(h5_file["tracks"][0, 1, 2, start_frame:end_frame+1])
                frames = np.arange(start_frame, end_frame+1)

                # Remove NaNs
                valid_indices = ~np.isnan(y_coords)
                frames = frames[valid_indices]
                y_coords = y_coords[valid_indices]

                # Plot the specific file
                plt.figure(figsize=(10, 6))
                plt.plot(frames, y_coords, marker='o', linestyle='-', label="8743_r1 Y-Coordinate")

                # Labels and title
                plt.xlabel("Frames")
                plt.ylabel("Y-Coordinate (Height)")
                plt.title("Y-Coordinate Over Time for 8745_r1 (Frames 2675-2715)")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                plt.gca().invert_yaxis()

                # Show the plot
                plt.show()

            except KeyError:
                print(f"KeyError: 'tracks' dataset missing in {file_path}")

# Call function to plot 8743_r1
plot_8956_l1()
