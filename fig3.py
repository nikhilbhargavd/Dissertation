import os
import numpy as np
import matplotlib.pyplot as plt

# Base directory for saved text files
base_dir = "C:/Users/nikhi/LabProject/fig3"

# List of processed text filenames
txt_filenames = [
    "8742_l4.txt", "8742_l5.txt", "8743_l3.txt", "8743_r1.txt",
    "8745_r1.txt", "8791_r1.txt", "8955_l4.txt", "8956_l1.txt",
]

# Separate files into _l and _r categories
l_files = [file for file in txt_filenames if "_l" in file]
r_files = [file for file in txt_filenames if "_r" in file]

# Split _l files into two groups
l_files_group1 = l_files[:3]  # First three _l files
l_files_group2 = l_files[3:]  # Remaining _l files

# Custom labels for the first graph
custom_labels = ["Mouse 1 Trial 1", "Mouse 1 Trial 2", "Mouse 2 Trial 1"]

# Number of values to ignore at the start
ignore_values = {"Mouse 1 Trial 2": 5, "Mouse 2 Trial 1": 9,"Mouse 1 Trial 1": 1}

# Number of values to remove from the end
truncate_values = {"Mouse 1 Trial 1": 7, "Mouse 1 Trial 2": 4,}

# Function to plot all processed text files while preserving NaN gaps
def plot_txt_files(file_list, title, interpolate_nan=False, custom_labels=None):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Y Coordinate (Pixels)")
    
    for i, filename in enumerate(file_list):
        file_path = os.path.join(base_dir, filename)

        if os.path.exists(file_path):
            # Load the data, preserving NaNs
            data = np.loadtxt(file_path, delimiter=',')
            
            # Assign custom labels if provided
            label = custom_labels[i] if custom_labels and i < len(custom_labels) else filename
            
            # Determine how many values to ignore and truncate
            ignore = ignore_values.get(label, 0)
            truncate = truncate_values.get(label, 0)
            
            # Generate time values (in seconds), considering 198 fps
            frames = np.arange(len(data) - ignore - truncate)  # Adjusted frame numbers
            time_seconds = frames / 198  # Convert frames to seconds
            data = data[ignore:len(data) - truncate]  # Remove initial and final values
            
            if interpolate_nan:
                # Interpolate NaN values to maintain line continuity
                valid_mask = ~np.isnan(data)
                data = np.interp(time_seconds, time_seconds[valid_mask], data[valid_mask])
            
            # Plot each text file
            plt.plot(time_seconds, data, linestyle='-', marker='o', markersize=4, label=label)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Plot first group of _l files with custom labels
if l_files_group1:
    plot_txt_files(l_files_group1, "Height Of Paw Withdrawal", custom_labels=custom_labels)

# Plot second group of _l files
if l_files_group2:
    plot_txt_files(l_files_group2, "Height Of Paw Withdrawal - Group 2")

# Plot _r files with NaN interpolation
if r_files:
    plot_txt_files(r_files, "Height Of Paw Withdrawal", interpolate_nan=True)
