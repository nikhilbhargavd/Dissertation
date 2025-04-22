import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os  # To extract file name

# File path
filename = r"C:\Users\nikhi\LabProject\new\v6\Predictions\8742_l4.mp4.000_8742_l4.analysis.h5"

# Extract the base file name (e.g., '8743_r1') without extension
file_id = os.path.basename(filename).split('.')[0]

# Load the HDF5 file and extract data
with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

# Set indices for nodes to process (from node_names)
node_indices = {
    "l_f_paw": 0,
    "r_f_paw": 1,
    "l_h_paw": 2,
    "r_h_paw": 3,
    "snout": 4,
    "l_hip": 5,
    "l_knee": 6,
    "l_ear": 7,
    "r_ear": 8,
    "r_hip": 9,
    "r_knee": 10,
    "back": 11,
}

# Function to filter out NaN or Inf values
def filter_valid_data(node_data):
    valid_mask = np.isfinite(node_data)  # Mask for valid data (finite values)
    return node_data[valid_mask]

# Function to smooth and calculate velocity
def smooth_diff(node_loc, win=25, poly=3):
    if len(node_loc) < win or len(node_loc) < poly + 1:
        return np.zeros(len(node_loc))
    node_loc_vel = np.zeros_like(node_loc)
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)
    return np.linalg.norm(node_loc_vel, axis=1)

# Visualization settings
sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15, 6]

num_nodes = len(node_indices)

# Generate each graph separately
for plot_idx in range(3):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (node_name, node_index) in enumerate(node_indices.items()):
        if i >= 12:
            break  # Ensure we don't exceed subplot limits
        node_data = locations[:, node_index, :, :]
        x_data = filter_valid_data(node_data[:, 0, 0])
        y_data = filter_valid_data(node_data[:, 1, 0])
        if len(x_data) == 0 or len(y_data) == 0:
            continue
        node_velocity = smooth_diff(np.column_stack((x_data, y_data)))
        
        if plot_idx == 0:
            axes[i].plot(x_data, label='X', color='blue')
            axes[i].plot(y_data, label='Y', color='green')
            axes[i].legend()
            axes[i].set_title(f'{node_name} Locations Over Time')
            axes[i].set_xlabel("Frames")
            axes[i].set_ylabel("Position")
        elif plot_idx == 1:
            axes[i].plot(x_data, y_data, label=f'{node_name} Tracks', color='orange')
            axes[i].legend()
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f'{node_name} Tracks')
            axes[i].set_xlabel("X Position")
            axes[i].set_ylabel("Y Position")
        elif plot_idx == 2:
            scatter = axes[i].scatter(x_data, y_data, c=node_velocity[:len(x_data)], s=4, vmin=0, vmax=10, cmap='viridis')
            fig.colorbar(scatter, ax=axes[i], label='Velocity')
            axes[i].set_title(f'{node_name} Tracks Colored by Velocity')
            axes[i].set_xlabel("X Position")
            axes[i].set_ylabel("Y Position")
    
    plt.tight_layout()
    plt.show()
