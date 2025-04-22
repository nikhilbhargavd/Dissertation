import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os  # To extract file name

# File path
filename = r"C:\Users\nikhi\LabProject\new\v10\Predictions\8743_l3.mp4.000_8743_l3.analysis.h5"

# Extract the base file name (e.g., '8743_r1') without extension
file_id = os.path.basename(filename).split('.')[0]

# Load the HDF5 file and extract data
with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

# Set node indices
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

# Define frame range
start_frame = 2075
end_frame = 2150

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

# Process and visualize each node
for node_name, node_index in node_indices.items():
    print(f"Processing node: {node_name}")

    # Select only the specified frame range
    node_data = locations[start_frame:end_frame + 1, node_index, :, :]

    # Extract X and Y coordinates while keeping valid indices
    x_data = node_data[:, 0, 0]
    y_data = node_data[:, 1, 0]  # Keep Y-values the same

    # Remove NaN values but keep frame indices aligned
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    frame_indices = np.arange(start_frame, end_frame + 1)[valid_mask]

    # Check if sufficient valid data is available
    if len(x_data) == 0 or len(y_data) == 0:
        print(f"No valid data for {node_name}, skipping...")
        continue

    # Calculate velocity with only valid data
    node_velocity = smooth_diff(np.column_stack((x_data, y_data)))

    # Ensure velocity array matches valid frames
    node_velocity = node_velocity[:len(x_data)]

    # === Graph 1: Time-Series of X and Y Positions ===
    plt.figure()
    plt.plot(frame_indices, x_data, label=f'{node_name} X', color='blue')
    plt.plot(frame_indices, y_data, label=f'{node_name} Y', color='green')
    plt.legend()
    plt.title(f'{node_name.capitalize()} Locations Over Time ({file_id})')
    plt.xlabel("Frames")
    plt.ylabel("Position")
    plt.gca().invert_yaxis()  # Flip Y-axis
    plt.show()

    # === Graph 2: XY Tracks (Scatter Plot) ===
    plt.figure(figsize=(7, 7))
    plt.plot(x_data, y_data, label=f'{node_name} Tracks', color='orange')
    plt.legend()
    plt.title(f'{node_name.capitalize()} Tracks ({file_id})')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()  # Flip Y-axis
    plt.show()

    # === Graph 3: XY Scatter Plot Colored by Velocity ===
    plt.figure(figsize=(7, 7))
    plt.scatter(x_data, y_data, c=node_velocity, s=4, vmin=0, vmax=10, cmap='viridis')
    plt.colorbar(label='Velocity')
    plt.title(f'{node_name.capitalize()} Tracks Colored by Velocity ({file_id})')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()  # Flip Y-axis
    plt.show()
