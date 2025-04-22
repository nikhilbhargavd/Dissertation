# Adjusted Code: Y-Axis Starts from 0 at Bottom, Updated Labels

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load data from uploaded text files
xv6 = np.loadtxt("C:/Users/nikhi/LabProject/8742l5/xv6.txt")
yv6 = np.loadtxt("C:/Users/nikhi/LabProject/8742l5/yv6.txt")
xv9 = np.loadtxt("C:/Users/nikhi/LabProject/8742l5/xv9.txt")
yv9 = np.loadtxt("C:/Users/nikhi/LabProject/8742l5/yv9.txt")

# Define frame range
start_frame = 2075
end_frame = 2150
frame_indices = np.arange(start_frame, end_frame + 1)

# Function to smooth and calculate velocity
def smooth_diff(node_loc, win=25, poly=3):
    if len(node_loc) < win or len(node_loc) < poly + 1:
        return np.zeros(len(node_loc))
    
    node_loc_vel = np.zeros_like(node_loc)
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)
    return np.linalg.norm(node_loc_vel, axis=1)

# Ensure data lengths match before processing
min_length_6 = min(len(xv6), len(yv6))
min_length_9 = min(len(xv9), len(yv9))

xv6, yv6 = xv6[:min_length_6], yv6[:min_length_6]
xv9, yv9 = xv9[:min_length_9], yv9[:min_length_9]

# Graph Titles in Required Order
graph_titles = [
    "V6 Position Over Time", "V6 Hindpaw Tracks", "V6 Coloured By Velocity",
    "V9 Position Over Time", "V9 Hindpaw Tracks", "V9 Coloured By Velocity"
]

plot_index = 0

# Process and visualize both sets of data
for (x_data, y_data, label) in [(xv6, yv6, 'V6'), (xv9, yv9, 'V9')]:

    # Remove NaN values but keep frame indices aligned
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data, y_data = x_data[valid_mask], y_data[valid_mask]
    valid_frames = frame_indices[:len(x_data)]  # Adjusting frame indices

    # Calculate velocity
    node_velocity = smooth_diff(np.column_stack((x_data, y_data)))
    node_velocity = node_velocity[:len(x_data)]

    # === Graph 1: Time-Series of X and Y Positions ===
    plt.figure()
    plt.plot(valid_frames, x_data, label='X Coordinate Path', color='blue')
    plt.plot(valid_frames, y_data, label='Y Coordiante Path', color='green')
    plt.legend()
    plt.title(graph_titles[plot_index])
    plt.xlabel("Frames")
    plt.ylabel("Position")
    plt.ylim(bottom=0)  # Set y-axis to start from 0 at the bottom
    plt.show()
    plot_index += 1

    # === Graph 2: XY Tracks (Scatter Plot) ===
    plt.figure(figsize=(7, 7))
    plt.plot(x_data, y_data, label='Paw Tracks', color='orange')
    plt.legend()
    plt.title(graph_titles[plot_index])
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.ylim(bottom=0)  # Set y-axis to start from 0 at the bottom
    plt.show()
    plot_index += 1

    # === Graph 3: XY Scatter Plot Colored by Velocity ===
    plt.figure(figsize=(7, 7))
    plt.scatter(x_data, y_data, c=node_velocity, s=18, vmin=0, vmax=8, cmap='viridis')
    plt.colorbar(label='Velocity')
    plt.title(graph_titles[plot_index])
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.ylim(bottom=0)  # Set y-axis to start from 0 at the bottom
    plt.show()
    plot_index += 1
