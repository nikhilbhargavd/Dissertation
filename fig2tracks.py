import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Function to load and process data
def load_and_process_data(base_path):
    xv6 = np.loadtxt(f"{base_path}/xv6.txt")
    yv6 = np.loadtxt(f"{base_path}/yv6.txt")
    xv9 = np.loadtxt(f"{base_path}/xv9.txt")
    yv9 = np.loadtxt(f"{base_path}/yv9.txt")
    
    # Ensure data lengths match before processing
    min_length_v6 = min(len(xv6), len(yv6))
    min_length_v9 = min(len(xv9), len(yv9))
    xv6, yv6 = xv6[:min_length_v6], yv6[:min_length_v6]
    xv9, yv9 = xv9[:min_length_v9], yv9[:min_length_v9]
    
    # Remove NaN values but keep frame indices aligned
    valid_mask_v6 = np.isfinite(xv6) & np.isfinite(yv6)
    valid_mask_v9 = np.isfinite(xv9) & np.isfinite(yv9)
    xv6, yv6 = xv6[valid_mask_v6], yv6[valid_mask_v6]
    xv9, yv9 = xv9[valid_mask_v9], yv9[valid_mask_v9]
    
    return xv6, yv6, xv9, yv9

# List of base directories for different trials
base_dirs = [
    "C:/Users/nikhi/LabProject/8742l5",
    "C:/Users/nikhi/LabProject/8743l3"
]

# Titles for each trial
titles = [
    "Comparison of V2 and V5 Hindpaw Tracks - Mouse 1 Trial 2",
    "Comparison of V2 and V5 Hindpaw Tracks - Mouse 2 Trial 1"
]

# Generate graphs for each dataset
for base_path, title in zip(base_dirs, titles):
    xv6, yv6, xv9, yv9 = load_and_process_data(base_path)
    
    # Plot the combined graph
    plt.figure(figsize=(9, 9))
    plt.plot(xv6, yv6, label='V2 Hindpaw Tracks', color='blue')
    plt.plot(xv9, yv9, label='V5 Hindpaw Tracks', color='red', linestyle='dashed')
    plt.legend(fontsize=14)
    plt.title(title, fontsize=16)
    plt.xlabel("X Position", fontsize=14)
    plt.ylabel("Y Position", fontsize=14)
    plt.ylim(bottom=0)  # Set y-axis to start from 0 at the bottom
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
