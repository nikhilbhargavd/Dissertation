import matplotlib.pyplot as plt
import numpy as np

# File paths
file_path_1 = r"C:\Users\nikhi\LabProject\8742l5\yv6.txt"
file_path_2 = r"C:\Users\nikhi\LabProject\8742l5\yv9.txt"

# Load data
data_1 = np.loadtxt(file_path_1, delimiter=',')
data_2 = np.loadtxt(file_path_2, delimiter=',')

# Define frame range
frames = np.arange(1940, 1980)
time_seconds = (frames - frames[0]) / 198  # Convert to seconds

# ---------------- Plot 1: v6_8742l5 ----------------
plt.figure(figsize=(10, 6))
plt.plot(time_seconds, data_1, label="V6", marker='o', linestyle='-')
plt.xlabel("Time (seconds)")
plt.ylabel("Height")
plt.title("Model V6 - Left Hind Paw Height Over Time (Video 8742_l5)")
plt.ylim(0, 40)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# ---------------- Plot 2: v9_8742l5 ----------------
plt.figure(figsize=(10, 6))
plt.plot(time_seconds, data_2, label="V9", marker='o', linestyle='-')
plt.xlabel("Time (seconds)")
plt.ylabel("Height")
plt.title("Model V9 - Left Hind Paw Height Over Time (Video 8742_l5)")
plt.ylim(0, 40)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# ---------------- Plot 3: Comparison of v6_8742l5 vs v9_8742l5 ----------------
plt.figure(figsize=(10, 6))
plt.plot(time_seconds, data_1, label="V2 (Solid Line)", marker='o', linestyle='-')
plt.plot(time_seconds, data_2, label="V5 (Dashed Line)", marker='x', linestyle='--')
plt.xlabel("Time Of Paw Withdrawal Onset (s)", fontsize=14)
plt.ylabel("Y Coordinate (Pixels)", fontsize=14)
plt.title("Comparison of Height - Model V2 vs V5 (Mouse 1 Trial 2)", fontsize=16)
plt.ylim(0, 40)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()
