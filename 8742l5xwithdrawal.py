import matplotlib.pyplot as plt
import numpy as np

# File paths
file_path_1 = r"C:\Users\nikhi\LabProject\8742l5\xv6.txt"
file_path_2 = r"C:\Users\nikhi\LabProject\8742l5\xv9.txt"

# Load data
data_1 = np.loadtxt(file_path_1, delimiter=',')
data_2 = np.loadtxt(file_path_2, delimiter=',')

# Define frame range
frames = np.arange(1940, 1981)
time_seconds = (frames - frames[0]) / 198  # Convert to seconds

# ---------------- Plot 1: xcoordV28742l5 ----------------
plt.figure(figsize=(10, 6))
plt.plot(time_seconds, data_1, label="V2", marker='o', linestyle='-')
plt.xlabel("Time (seconds)", fontsize=14)
plt.ylabel("X-Coordinate", fontsize=14)
plt.title("Model V2 - Left Hind Paw X-Coordinate Over Time (Video 8742_l5)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()

# ---------------- Plot 2: xcoordV58742l5 ----------------
plt.figure(figsize=(10, 6))
plt.plot(time_seconds, data_2, label="V5", marker='o', linestyle='-')
plt.xlabel("Time (seconds)", fontsize=14)
plt.ylabel("X-Coordinate", fontsize=14)
plt.title("Model V5 - Left Hind Paw X-Coordinate Over Time (Video 8742_l5)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()

# ---------------- Plot 3: Comparison of xcoordV28742l5 vs xcoordV58742l5 ----------------
plt.figure(figsize=(10, 6))
plt.plot(time_seconds, data_1, label="V2 (Solid Line)", marker='o', linestyle='-')
plt.plot(time_seconds, data_2, label="V5 (Dashed Line)", marker='x', linestyle='--')
plt.xlabel("Time Of Paw Withdrawal Onset (s)", fontsize=14)
plt.ylabel("X Coordinate (Pixels)", fontsize=14)
plt.title("Comparison of Model V2 vs V5 - Left Hind Paw X-Coordinate Over Time (Mouse 1 Trial 2)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()
