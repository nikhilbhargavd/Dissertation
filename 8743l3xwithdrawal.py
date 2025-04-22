import matplotlib.pyplot as plt
import numpy as np

# File paths
file_path_1 = r"C:\Users\nikhi\LabProject\8743l3\xv6.txt"
file_path_2 = r"C:\Users\nikhi\LabProject\8743l3\xv9.txt"

# Load data
data_1 = np.loadtxt(file_path_1, delimiter=',')
data_2 = np.loadtxt(file_path_2, delimiter=',')

# Define frame range
frames = np.arange(2100, 2141)
time_seconds = (frames - frames[0]) / 198  # Convert to seconds

# ---------------- Plot 1: xcoordv68743l3 ----------------
plt.figure(figsize=(10, 6))
plt.plot(time_seconds, data_1, label="XCoord V6", marker='o', linestyle='-')
plt.xlabel("Time (seconds)")
plt.ylabel("X-Coordinate (Normalized)")
plt.title("Model V6 - Left Hind Paw X-Coordinate Over Time (Video 8743_l3)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# ---------------- Plot 2: xcoordv98743l3 ----------------
plt.figure(figsize=(10, 6))
plt.plot(time_seconds, data_2, label="XCoord V9", marker='o', linestyle='-')
plt.xlabel("Time (seconds)")
plt.ylabel("X-Coordinate (Normalized)")
plt.title("Model V9 - Left Hind Paw X-Coordinate Over Time (Mouse 2)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# ---------------- Plot 3: Comparison of xcoordv68743l3 vs xcoordv98743l3 ----------------
plt.figure(figsize=(10, 6))
plt.plot(time_seconds, data_1, label=" V2 (Solid Line)", marker='o', linestyle='-')
plt.plot(time_seconds, data_2, label=" V5 (Dashed Line)", marker='x', linestyle='--')
plt.xlabel("Time Of Paw Withdrawal Onset (s)", fontsize=14)
plt.ylabel("X Coordinate (Pixels)", fontsize=14)
plt.title("Comparison of Model V2 vs V5 - Left Hind Paw X-Coordinate Over Time (Mouse 2 Trial 1)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()
