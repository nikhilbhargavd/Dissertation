import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Define file paths for each model version
metrics_files = {
    "v1": r"C:\Users\nikhi\LabProject\new\v5\v5metrics.val.npz",
    "v2": r"C:\Users\nikhi\LabProject\new\v6\v6metrics.val.npz",
    "v3": r"C:\Users\nikhi\LabProject\new\v7\v7metrics.val.npz",
    "v4": r"C:\Users\nikhi\LabProject\new\v8\v8metrics.val.npz",
    "v5": r"C:\Users\nikhi\LabProject\new\v9\v9metrics.val.npz",
    "v6": r"C:\Users\nikhi\LabProject\new\v10\v10metrics.val.npz",
    "v7": r"C:\Users\nikhi\LabProject\new\v11\v11metrics.val.npz",
    "v8": r"C:\Users\nikhi\LabProject\new\v12\v12metrics.val.npz",
}

# Define node names
node_names = [
    "l_f_paw", "r_f_paw", "l_h_paw", "r_h_paw", "snout",
    "l_hip", "l_knee", "l_ear", "r_ear", "r_hip", "r_knee", "back"
]

# Nodes related to paws
paw_nodes = ["l_f_paw", "r_f_paw", "l_h_paw", "r_h_paw"]
hind_paw_nodes = ["l_h_paw", "r_h_paw"]

# Data storage for AP and AR values per node across versions
map_paws = []
mar_paws = []
map_hind_paws = []
mar_hind_paws = []
std_map_paws = []
std_mar_paws = []
std_map_hind_paws = []
std_mar_hind_paws = []
versions = list(metrics_files.keys())

# Process each file to extract mean and standard deviation of AP and AR values for paws and hind paws
for version, file_path in metrics_files.items():
    data = np.load(file_path, allow_pickle=True)["metrics"].item()
    
    # Extract per-node AP and AR values
    ap_values = data.get("oks_voc.AP", None)
    ar_values = data.get("oks_voc.AR", None)
    
    if ap_values is not None and ar_values is not None:
        # Compute mean and standard deviation for AP and AR of paw nodes
        paw_indices = [node_names.index(node) for node in paw_nodes]
        hind_paw_indices = [node_names.index(node) for node in hind_paw_nodes]
        
        mean_map_paws = np.nanmean([ap_values[i] for i in paw_indices])
        std_map_paws.append(np.nanstd([ap_values[i] for i in paw_indices]))
        mean_mar_paws = np.nanmean([ar_values[i] for i in paw_indices])
        std_mar_paws.append(np.nanstd([ar_values[i] for i in paw_indices]))
        
        mean_map_hind_paws = np.nanmean([ap_values[i] for i in hind_paw_indices])
        std_map_hind_paws.append(np.nanstd([ap_values[i] for i in hind_paw_indices]))
        mean_mar_hind_paws = np.nanmean([ar_values[i] for i in hind_paw_indices])
        std_mar_hind_paws.append(np.nanstd([ar_values[i] for i in hind_paw_indices]))
        
        map_paws.append(mean_map_paws)
        mar_paws.append(mean_mar_paws)
        map_hind_paws.append(mean_map_hind_paws)
        mar_hind_paws.append(mean_mar_hind_paws)

# Plot Mean Average Precision of Paws with standard deviation
plt.figure(figsize=(8, 5))
plt.bar(versions, map_paws, yerr=std_map_paws, color='orange', capsize=5)
plt.xlabel("Model Version")
plt.ylabel("Mean Average Precision (mAP)")
plt.title("Mean Average Precision of Paws with Std Dev")
plt.show()

# Plot Mean Average Recall of Paws with standard deviation
plt.figure(figsize=(8, 5))
plt.bar(versions, mar_paws, yerr=std_mar_paws, color='orange', capsize=5)
plt.xlabel("Model Version")
plt.ylabel("Mean Average Recall (mAR)")
plt.title("Mean Average Recall of Paws with Std Dev")
plt.show()

# Plot Mean Average Precision of Hind Paws with standard deviation
plt.figure(figsize=(8, 5))
plt.bar(versions, map_hind_paws, yerr=std_map_hind_paws, color='orange', capsize=5)
plt.xlabel("Model Version")
plt.ylabel("Mean Average Precision (mAP)")
plt.title("Mean Average Precision of Hind Paws")
plt.show()

# Plot Mean Average Recall of Hind Paws with standard deviation
plt.figure(figsize=(8, 5))
plt.bar(versions, mar_hind_paws, yerr=std_mar_hind_paws, color='orange', capsize=5)
plt.xlabel("Model Version")
plt.ylabel("Mean Average Recall (mAR)")
plt.title("Mean Average Recall of Hind Paws")
plt.show()
