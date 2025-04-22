import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Define file paths for each model version
metrics_files = {
    "v5": r"C:\Users\nikhi\LabProject\new\v5\v5metrics.val.npz",
    "v6": r"C:\Users\nikhi\LabProject\new\v6\v6metrics.val.npz",
    "v7": r"C:\Users\nikhi\LabProject\new\v7\v7metrics.val.npz",
    "v8": r"C:\Users\nikhi\LabProject\new\v8\v8metrics.val.npz",
    "v9": r"C:\Users\nikhi\LabProject\new\v9\v9metrics.val.npz",
    "v10": r"C:\Users\nikhi\LabProject\new\v10\v10metrics.val.npz",
    "v11": r"C:\Users\nikhi\LabProject\new\v11\v11metrics.val.npz"
}

# Define node names based on H5 file
node_names = [
    "Left Front Paw", "Right Front Paw", "Left Hind Paw", "Right Hind Paw", "Snout",
    "Left Hip", "Left Knee", "Left Ear", "Right Ear", "Right Hip", "Right Knee", "Back"
]

# Data storage for AP and AR values per node across versions
ap_per_node = {node: [] for node in node_names}
ar_per_node = {node: [] for node in node_names}
versions = list(metrics_files.keys())

# Process each file to extract per-node AP and AR values
for version, file_path in metrics_files.items():
    data = np.load(file_path, allow_pickle=True)["metrics"].item()
    
    # Extract per-node AP and AR values
    ap_values = data.get("oks_voc.AP", None)
    ar_values = data.get("oks_voc.AR", None)

    # Handle missing values
    if ap_values is not None and ar_values is not None:
        for i in range(len(ap_values)):  # Only loop through available nodes
            ap_per_node[node_names[i]].append(ap_values[i])
            ar_per_node[node_names[i]].append(ar_values[i])
        
        # Fill missing nodes with NaN if they were not included
        for i in range(len(ap_values), len(node_names)):
            ap_per_node[node_names[i]].append(np.nan)
            ar_per_node[node_names[i]].append(np.nan)

# Identify missing nodes (nodes with only NaN values)
missing_nodes = [node for node in node_names if all(np.isnan(ap_per_node[node]))]

# Plot AP values for each node
for node in node_names:
    if node in missing_nodes:
        continue  # Skip missing nodes

    plt.figure(figsize=(8, 5))
    plt.bar(versions, ap_per_node[node], color='orange')
    plt.xlabel("Model Version")
    plt.ylabel("AP (Average Precision)")
    plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
    plt.title(f"Average Precision for {node} Across Models")
    plt.show()

# Plot AR values for each node
for node in node_names:
    if node in missing_nodes:
        continue  # Skip missing nodes

    plt.figure(figsize=(8, 5))
    plt.bar(versions, ar_per_node[node], color='orange')
    plt.xlabel("Model Version")
    plt.ylabel("AR (Average Recall)")
    plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
    plt.title(f"Average Recall for {node} Across Models")
    plt.show()

# Save missing nodes information
if missing_nodes:
    df_missing_nodes = pd.DataFrame({"Missing Nodes": missing_nodes})
    df_missing_nodes.to_excel("missing_nodes_ap_ar.xlsx", index=False)
    print("Missing nodes information saved to missing_nodes_ap_ar.xlsx")
