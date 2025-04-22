import numpy as np

# Define file paths for v2 and v5
metrics_files = {
    "v2": r"C:\Users\nikhi\LabProject\new\v6\v6metrics.val.npz",
    "v5": r"C:\Users\nikhi\LabProject\new\v9\v9metrics.val.npz"
}

# Define node names
node_names = [
    "l_f_paw", "r_f_paw", "l_h_paw", "r_h_paw", "snout",
    "l_hip", "l_knee", "l_ear", "r_ear", "r_hip", "r_knee", "back"
]

# Nodes related to hind paws
hind_paw_nodes = ["l_h_paw", "r_h_paw"]

# Storage for extracted metrics
results = {}

# Process each file
for version, file_path in metrics_files.items():
    data = np.load(file_path, allow_pickle=True)["metrics"].item()
    
    # Extract localization error
    localization_error = np.nanmean(data["dist.dists"])
    
    # Extract mAP and mAR for hind paws
    ap_values = data.get("oks_voc.AP", None)
    ar_values = data.get("oks_voc.AR", None)
    hind_paw_indices = [node_names.index(node) for node in hind_paw_nodes]
    
    map_hind_paw = np.nanmean([ap_values[i] for i in hind_paw_indices])
    mar_hind_paw = np.nanmean([ar_values[i] for i in hind_paw_indices])
    
    # Extract OKS
    oks_score = np.nanmean(data["oks.mOKS"])
    
    # Store results
    results[version] = {
        "Localization Error": localization_error,
        "mAP Hind Paw": map_hind_paw,
        "mAR Hind Paw": mar_hind_paw,
        "OKS": oks_score
    }

# Calculate differences between v5 and v2
differences = {
    metric: results["v5"][metric] - results["v2"][metric]
    for metric in results["v2"].keys()
}

# Print results
print("Metrics for v2:")
for key, value in results["v2"].items():
    print(f"{key}: {value:.4f}")

print("\nMetrics for v5:")
for key, value in results["v5"].items():
    print(f"{key}: {value:.4f}")

print("\nDifferences (v5 - v2):")
for key, value in differences.items():
    print(f"{key}: {value:.4f}")
