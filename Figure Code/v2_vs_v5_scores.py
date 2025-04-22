import numpy as np

# Define file paths
v6_metrics_path = "C:/Users/nikhi/LabProject/new/v6/v6metrics.val.npz"
v9_metrics_path = "C:/Users/nikhi/LabProject/new/v9/v9metrics.val.npz"

# Define paw nodes (Adjust according to your setup)
paw_nodes = ["left_hind_paw"]  # Add other paw nodes if needed
hind_paw_nodes = ["left_hind_paw"]  # Update if tracking more hind paws

# Function to load and compute mean metrics
def compute_metrics(file_path, model_name):
    # Load the .npz file
    data = np.load(file_path, allow_pickle=True)

    # Extract metrics from correct keys
    mean_loc_error = np.nanmean(data['dist.dists']) if 'dist.dists' in data else "N/A"
    mean_oks = np.nanmean(data['oks.mOKS']) if 'oks.mOKS' in data else "N/A"
    
    # Extract node names and AP/AR values
    if 'node_names' in data and 'oks_voc.AP' in data and 'oks_voc.AR' in data:
        node_names = data['node_names'].tolist()  # Convert to list
        ap_values = data['oks_voc.AP']
        ar_values = data['oks_voc.AR']
        
        # Get indices for paw nodes and hind paw nodes
        paw_indices = [node_names.index(node) for node in paw_nodes if node in node_names]
        hind_paw_indices = [node_names.index(node) for node in hind_paw_nodes if node in node_names]

        # Compute mean AP and AR for paws and hind paws
        mean_map_paws = np.nanmean([ap_values[i] for i in paw_indices]) if paw_indices else "N/A"
        mean_mar_paws = np.nanmean([ar_values[i] for i in paw_indices]) if paw_indices else "N/A"

        mean_map_hind_paws = np.nanmean([ap_values[i] for i in hind_paw_indices]) if hind_paw_indices else "N/A"
        mean_mar_hind_paws = np.nanmean([ar_values[i] for i in hind_paw_indices]) if hind_paw_indices else "N/A"

    else:
        mean_map_paws = mean_mar_paws = mean_map_hind_paws = mean_mar_hind_paws = "N/A"

    # Print results
    print(f"Metrics for {model_name}:")
    print(f"  Mean Localization Error: {mean_loc_error}")
    print(f"  Mean OKS Score: {mean_oks}")
    print(f"  Mean Average Precision (Paws): {mean_map_paws}")
    print(f"  Mean Average Recall (Paws): {mean_mar_paws}")
    print(f"  Mean Average Precision (Hind Paws): {mean_map_hind_paws}")
    print(f"  Mean Average Recall (Hind Paws): {mean_mar_hind_paws}")
    print("\n" + "-"*40 + "\n")

# Compute and print metrics for both models
compute_metrics(v6_metrics_path, "V6 Model")
compute_metrics(v9_metrics_path, "V9 Model")
