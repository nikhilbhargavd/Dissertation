import numpy as np
import os

# Base directory path
base_dir = r"C:\Users\nikhi\LabProject\new"

# Loop through versions v5 to v12
for v in range(5, 13):
    file_path = os.path.join(base_dir, f"v{v}", f"v{v}metrics.val.npz")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    # Load the metrics.val.npz file
    data = np.load(file_path, allow_pickle=True)
    metrics = data["metrics"].item()  # Extract the dictionary

    # Print available keys for reference
    print(f"Processing v{v} - Available keys: {metrics.keys()}")

    # Extract per-node AP and AR values
    per_node_ap = metrics.get("oks_voc.AP", None)
    per_node_ar = metrics.get("oks_voc.AR", None)

    if per_node_ap is not None and per_node_ar is not None:
        print(f"Per-node AP and AR found for v{v}. Saving to file...")
        output_filename = f"per_node_ap_ar_v{v}.npz"
        np.savez(output_filename, ap=per_node_ap, ar=per_node_ar)
        save_path = os.path.abspath(output_filename)
        print(f"Saved as {save_path}")
    else:
        print(f"Per-node AP or AR not found in v{v} metrics file.")
