import numpy as np
import os

# Base directory path
base_dir = r"C:\Users\nikhi\LabProject\new"

# Loop through versions v5 to v13
for v in range(5, 14):
    file_path = os.path.join(base_dir, f"v{v}", f"v{v}metrics.val.npz")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    # Load the metrics.val.npz file
    data = np.load(file_path, allow_pickle=True)
    metrics = data["metrics"].item()  # Extract the dictionary

    # Print available keys for reference
    print(f"Processing v{v} - Available keys: {metrics.keys()}")

    # Extract key metrics
    localization_error = metrics.get("dist.dists", None)  # Localization error per keypoint
    mean_oks = metrics.get("oks.mOKS", None)  # Mean OKS score
    precision = metrics.get("vis.precision", None)  # Precision per keypoint
    recall = metrics.get("vis.recall", None)  # Recall per keypoint
    ap = metrics.get("oks_voc.mAP", None)  # Average Precision (AP)
    ar = metrics.get("oks_voc.mAR", None)  # Average Recall (AR)
    pck = metrics.get("pck.mPCK", None)  # Percentage of Correct Keypoints (PCK)

    # Save extracted data in a simplified format with version in the filename
    simplified_filename = f"simplified_metrics_v{v}.npz"
    np.savez(simplified_filename, 
             localization_error=localization_error, 
             mean_oks=mean_oks, 
             precision=precision, 
             recall=recall, 
             ap=ap, 
             ar=ar, 
             pck=pck)

    save_path = os.path.abspath(simplified_filename)
    print(f"Simplified metrics file saved at: {save_path}")
