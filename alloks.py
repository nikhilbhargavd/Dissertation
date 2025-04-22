import numpy as np
import os
import pandas as pd

# Paths to your models
model_paths = {
    "v1": "C:/Users/nikhi/LabProject/new/v5/v5metrics.val.npz",
    "v2": "C:/Users/nikhi/LabProject/new/v6/v6metrics.val.npz",
    "v3": "C:/Users/nikhi/LabProject/new/v7/v7metrics.val.npz",
    "v4": "C:/Users/nikhi/LabProject/new/v8/v8metrics.val.npz",
    "v5": "C:/Users/nikhi/LabProject/new/v9/v9metrics.val.npz",
    "v6": "C:/Users/nikhi/LabProject/new/v10/v10metrics.val.npz",
    "v7": "C:/Users/nikhi/LabProject/new/v11/v11metrics.val.npz",
    "v8": "C:/Users/nikhi/LabProject/new/v12/v12metrics.val.npz"
}

# Output dictionary
oks_data = {}

# Extract pose-level OKS scores
for model_name, path in model_paths.items():
    try:
        data = np.load(path, allow_pickle=True)
        metrics = data['metrics'].item()
        oks_scores = metrics['oks_voc.match_scores']  # Shape: (n_instances,)
        oks_data[model_name] = oks_scores
        print(f"{model_name}: Loaded {len(oks_scores)} OKS scores.")
    except Exception as e:
        print(f"Error in {model_name}: {e}")

# Convert to DataFrame for easier handling
oks_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in oks_data.items()]))

# Save to CSV
output_path = "C:/Users/nikhi/LabProject/pose_level_oks_scores.csv"
oks_df.to_csv(output_path, index=False)
print(f"\nPose-level OKS scores saved to {output_path}")
