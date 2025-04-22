from sleap import Labels
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- File paths ---
slp_path = r"C:\Users\nikhi\LabProject\backuplabels.slp"
h5_path = r"C:\Users\nikhi\LabProject\new\v5\Predictions\8742_l4.mp4.000_8742_l4.analysis.h5"

# --- Load ground truth labels ---
labels = Labels.load_file(slp_path)

# --- Load tracking predictions ---
with h5py.File(h5_path, "r") as f:
    tracks = f["tracks"][:]  # shape: (frames, nodes, 2, tracks)
    pred_data = tracks[:, :, :, 0]  # use first track only
    node_names = [n.decode() for n in f["node_names"][:]]

# --- Compare frame-by-frame ---
results = []

for lf in tqdm(labels.labeled_frames, desc="Comparing frames"):
    frame_idx = lf.frame_idx
    if frame_idx >= pred_data.shape[0]:
        continue

    pred_points = pred_data[frame_idx]  # shape: (nodes, 2)

    for inst in lf.instances:
        for node in inst.skeleton.nodes:
            if node.name not in node_names:
                continue

            node_index = node_names.index(node.name)
            if node_index >= pred_points.shape[0]:
                continue

            true_kp = inst[node.name]
            if not true_kp.visible or np.any(np.isnan(pred_points[node_index])):
                continue

            pred_kp = pred_points[node_index]  # shape: (2,)

            if pred_kp.shape[0] != 2:
                continue

            dist = np.linalg.norm(np.array([true_kp.x, true_kp.y]) - np.array(pred_kp))

            results.append({
                "frame": frame_idx,
                "node": node.name,
                "true_x": true_kp.x,
                "true_y": true_kp.y,
                "pred_x": pred_kp[0],
                "pred_y": pred_kp[1],
                "euclidean_error": dist
            })

# --- Convert to DataFrame ---
df = pd.DataFrame(results)

if not df.empty and "node" in df.columns:
    summary = df.groupby("node")["euclidean_error"].agg(["mean", "std", "max", "count"])
    print("\n=== Per-Node Error Summary ===\n")
    print(summary.round(2))
else:
    print("\nNo valid keypoints found. Please verify frame indices and prediction structure.\n")

# --- Save full results to CSV ---
csv_path = "pose_estimation_errors_v5_vs_ground_truth.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved full error data to: {csv_path}")

# --- Preview the first few rows for debugging ---
print("\nPreview of collected results:")
print(df.head())
