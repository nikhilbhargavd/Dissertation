import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths for each model version
metrics_files = {
    "v5": r"C:\Users\nikhi\LabProject\new\v5\v5metrics.val.npz",
    "v6": r"C:\Users\nikhi\LabProject\new\v6\v6metrics.val.npz",
    "v7": r"C:\Users\nikhi\LabProject\new\v7\v7metrics.val.npz",
    "v8": r"C:\Users\nikhi\LabProject\new\v8\v8metrics.val.npz",
    "v9": r"C:\Users\nikhi\LabProject\new\v9\v9metrics.val.npz",
    "v10": r"C:\Users\nikhi\LabProject\new\v10\v10metrics.val.npz",
    "v11": r"C:\Users\nikhi\LabProject\new\v11\v11metrics.val.npz",
    "v12": r"C:\Users\nikhi\LabProject\new\v12\v12metrics.val.npz"
}

# Data storage for mAP and mAR values
map_scores = {}
filtered_map_scores = {}
mar_scores = {}
filtered_mar_scores = {}

# Process each file for mAP and mAR extraction
for version, file_path in metrics_files.items():
    data = np.load(file_path, allow_pickle=True)["metrics"].item()
    map_values = data["oks_voc.mAP"]  # Extract mAP values
    mar_values = data["oks_voc.mAR"]  # Extract mAR values

    # Compute overall mean mAP and mAR scores
    map_scores[version] = np.nanmean(map_values)
    mar_scores[version] = np.nanmean(mar_values)

    # Compute per-model mean and standard deviation
    mean_map = np.nanmean(map_values)
    std_map = np.nanstd(map_values)
    mean_mar = np.nanmean(mar_values)
    std_mar = np.nanstd(mar_values)

    # Define per-model outlier thresholds (values beyond 2 standard deviations from the mean)
    threshold_low_map = mean_map - 2 * std_map
    threshold_high_map = mean_map + 2 * std_map
    threshold_low_mar = mean_mar - 2 * std_mar
    threshold_high_mar = mean_mar + 2 * std_mar

    # Filter mAP and mAR values within the threshold
    filtered_map = map_values[
        (map_values >= threshold_low_map) & (map_values <= threshold_high_map)
    ]
    filtered_mar = mar_values[
        (mar_values >= threshold_low_mar) & (mar_values <= threshold_high_mar)
    ]

    # Compute the new mean after filtering outliers
    filtered_map_scores[version] = np.nanmean(filtered_map)
    filtered_mar_scores[version] = np.nanmean(filtered_mar)

# Convert to DataFrames for plotting
df_map = list(map_scores.items())
df_filtered_map = list(filtered_map_scores.items())
df_mar = list(mar_scores.items())
df_filtered_mar = list(filtered_mar_scores.items())

# Plot Mean Average Precision (mAP) (Original)
plt.figure(figsize=(8, 5))
plt.bar(*zip(*df_map), color='red')
plt.xlabel("Model Version")
plt.ylabel("Mean Average Precision (mAP)")
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.title("Mean Average Precision Across Models (Original)")
plt.show()

# Plot Mean Average Precision (mAP) (Filtered)
plt.figure(figsize=(8, 5))
plt.bar(*zip(*df_filtered_map), color='red')
plt.xlabel("Model Version")
plt.ylabel("Mean Average Precision (mAP)")
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.title("Mean Average Precision Per Model")
plt.show()

# Plot Mean Average Recall (mAR) (Original)
plt.figure(figsize=(8, 5))
plt.bar(*zip(*df_mar), color='red')
plt.xlabel("Model Version")
plt.ylabel("Mean Average Recall (mAR)")
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.title("Mean Average Recall Across Models (Original)")
plt.show()

# Plot Mean Average Recall (mAR) (Filtered)
plt.figure(figsize=(8, 5))
plt.bar(*zip(*df_filtered_mar), color='red')
plt.xlabel("Model Version")
plt.ylabel("Mean Average Recall (mAR)")
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.title("Mean Average Recall Per Model")
plt.show()
