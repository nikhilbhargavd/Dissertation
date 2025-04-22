import sleap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths for each version
file_paths = {
    "v6": r"C:\Users\nikhi\LabProject\new\v6\v6metrics.val.npz",
    "v9": r"C:\Users\nikhi\LabProject\new\v9\v9metrics.val.npz"
}

# Dictionary to store extracted metrics for comparison
metrics_comparison = {}

# Loop through each version and evaluate
for version, file_path in file_paths.items():
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    print(f"\nEvaluating model for {version}...\n")

    metrics = sleap.load_metrics(file_path, split="val")

    # Extract mean localization error and mean OKS score
    mean_localization_error = np.mean(metrics["dist.dists"])
    mean_oks_score = np.mean(metrics["oks_voc.match_scores"])

    # Store metrics for comparison
    metrics_comparison[version] = {
        "Error distance (50%)": metrics["dist.p50"],
        "Error distance (90%)": metrics["dist.p90"],
        "Error distance (95%)": metrics["dist.p95"],
        "mAP": metrics["oks_voc.mAP"],
        "mAR": metrics["oks_voc.mAR"],
        "Mean Localization Error": mean_localization_error,
        "Mean OKS Score": mean_oks_score,
    }

    # Print extracted values
    print("Error distance (50%):", metrics["dist.p50"])
    print("Error distance (90%):", metrics["dist.p90"])
    print("Error distance (95%):", metrics["dist.p95"])
    print("mAP:", metrics["oks_voc.mAP"])
    print("mAR:", metrics["oks_voc.mAR"])
    print("Mean Localization Error:", mean_localization_error)
    print("Mean OKS Score:", mean_oks_score)

    # Plot localization error distribution
    plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
    sns.histplot(metrics["dist.dists"].flatten(), binrange=(0, 20), kde=True, kde_kws={"clip": (0, 20)}, stat="probability")
    plt.xlabel("Localization error (px)")
    plt.title(f"Localization Error Distribution - {version}")
    plt.show()

    # Plot OKS match scores
    plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
    sns.histplot(metrics["oks_voc.match_scores"].flatten(), binrange=(0, 1), kde=True, kde_kws={"clip": (0, 1)}, stat="probability")
    plt.xlabel("Object Keypoint Similarity")
    plt.title(f"OKS Match Scores - {version}")
    plt.show()

    # Precision-recall curve
    plt.figure(figsize=(4, 4), dpi=150, facecolor="w")
    for precision, thresh in zip(metrics["oks_voc.precisions"][::2], metrics["oks_voc.match_score_thresholds"][::2]):
        plt.plot(metrics["oks_voc.recall_thresholds"], precision, "-", label=f"OKS @ {thresh:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.title(f"Precision-Recall Curve - {version}")
    plt.show()

# Compute and print the difference between v6 and v9
if "v6" in metrics_comparison and "v9" in metrics_comparison:
    print("\n=== Comparison of v6 and v9 ===\n")
    for key in metrics_comparison["v6"].keys():
        diff = metrics_comparison["v9"][key] - metrics_comparison["v6"][key]
        print(f"Difference in {key}: {diff:.4f}")
