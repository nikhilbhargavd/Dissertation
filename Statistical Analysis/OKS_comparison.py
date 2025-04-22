import numpy as np
import matplotlib.pyplot as plt
import os

# Define file paths for each model version
metrics_files = {
    "v1": r"C:\Users\nikhi\LabProject\new\v5\v5metrics.val.npz",
    "v2": r"C:\Users\nikhi\LabProject\new\v6\v6metrics.val.npz",
    "v3": r"C:\Users\nikhi\LabProject\new\v7\v7metrics.val.npz",
    "v4": r"C:\Users\nikhi\LabProject\new\v8\v8metrics.val.npz",
    "v5": r"C:\Users\nikhi\LabProject\new\v9\v9metrics.val.npz",
    "v6": r"C:\Users\nikhi\LabProject\new\v10\v10metrics.val.npz",
    "v7": r"C:\Users\nikhi\LabProject\new\v11\v11metrics.val.npz",
    "v8": r"C:\Users\nikhi\LabProject\new\v12\v12metrics.val.npz" 
}

# Data storage for OKS values
oks_scores = {}
std_devs = {}
filtered_oks_scores = {}
filtered_std_devs = {}

# Process each file for OKS extraction
for version, file_path in metrics_files.items():
    data = np.load(file_path, allow_pickle=True)["metrics"].item()
    oks_values = data["oks.mOKS"]  # Extract OKS values

    # Compute mean and standard deviation
    mean_oks_score = np.nanmean(oks_values)
    std_oks_score = np.nanstd(oks_values)

    oks_scores[version] = mean_oks_score
    std_devs[version] = std_oks_score

    # Define per-model outlier thresholds (values beyond 2 standard deviations from the mean)
    threshold_low = mean_oks_score - 2 * std_oks_score
    threshold_high = mean_oks_score + 2 * std_oks_score

    # Filter OKS values within the threshold
    filtered_oks = oks_values[(oks_values >= threshold_low) & (oks_values <= threshold_high)]

    # Compute mean and standard deviation after filtering outliers
    filtered_mean_oks = np.nanmean(filtered_oks)
    filtered_std_oks = np.nanstd(filtered_oks)

    filtered_oks_scores[version] = filtered_mean_oks
    filtered_std_devs[version] = filtered_std_oks

# Convert data to lists for plotting
df_oks = list(oks_scores.items())
df_std = list(std_devs.values())
df_filtered_oks = list(filtered_oks_scores.items())
df_filtered_std = list(filtered_std_devs.values())

# Plot Mean OKS Score (Original) with Standard Deviation
plt.figure(figsize=(8, 5))
plt.bar(*zip(*df_oks), yerr=df_std, capsize=5, color='green', alpha=0.7)
plt.xlabel("Model Version")
plt.ylabel("Mean OKS Score")
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.title("Mean OKS Score Per Model (With Standard Deviation)")
plt.show()

# Plot Mean OKS Score (Filtered Within Each Model) with Standard Deviation
plt.figure(figsize=(8, 5))
plt.bar(*zip(*df_filtered_oks), yerr=df_filtered_std, capsize=5, color='blue', alpha=0.7)
plt.xlabel("Model Version")
plt.ylabel("Mean OKS Score")
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.title("Mean OKS Score Per Model (Filtered With Standard Deviation)")
plt.show()

# Print standard deviation values for each model
print("Standard Deviations for Mean OKS Scores:")
for version, std_value in std_devs.items():
    print(f"{version}: ±{std_value:.4f}")

print("\nStandard Deviations for Filtered OKS Scores:")
for version, std_value in filtered_std_devs.items():
    print(f"{version}: ±{std_value:.4f}")
