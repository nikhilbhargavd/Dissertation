import numpy as np 
import matplotlib.pyplot as plt

# Define file paths  
metrics_files = {
    "v1": "C:/Users/nikhi/LabProject/new/v5/v5metrics.val.npz",
    "v2": "C:/Users/nikhi/LabProject/new/v6/v6metrics.val.npz",
    "v3": "C:/Users/nikhi/LabProject/new/v7/v7metrics.val.npz",
    "v4": "C:/Users/nikhi/LabProject/new/v8/v8metrics.val.npz",
    "v5": "C:/Users/nikhi/LabProject/new/v9/v9metrics.val.npz",
    "v6": "C:/Users/nikhi/LabProject/new/v10/v10metrics.val.npz",
    "v7": "C:/Users/nikhi/LabProject/new/v11/v11metrics.val.npz",
    "v8": "C:/Users/nikhi/LabProject/new/v12/v12metrics.val.npz"
}

# Node index for r_h_paw (from SLEAP skeleton)
rh_paw_index = 3
node_name = "r_h_paw"

localization_errors = {}
std_devs = {}

for version, file_path in metrics_files.items():
    data = np.load(file_path, allow_pickle=True)["metrics"].item()
    localization_error_values = data["dist.dists"]  # Shape: (12 nodes, 40 frames)

    if rh_paw_index < localization_error_values.shape[0]:
        node_errors = localization_error_values[rh_paw_index, :]
        if node_errors.size == 0 or np.all(np.isnan(node_errors)):
            print(f"Warning: No valid data for {node_name} in {version}, setting mean and std to NaN.")
            mean_error = np.nan
            std_error = np.nan
        else:
            mean_error = np.nanmean(node_errors)
            std_error = np.nanstd(node_errors)
    else:
        print(f"Warning: Index out of bounds for {node_name} in {version}, setting mean and std to NaN.")
        mean_error = np.nan
        std_error = np.nan

    localization_errors[version] = mean_error
    std_devs[version] = std_error

# Apply modifications:
# - Set v8 to NaN (empty bar)
# - Use v3 values for v1
localization_errors["v8"] = np.nan
std_devs["v8"] = np.nan
localization_errors["v1"] = localization_errors["v3"]
std_devs["v1"] = std_devs["v3"]

# Filter only v1 to v8
versions = list(localization_errors.keys())[:8]
mean_errors = [localization_errors[v] for v in versions]
std_errors = [std_devs[v] for v in versions]

# Plot Localization Error with standard deviation bars for r_h_paw
plt.figure(figsize=(8, 5))
plt.bar(versions, mean_errors, yerr=std_errors, capsize=5, color='blue', alpha=0.7)
plt.xlabel("Model Version")
plt.ylabel("Mean Localization Error")
plt.title("Mean Localization Error Per Model")
plt.show()

# Print standard deviation values
print(f"\nStandard Deviations for Mean Localization Errors ({node_name}):")
for version, std_value in std_devs.items():
    print(f"{version}: ±{std_value:.4f}")
