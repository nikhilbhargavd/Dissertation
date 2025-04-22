import numpy as np
import pandas as pd
from scipy.stats import f_oneway

# Local paths to your metrics files
metrics_files = {
    "v1": r"C:\Users\nikhi\LabProject\new\v5\v5metrics.val.npz",
    "v2": r"C:\Users\nikhi\LabProject\new\v6\v6metrics.val.npz",
    "v3": r"C:\Users\nikhi\LabProject\new\v7\v7metrics.val.npz",
    "v4": r"C:\Users\nikhi\LabProject\new\v8\v8metrics.val.npz",
    "v5": r"C:\Users\nikhi\LabProject\new\v9\v9metrics.val.npz",
    "v6": r"C:\Users\nikhi\LabProject\new\v10\v10metrics.val.npz",
    "v7": r"C:\Users\nikhi\LabProject\new\v11\v11metrics.val.npz",
    "v8": r"C:\Users\nikhi\LabProject\new\v12\v12metrics.val.npz",
}

# Node names and index for left hind paw
node_names = [
    "l_f_paw", "r_f_paw", "l_h_paw", "r_h_paw", "snout",
    "l_hip", "l_knee", "l_ear", "r_ear", "r_hip", "r_knee", "back"
]
l_h_paw_index = node_names.index("l_h_paw")

# Collect per-sample localisation errors
all_errors = {}

for version, path in metrics_files.items():
    metrics = np.load(path, allow_pickle=True)["metrics"].item()
    if "dist.dists" in metrics:
        dists = metrics["dist.dists"]
        if dists.ndim == 2 and dists.shape[1] > l_h_paw_index:
            all_errors[version] = dists[:, l_h_paw_index]

# One-way ANOVA
anova_result = f_oneway(*all_errors.values())
print(f"ANOVA F-statistic: {anova_result.statistic:.4f}, p-value: {anova_result.pvalue:.4f}")

# Save values for plotting or Tukey test
df_long = pd.DataFrame([(k, v) for k, arr in all_errors.items() for v in arr],
                       columns=["Model", "Localisation Error"])
df_long.to_csv("lhp_localisation_errors.csv", index=False)
print("Saved: lhp_localisation_errors.csv")
