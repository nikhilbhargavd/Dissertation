import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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
}

# Node names and hind paw indices
node_names = [
    "l_f_paw", "r_f_paw", "l_h_paw", "r_h_paw", "snout",
    "l_hip", "l_knee", "l_ear", "r_ear", "r_hip", "r_knee", "back"
]
hind_paw_nodes = ["l_h_paw", "r_h_paw"]
hind_paw_indices = [node_names.index(n) for n in hind_paw_nodes]

# Store raw AP and AR values
raw_ap_hind = []
raw_ar_hind = []
group_labels = []

# Load and collect raw AP/AR scores for hind paws
for version, path in metrics_files.items():
    data = np.load(path, allow_pickle=True)["metrics"].item()
    ap_values = data["oks_voc.AP"]
    ar_values = data["oks_voc.AR"]

    # Extract hind paw scores
    ap_hind = [ap_values[i] for i in hind_paw_indices]
    ar_hind = [ar_values[i] for i in hind_paw_indices]

    raw_ap_hind.extend(ap_hind)
    raw_ar_hind.extend(ar_hind)
    group_labels.extend([version] * len(hind_paw_indices))

# Group AP and AR for ANOVA
grouped_ap = {v: [] for v in metrics_files}
grouped_ar = {v: [] for v in metrics_files}
for i, version in enumerate(group_labels):
    grouped_ap[version].append(raw_ap_hind[i])
    grouped_ar[version].append(raw_ar_hind[i])

# Prepare data for ANOVA
ap_groups = [grouped_ap[v] for v in metrics_files]
ar_groups = [grouped_ar[v] for v in metrics_files]

# Run ANOVA
f_ap, p_ap = stats.f_oneway(*ap_groups)
f_ar, p_ar = stats.f_oneway(*ar_groups)

# Run Tukey post-hoc tests
tukey_ap = pairwise_tukeyhsd(endog=raw_ap_hind, groups=group_labels, alpha=0.05)
tukey_ar = pairwise_tukeyhsd(endog=raw_ar_hind, groups=group_labels, alpha=0.05)

# Calculate means and stds for plotting
versions = list(metrics_files.keys())
mean_ap = [np.mean(grouped_ap[v]) for v in versions]
std_ap = [np.std(grouped_ap[v]) for v in versions]
mean_ar = [np.mean(grouped_ar[v]) for v in versions]
std_ar = [np.std(grouped_ar[v]) for v in versions]

# Plot mAP
plt.figure(figsize=(10, 5))
plt.bar(versions, mean_ap, yerr=std_ap, capsize=5, color='orange')
plt.title(f"Hind Paw mAP by Model (ANOVA p = {p_ap:.4f})")
plt.ylabel("Mean Average Precision (mAP)")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Plot mAR
plt.figure(figsize=(10, 5))
plt.bar(versions, mean_ar, yerr=std_ar, capsize=5, color='orange')
plt.title(f"Hind Paw mAR by Model (ANOVA p = {p_ar:.4f})")
plt.ylabel("Mean Average Recall (mAR)")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Print post-hoc test results
print("Tukey HSD Results for mAP (hind paws):")
print(tukey_ap)

print("\nTukey HSD Results for mAR (hind paws):")
print(tukey_ar)

# After everything else (at the end of your script):

print("\n=== Raw AP Values per Model (Hind Paws) ===")
for v in versions:
    print(f"{v}: {grouped_ap[v]}")

print("\n=== Raw AR Values per Model (Hind Paws) ===")
for v in versions:
    print(f"{v}: {grouped_ar[v]}")

