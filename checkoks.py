import numpy as np

# Load the metrics file
data = np.load(r"C:\Users\nikhi\LabProject\new\v5\v5metrics.val.npz", allow_pickle=True)

# Unpack the 'metrics' dictionary
metrics = data["metrics"].item()

# Print all available metric keys
print("Available metrics keys:", metrics.keys())

# ---- OKS Match Scores ----
if "oks_voc.match_scores" in metrics:
    oks_scores = metrics["oks_voc.match_scores"]
    print("\n✅ OKS match scores found.")
    print("Number of OKS match scores (frames evaluated with OKS):", len(oks_scores))
    print("First 10 OKS match scores:", oks_scores[:10])
else:
    print("\n⚠️ No 'oks_voc.match_scores' found in the metrics file.")

# ---- Mean OKS ----
if "oks.mOKS" in metrics:
    print("\nMean OKS (mOKS):", metrics["oks.mOKS"])

# ---- Localization Errors ----
if "dist.dists" in metrics:
    loc_errors = metrics["dist.dists"]
    print("\n✅ Localization error data found.")
    print("Localization error shape (frames, nodes):", loc_errors.shape)
    print("Total number of localization error values:", loc_errors.size)
else:
    print("\n⚠️ No localization error data ('dist.dists') found in the metrics.")
