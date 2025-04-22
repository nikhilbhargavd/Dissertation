import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

# Define your data
data = {
    "v1": [0.6234620278666237, 0.5328692310172081],
    "v2": [0.8823305407463824, 0.8195932063344891],
    "v3": [0.822750844515952, 0.6695104302978648],
    "v4": [0.7621620270135121, 0.6227735949270603],
    "v5": [0.3074582834223272, 0.16942220537843253],
    "v6": [0.7969850383289523, 0.7439611043775284],
    "v7": [0.7271473831733307, 0.5834803896815363]
}

# Convert to long-form DataFrame
df_long = pd.DataFrame([(k, v) for k, vals in data.items() for v in vals], columns=['Version', 'mAP'])

# Games-Howell post-hoc test
games_howell = pg.pairwise_gameshowell(data=df_long, dv='mAP', between='Version')
print("Games-Howell Results:\n", games_howell)

# Plot bar chart with error bars
summary = df_long.groupby('Version')['mAP'].agg(['mean', 'std']).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=summary, x='Version', y='mean', yerr=summary['std'], capsize=0.2, color='orange')
plt.ylabel("Mean Average Precision (mAP)")
plt.title("Games-Howell Test: Hind Paw mAP Across Versions")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
