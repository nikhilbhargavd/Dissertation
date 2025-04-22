import numpy as np
import pandas as pd
import os

# Define the base directory and model versions
base_dir = r"C:\Users\nikhi\LabProject\new"
models = [f"v{i}" for i in range(5, 13)]  # v5 to v12

# Initialize an empty list to store extracted data
metrics_list = []

# Loop through each model directory and extract data
for model in models:
    file_path = os.path.join(base_dir, model, f"{model}metrics.val.npz")
    
    # Load the metrics file
    if os.path.exists(file_path):
        print(f"Processing: {file_path}")
        metrics_data = np.load(file_path, allow_pickle=True)

        # Extract relevant metrics
        metrics_dict = metrics_data["metrics"].item()  # Convert object array to dictionary

        # Add the model name to the data
        metrics_dict["Model"] = model

        # Append extracted metrics to list
        metrics_list.append(metrics_dict)
    else:
        print(f"File not found: {file_path}")

# Convert list of dictionaries to Pandas DataFrame
df_metrics = pd.DataFrame(metrics_list)

# Save extracted data to a CSV file
csv_output_path = os.path.join(base_dir, "all_metrics.csv")
df_metrics.to_csv(csv_output_path, index=False)

print(f"✅ Extraction complete! Data saved to: {csv_output_path}")
