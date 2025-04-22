import glob
import subprocess

# Directory containing your .predictions.slp files
input_dir = r"C:/Users/nikhi/LabProject/new/v15/Predictions"

# Find all .predictions.slp files
file_paths = glob.glob(f"{input_dir}/*.predictions.slp")

# Loop through and run sleap-convert on each file
for file_path in file_paths:
    cmd = ["sleap-convert", "--format", "analysis", file_path]
    subprocess.run(cmd)
    print(f"Processed: {file_path}")
