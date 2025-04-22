import numpy as np

# File paths
xv6_file = r"C:\Users\nikhi\LabProject\8742l5\xv6.txt"
xv9_file = r"C:\Users\nikhi\LabProject\8742l5\xv9.txt"
yv6_file = r"C:\Users\nikhi\LabProject\8742l5\yv6.txt"
yv9_file = r"C:\Users\nikhi\LabProject\8742l5\yv9.txt"

# Function to read data from a file
def read_data(file_path):
    with open(file_path, "r") as f:
        data = [float(line.strip()) for line in f if line.strip() and line.strip() != "nan"]
    return np.array(data)

# Load data
xv6 = read_data(xv6_file)
xv9 = read_data(xv9_file)
yv6 = read_data(yv6_file)
yv9 = read_data(yv9_file)

# Ensure arrays have the same length
min_len_x = min(len(xv6), len(xv9))
min_len_y = min(len(yv6), len(yv9))

xv6, xv9 = xv6[:min_len_x], xv9[:min_len_x]
yv6, yv9 = yv6[:min_len_y], yv9[:min_len_y]

# Compute variance
variance_x = np.var(xv6 - xv9)
variance_y = np.var(yv6 - yv9)

print(f"Variance between xv6 and xv9: {variance_x}")
print(f"Variance between yv6 and yv9: {variance_y}")
