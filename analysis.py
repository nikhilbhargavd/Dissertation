filename = r"C:\Users\nikhi\LabProject\brush_predictions_v4_slp\predictions_v4_copy\8791_r1.mp4.000_8791_r1.analysis.h5"

## loads the file and returns file name, dataset, locations and the different nodes 
import h5py
import numpy as np

with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

print("===filename===")
print(filename)
print()

print("===HDF5 datasets===")
print(dset_names)
print()

print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print()



## returns frame,node and instance counts
frame_count, node_count, _, instance_count = locations.shape

print("frame count:", frame_count)
print("node count:", node_count)
print("instance count:", instance_count)

"""
def fill_missing(Y, kind="linear"):
    #Fills missing values independently along each dimension after the first.
    # Store initial shape.
    initial_shape = Y.shape
    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))
    # Interpolate along each slice.
      for i in range(Y.shape[-1]):
        y = Y[:, i]
        # Build interpolant.

        x = np.flatnonzero(~np.isnan(y))
        if not x.size: # check if array is empty
            y[0]=1.0 # append nonzero float to start
            y[-1]=1.0 # append nonzero float to end
            x=np.flatnonzero(~np.isnan(y)) # recreate x

        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y"""


lfpaw_INDEX = 0
rfpaw_INDEX = 1
snout_INDEX = 8

lfpaw_loc = locations[:, lfpaw_INDEX, :, :]
rfpaw_loc = locations[:, rfpaw_INDEX, :, :]
snout_loc = locations[:, snout_INDEX, :, :]

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

lfpaw_INDEX = 0
rfpaw_INDEX = 1
snout_INDEX = 8

# Assuming locations is already defined
lfpaw_loc = locations[:, lfpaw_INDEX, :, :]
rfpaw_loc = locations[:, rfpaw_INDEX, :, :]
snout_loc = locations[:, snout_INDEX, :, :]

# Print the shape and first few rows of snout_loc to debug
print("snout_loc shape:", snout_loc.shape)
print("First few values in snout_loc:", snout_loc[:10])  # Print first 10 rows

# Check if snout_loc contains NaNs or Infs
if np.any(np.isnan(snout_loc)) or np.any(np.isinf(snout_loc)):
    print("snout_loc contains NaNs or Infs.")
    snout_loc = np.nan_to_num(snout_loc, nan=0, posinf=1024, neginf=-1024)

# Ensure snout_loc contains valid data
if np.any(np.isnan(snout_loc)) or np.any(np.isinf(snout_loc)):
    print("snout_loc still contains invalid values after cleaning.")

sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15, 6]

# Plotting snout locations
plt.figure()
plt.plot(snout_loc[:, 0, 0], 'y', label='mouse-0')  # Plot for mouse-0

plt.legend(loc="center right")
plt.title('Snout locations')

# Plotting snout tracks
plt.figure(figsize=(7, 7))
plt.plot(snout_loc[:, 0, 0], snout_loc[:, 1, 0], 'y', label='mouse-0')

plt.legend()

# Set limits based on the data range
"""# Avoid NaN or Inf values in the min/max calculations
x_min = np.nanmin(snout_loc[:, 0, 0])
x_max = np.nanmax(snout_loc[:, 0, 0])
y_min = np.nanmin(snout_loc[:, 1, 0])
y_max = np.nanmax(snout_loc[:, 1, 0])

# Check for NaN or Inf before setting axis limits
if np.any(np.isnan([x_min, x_max, y_min, y_max])) or np.any(np.isinf([x_min, x_max, y_min, y_max])):
    print("Error: Axis limits contain NaN or Inf values.")
else:
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
"""
plt.xticks([])
plt.yticks([])

plt.title('Snout tracks')
plt.show()

from scipy.signal import savgol_filter

def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array
    
    win defines the window to smooth over
    
    poly defines the order of the polynomial
    to fit with
    
    """
    node_loc_vel = np.zeros_like(node_loc)
    
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)
    
    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel

snout_mouse = smooth_diff(snout_loc[:, :, 0])

fig = plt.figure(figsize=(15,7))
ax1 = fig.add_subplot(211)
ax1.plot(snout_loc[:, 0, 0], 'k', label='x')
ax1.plot(-1*snout_loc[:, 1, 0], 'k', label='y')
ax1.legend()
ax1.set_xticks([])
ax1.set_title('Snout')

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.imshow(snout_mouse[:,np.newaxis].T, aspect='auto', vmin=0, vmax=10)
ax2.set_yticks([])
ax2.set_title('Velocity')

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax1.plot(snout_loc[:, 0, 0], 'k')
ax1.set_xlim(0,1024)
ax1.set_xticks([])
ax1.set_ylim(0,1024)
ax1.set_yticks([])
ax1.set_title('Snout traks')

kp = snout_mouse

vmin = 0
vmax = 10

ax2 = fig.add_subplot(122)
ax2.scatter(snout_loc[:, 0, 0], snout_loc[:, 1, 0], c=kp, s=4, vmin=vmin, vmax=vmax)
ax2.set_xlim(0,1024)
ax2.set_xticks([])
ax2.set_ylim(0,1024)
ax2.set_yticks([])
ax2.set_title('Snout tracks colored by magnitude of mouse speed')



























