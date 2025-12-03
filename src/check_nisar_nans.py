import h5py
import numpy as np
import matplotlib.pyplot as plt

# Path to your NISAR file
nisar_path = "data/raw/NISAR_L2_PR_GCOV.h5"

# Path to the main backscatter band (update if needed)
dset_path = "/science/LSAR/GCOV/grids/frequencyA/HHHH"

with h5py.File(nisar_path, "r") as f:
    arr = f[dset_path][...]

# Count NaNs and fill values (commonly -9999 or similar)
nan_count = np.isnan(arr).sum()
fill_count = np.sum(arr == -9999)
total = arr.size

print(f"Total pixels: {total}")
print(f"NaN pixels: {nan_count} ({100*nan_count/total:.2f}%)")
print(f"Fill value (-9999) pixels: {fill_count} ({100*fill_count/total:.2f}%)")

# Visualize valid data mask
plt.figure(figsize=(8,6))
plt.imshow(np.isfinite(arr), cmap="gray")
plt.title("NISAR HHHH Valid Data Mask (white=valid)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
