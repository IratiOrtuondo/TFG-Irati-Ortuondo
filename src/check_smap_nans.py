import h5py
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Path to your SMAP file
smap_path = "data/raw/SMAP_L3_SM_P_20210601_R19240_001.h5"
# Try to find the TB variable automatically (as in your pipeline)
def find_tb_var(ds, pol="V"):
    pol = pol.upper()
    for v in ds.data_vars:
        if pol in v.upper() and "TB" in v.upper():
            return v
    # fallback: first 2D variable
    for v in ds.data_vars:
        if ds[v].ndim == 2:
            return v
    return None

with xr.open_dataset(smap_path, engine="h5netcdf", phony_dims="sort") as ds:
    tb_var = find_tb_var(ds, pol="V")
    if tb_var is None:
        raise RuntimeError("No TB variable found!")
    arr = ds[tb_var].values

nan_count = np.isnan(arr).sum()
total = arr.size
print(f"TB variable: {tb_var}")
print(f"Total pixels: {total}")
print(f"NaN pixels: {nan_count} ({100*nan_count/total:.2f}%)")

plt.figure(figsize=(8,6))
plt.imshow(np.isfinite(arr), cmap="gray")
plt.title(f"SMAP {tb_var} Valid Data Mask (white=valid)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
