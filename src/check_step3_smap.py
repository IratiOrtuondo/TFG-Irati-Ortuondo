import numpy as np
from pathlib import Path

path = Path("..") / "data" / "interim" / "step3_smap_boulder_inversion_params.npz"
npz = np.load(path, allow_pickle=True)

beta = npz["beta_K_per_dB"]
gamma = npz["gamma_unitless"]
n_samples = npz["n_samples"]
r2 = npz["r2"]

print("beta stats:", np.nanmin(beta), np.nanmax(beta))
print("n_samples unique:", np.unique(n_samples))
print("r2 stats:", np.nanmin(r2), np.nanmax(r2))
print("meta:", npz["meta"])
