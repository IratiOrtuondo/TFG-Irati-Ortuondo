import numpy as np

path = r"c:\Users\ortuo\tfgirati\tfg-nisar\data\interim\smap-ancillary-20150607.npz"
d = np.load(path, allow_pickle=True)

print("Keys:", list(d.keys()))
print()
for k in d.keys():
    v = d[k]
    if hasattr(v, 'shape'):
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"{k}: {type(v)}")
