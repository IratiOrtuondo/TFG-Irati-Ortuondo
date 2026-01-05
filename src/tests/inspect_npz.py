import numpy as np
from pathlib import Path
p = Path('c:/Users/ortuo/tfgirati/tfg-nisar/data/interim/tb_disagg-20150607-VV.npz')
print('exists', p.exists())
if p.exists():
    z = np.load(p)
    print('keys', list(z.keys()))
    for k in z.files:
        arr = z[k]
        print(k, 'shape', getattr(arr, 'shape', None), 'dtype', getattr(arr, 'dtype', None))
else:
    print('File not found')
