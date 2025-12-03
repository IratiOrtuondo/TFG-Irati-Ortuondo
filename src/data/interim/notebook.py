import rasterio, numpy as np
for p in [r"src\data\interim\debug_HH.tif", r"src\data\interim\debug_TB.tif"]:
    try:
        with rasterio.open(p) as ds:
            a = ds.read(1)
            print(p, "CRS:", ds.crs, "Shape:", a.shape, "NaN%:", np.isnan(a).mean()*100,
                  "min:", np.nanmin(a), "max:", np.nanmax(a))
    except Exception as e:
        print(p, "NO ABRE:", e)