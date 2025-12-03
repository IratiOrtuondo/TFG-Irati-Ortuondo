"""Comprueba que las dependencias mínimas se pueden importar y muestra versiones.

Uso:
    python scripts/check_env.py
"""

import importlib
import sys

pkgs = ["numpy", "h5py", "rasterio", "loguru"]

ok = True
for p in pkgs:
    try:
        m = importlib.import_module(p)
        ver = getattr(m, "__version__", "<no __version__>")
        print(f"{p}: OK, version={ver}")
    except Exception as e:
        print(f"{p}: ERROR importing: {type(e).__name__}: {e}")
        ok = False

# Si rasterio está disponible, mostrar CRS/Driver info mínimo
try:
    import rasterio
    print("rasterio drivers sample:", list(rasterio.supported_drivers.items())[:5])
except Exception:
    pass

if not ok:
    print("\nAl menos una importación falló. Revisa la instalación del environment.")
    sys.exit(2)

print("\nComprobación terminada: todas las importaciones principales cargaron correctamente.")
