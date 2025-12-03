import os, glob, numpy as np, rasterio
import matplotlib.pyplot as plt

interim = os.path.join('..','data','interim')

def load_tif_like(pattern):
    files = sorted(glob.glob(os.path.join(interim, pattern)))
    if not files:
        print("No encontré", pattern); return None, None
    path = files[0]
    with rasterio.open(path) as src:
        arr = src.read(1)
    print("OK:", os.path.basename(path), arr.shape, f"range {np.nanmin(arr):.2f}..{np.nanmax(arr):.2f}")
    return arr, path

# 1) MASK (QC)
mask, _ = load_tif_like("*mask*.tif")  # si la exportaste; si no, lee directo del H5
if mask is not None:
    plt.figure(figsize=(7,6)); plt.imshow(mask, vmin=np.nanpercentile(mask,1), vmax=np.nanpercentile(mask,99))
    plt.title("Mask (layover/shadow/no-data)"); plt.colorbar(); plt.tight_layout(); plt.show()

# 2) INCIDENCE ANGLE (mapa + hist)
inc, _ = load_tif_like("*incidenceAngle*.tif")
if inc is not None:
    plt.figure(figsize=(7,6)); plt.imshow(inc, vmin=np.nanpercentile(inc,1), vmax=np.nanpercentile(inc,99))
    plt.title("Incidence angle (deg)"); plt.colorbar(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,4)); plt.hist(inc[np.isfinite(inc)].ravel(), bins=60)
    plt.xlabel("Incidence angle (deg)"); plt.ylabel("px"); plt.title("Histograma de ángulos"); plt.tight_layout(); plt.show()

# 3) NUMBER OF LOOKS
nlooks, _ = load_tif_like("*numberOfLooks*.tif")
if nlooks is not None:
    plt.figure(figsize=(7,6)); plt.imshow(nlooks, vmin=np.nanpercentile(nlooks,1), vmax=np.nanpercentile(nlooks,99))
    plt.title("Number of Looks"); plt.colorbar(); plt.tight_layout(); plt.show()

# 4) γ⁰ HH vs σ⁰ HH (si exportaste sigma0 con rtc factor)
hh_gamma_db, _ = load_tif_like("*HHHH_dB.tif")
hh_sigma_db, _ = load_tif_like("*HHHH_sigma0_dB.tif")
if (hh_gamma_db is not None) and (hh_sigma_db is not None):
    diff = hh_sigma_db - hh_gamma_db
    plt.figure(figsize=(7,6)); 
    v = np.nanpercentile(np.abs(diff), 99); plt.imshow(diff, vmin=-v, vmax=v)
    plt.title("σ⁰_dB − γ⁰_dB"); plt.colorbar(); plt.tight_layout(); plt.show()

# 5) HV dB y CPR (si existe HV)
hv_db, _ = load_tif_like("*HVHV_dB*.tif")
if (hh_gamma_db is not None) and (hv_db is not None):
    plt.figure(figsize=(7,6)); 
    plt.imshow(hv_db, vmin=np.nanpercentile(hv_db,1), vmax=np.nanpercentile(hv_db,99))
    plt.title("HV (dB)"); plt.colorbar(); plt.tight_layout(); plt.show()

    cpr, _ = load_tif_like("*CPR*.tif")  # si ya lo generaste en tu script
    if cpr is not None:
        plt.figure(figsize=(7,6));
        plt.imshow(cpr, vmin=np.nanpercentile(cpr,1), vmax=np.nanpercentile(cpr,99))
        plt.title("CPR = HV/HH"); plt.colorbar(); plt.tight_layout(); plt.show()

        # Scatter/hexbin HH vs HV (muestra aleatoria para no petar RAM)
        valid = np.isfinite(hh_gamma_db) & np.isfinite(hv_db)
        yy = hv_db[valid].ravel(); xx = hh_gamma_db[valid].ravel()
        if xx.size > 200_000:
            idx = np.random.choice(xx.size, 200_000, replace=False)
            xx, yy = xx[idx], yy[idx]
        plt.figure(figsize=(6,5)); plt.hexbin(xx, yy, gridsize=80, mincnt=1)
        plt.xlabel("HH (dB)"); plt.ylabel("HV (dB)"); plt.title("Hexbin HH vs HV"); plt.colorbar(label="count")
        plt.tight_layout(); plt.show()

# 6) Efecto de normalización angular (si guardaste versiones *_norm35.tif)
hh_db_n, _ = load_tif_like("*HHHH_dB_norm35*.tif")
if (hh_gamma_db is not None) and (hh_db_n is not None):
    # histogramas comparados
    plt.figure(figsize=(6,4))
    a = hh_gamma_db[np.isfinite(hh_gamma_db)]; b = hh_db_n[np.isfinite(hh_db_n)]
    plt.hist(a, bins=60, alpha=0.5, label="sin normalizar")
    plt.hist(b, bins=60, alpha=0.5, label="norm 35°")
    plt.legend(); plt.xlabel("dB"); plt.ylabel("px"); plt.title("HH dB: efecto normalización"); plt.tight_layout(); plt.show()

    # dependencia con ángulo (antes vs después)
    if inc is not None:
        valid = np.isfinite(inc) & np.isfinite(hh_gamma_db) & np.isfinite(hh_db_n)
        ang = inc[valid].ravel()
        h0 = hh_gamma_db[valid].ravel()
        h1 = hh_db_n[valid].ravel()
        # muestreo para rapidez
        if ang.size > 200_000:
            idx = np.random.choice(ang.size, 200_000, replace=False)
            ang, h0, h1 = ang[idx], h0[idx], h1[idx]
        plt.figure(figsize=(6,5)); plt.hexbin(ang, h0, gridsize=80, mincnt=1)
        plt.xlabel("Incidence (deg)"); plt.ylabel("HH dB"); plt.title("Antes de normalizar"); plt.colorbar(); plt.tight_layout(); plt.show()
        plt.figure(figsize=(6,5)); plt.hexbin(ang, h1, gridsize=80, mincnt=1)
        plt.xlabel("Incidence (deg)"); plt.ylabel("HH dB"); plt.title("Después de normalizar"); plt.colorbar(); plt.tight_layout(); plt.show()

# 7) Speckle: std local (opcional; requiere scipy)
try:
    from scipy.ndimage import uniform_filter
    if hh_gamma_db is not None:
        # std local 5x5 en dB
        a = hh_gamma_db.copy()
        m = np.isfinite(a); a[~m] = np.nanmedian(a)
        k = 5
        mean = uniform_filter(a, size=k)
        mean2 = uniform_filter(a*a, size=k)
        local_std = np.sqrt(np.clip(mean2 - mean*mean, 0, None))
        local_std[~m] = np.nan
        plt.figure(figsize=(7,6)); 
        v = np.nanpercentile(local_std, 99)
        plt.imshow(local_std, vmin=0, vmax=v)
        plt.title("STD local (5×5) — speckle"); plt.colorbar(); plt.tight_layout(); plt.show()
except Exception as e:
    print("Speckle opcional: instala scipy si lo quieres (conda install -c conda-forge scipy).", e)
