# SMAP-SMAP Soil Moisture Disaggregation Pipeline

This repository provides a complete workflow to disaggregate SMAP L3 brightness temperature (TB) from 36 km to fine resolution (~3 km) using SMAP radar backscatter data. The pipeline implements an ATBD-consistent tau-omega radiative transfer model to retrieve high-resolution soil moisture from disaggregated TB.

## Overview

The pipeline consists of the following steps:

1. **Step 0**: Download SMAP L3 radiometer (TB, SM) and radar (σ0) data
2. **Step 1**: Collocate SMAP TB, SM, co-pol, and cross-pol backscatter on a common grid (both coarse 36km and native ~3km resolution)
3. **Step 2**: Estimate active-passive parameters (β and Γ) from multi-temporal analysis
4. **Step 3**: Disaggregate TB from 36 km to native resolution using β, Γ, and fine-scale backscatter
5. **Step 4**: Retrieve soil moisture from disaggregated TB using tau-omega radiative transfer model

## Prerequisites

- Python 3.9+ (recommended: 3.12)
- Windows (PowerShell) or Linux/Mac (bash)
- Earthdata credentials (for SMAP download)
- Recommended: conda or venv for environment management

## Installation

1. Clone the repository and navigate to the project root.

2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   Or with conda:
   ```bash
   conda create -n nisar-smap python=3.12
   conda activate nisar-smap
   ```

3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Step-by-Step Pipeline

### Step 0: Download SMAP Data

Download SMAP L3 radiometer (SPL3SMP) and optionally radar (SPL3SMA) data using `step0_download_data.py`.

**Set Earthdata credentials** (if not using `~/.netrc`):
```powershell
$env:EARTHDATA_USERNAME="your_username"
$env:EARTHDATA_PASSWORD="your_password"
```

**Download for a single date** (global):
```powershell
python src\step0_download_data.py --date 2015-05-01
```

**Download radiometer + radar** for a date:
```powershell
python src\step0_download_data.py --date 2015-05-01 --also-radar
```

**Download for a specific region** (e.g., Boulder, CO):
```powershell
python src\step0_download_data.py --date 2015-05-01 --boulder-36km --also-radar
```

**Download using custom bounding box**:
```powershell
python src\step0_download_data.py --date 2015-05-01 --bbox -105,39,-104,41 --also-radar
```

Files will be saved in `data/raw/`.

### Step 1: Collocate Data on Common Grid

This step aligns SMAP TB (36 km), co-pol backscatter (VV or HH), and cross-pol backscatter (VH or HV) on a common EASE2 grid at both coarse (36 km) and native (~3 km) resolution.

#### Step 1a: Extract TB at 36 km

```powershell
python src\step1_tb36_coarse.py --data-dir data\raw --date 20150501 --out-dir data\interim --pol V
```

#### Step 1b: Collocate Co-pol Backscatter

```powershell
python src\step1_coarse_copol.py --date 20150501 --pol VV --bbox -105,39,-103,41 --pixel-size 1000
```

Arguments:
- `--date`: Date in YYYYMMDD format
- `--pol`: Polarization (VV or HH)
- `--bbox`: Bounding box (lon_min,lat_min,lon_max,lat_max)
- `--pixel-size`: Output pixel size in meters (default: 1000 for ~1 km)

Outputs:
- `data/interim/aligned-smap-copol-<date>-<pol>.npz` (coarse 36 km)
- `data/interim/aligned-smap-copol-<date>-<pol>-native.npz` (native ~3 km)

#### Step 1c: Collocate Cross-pol Backscatter

```powershell
python src\step1_coarse_crosspol.py --date 20150501 --bbox -105,39,-103,41 --pixel-size 1000
```

Outputs:
- `data/interim/aligned-smap-xpol-<date>.npz` (coarse 36 km)
- `data/interim/aligned-smap-xpol-<date>-native.npz` (native ~3 km)

#### Step 1d: (Optional) Collocate Soil Moisture for Validation

```powershell
python src\step1_coarse_sm.py
```

This generates `data/processed/smap_sm_coarse_<date>.npz` for validation purposes.

**Repeat Steps 1a-1c for multiple dates** to build a temporal stack (required for Step 2).

### Step 2: Estimate Active-Passive Parameters (β and Γ)

This step performs multi-temporal regression to estimate:
- **β(C)**: Active-passive coupling coefficient [K/dB]
- **Γ(C)**: Heterogeneity parameter [dimensionless]

Using the ATBD-like model:
```
ΔTB(C,t) ≈ β·Δσ_pp(C,t) + β·Γ·Δσ_pq(C,t)
```

**Run the regression**:
```powershell
python src\step2_beta_gamma.py --start-date 20150501 --end-date 20150531 --pol VV
```

Arguments:
- `--start-date`: Start date (YYYYMMDD)
- `--end-date`: End date (YYYYMMDD)
- `--pol`: Polarization (VV or HH)
- `--min-finite-frac`: Minimum fraction of valid TB pixels per date (default: 0.7)

Outputs:
- `data/processed/beta_36km_<pol>.tif`: β parameter map
- `data/processed/gamma_36km_<pol>.tif`: Γ parameter map
- `data/processed/beta_36km_<pol>.npz`: β parameter (NPZ format)
- `data/processed/gamma_36km_<pol>.npz`: Γ parameter (NPZ format)

### Step 3: Disaggregate TB to Native Resolution

This step disaggregates TB from 36 km to native resolution (~3 km) using the equation:

```
TB_p(M_j) = TB_p(C) + β(C)·[(σ_pp(M_j) - σ_pp(C)) + Γ(C)·(σ_pq(C) - σ_pq(M_j))]
```

**Run disaggregation**:
```powershell
python src\step3_disagg_TB.py --date 20150501 --pol VV --step3-dir data\processed --step2-dir data\interim --interim-dir data\interim --out-dir data\processed
```

Arguments:
- `--date`: Date (YYYYMMDD)
- `--pol`: Polarization (VV or HH)
- `--step3-dir`: Directory with β and Γ maps (from Step 2)
- `--step2-dir`: Directory with TB 36 km files (from Step 1a)
- `--interim-dir`: Directory with native backscatter files (from Step 1b/1c)
- `--out-dir`: Output directory

Outputs:
- `data/processed/TB_fine_<date>_<pol>_native.npz`: Disaggregated TB at native resolution

### Step 4: Retrieve Soil Moisture from TB

This step inverts the ATBD tau-omega radiative transfer model to retrieve soil moisture from disaggregated TB:

```
TB_p = Ts·e_p·γ + Tc·(1 - ω_p)·(1 - γ)·(1 + r_p·γ)
γ = exp(-τ_p·sec(θ))
e_p = 1 - r_p
```

**Run soil moisture retrieval**:
```powershell
python src\step4_final.py --tb data\processed\TB_fine_20150501_VV_native.npz --date 20150501 --Teff-npz data\processed\Teff_K.npz --theta-npz data\processed\theta_deg.npz --tau-npz data\processed\tau.npz --omega-npz data\processed\omega.npz --h-npz data\processed\h.npz --out-dir data\processed
```

Arguments:
- `--tb`: Disaggregated TB NPZ file (from Step 3)
- `--date`: Date (YYYYMMDD)
- `--Teff-npz`: Effective temperature map (optional, uses constant if not provided)
- `--Ts-npz`: Surface temperature map (optional)
- `--Tc-npz`: Canopy temperature map (optional)
- `--theta-npz`: Incidence angle map (optional, uses SMAP nominal 40° if not provided)
- `--tau-npz`: Vegetation optical depth map (optional)
- `--omega-npz`: Single scattering albedo map (optional)
- `--h-npz`: Surface roughness parameter map (optional)
- `--out-dir`: Output directory

Outputs:
- `data/processed/SM_<date>_native.npz`: Soil moisture at native resolution [m³/m³]
- `data/processed/SM_<date>_native.tif`: Soil moisture GeoTIFF for visualization

### Step 1 (Alternative): Generate ATBD Parameters and Run Pipeline

For automated processing of multiple dates, use `step1_generate_atbd_params.py` which:
1. Reprojects ancillary parameters to native grid
2. Calls Step 4 automatically for each date

**Edit the script** to configure dates and paths, then run:
```powershell
python src\step1_generate_atbd_params.py
```

## Outputs

### Intermediate Files (`data/interim/`)
- `aligned-smap-copol-<date>-<pol>.npz`: Co-pol backscatter at 36 km
- `aligned-smap-copol-<date>-<pol>-native.npz`: Co-pol backscatter at native resolution
- `aligned-smap-xpol-<date>.npz`: Cross-pol backscatter at 36 km
- `aligned-smap-xpol-<date>-native.npz`: Cross-pol backscatter at native resolution
- `TB_36km_<date>_<pol>.npz`: SMAP TB at 36 km

### Processed Files (`data/processed/`)
- `beta_36km_<pol>.tif/npz`: Active-passive coupling coefficient
- `gamma_36km_<pol>.tif/npz`: Heterogeneity parameter
- `TB_fine_<date>_<pol>_native.npz`: Disaggregated TB at native resolution
- `SM_<date>_native.npz/tif`: Retrieved soil moisture at native resolution

## Analysis and Visualization

### Analyze Soil Moisture Statistics
```powershell
python src\analyze_sm_statistics.py
```

### Plot Soil Moisture Comparison (Fine vs Coarse)
```powershell
python src\plot_sm_fine_coarse.py
```

### Visualize GCOV Data
Open `notebooks/visualize_gcov.ipynb` in Jupyter to explore the data interactively.

## Notes and Troubleshooting

- **Multi-temporal analysis**: Step 2 requires at least 10-15 dates for robust parameter estimation.
- **EASE2 grid**: All processing uses the EASE2 projection (EPSG:6933) for consistency with SMAP products.
- **Native resolution**: SMAP L1C radar has ~3 km posting; final resolution depends on sensor characteristics.
- **Missing cross-pol**: If cross-pol data is unavailable, the pipeline falls back to co-pol-only regression (Γ = 0).
- **Parameter validation**: β typically ranges from 1-10 K/dB; Γ ranges from 0-1. Values outside these ranges are clipped.
- **Regularization**: The tau-omega inversion includes emissivity and SM clipping to prevent unphysical retrievals.

## Advanced Usage

### Custom Bounding Box
Specify a custom study area using `--bbox` in Step 1 scripts:
```powershell
python src\step1_coarse_copol.py --date 20150501 --pol VV --bbox -110,35,-100,45 --pixel-size 1000
```

### High-Resolution Output
Decrease `--pixel-size` for finer output grids (limited by SMAP L1C native resolution):
```powershell
python src\step1_coarse_copol.py --date 20150501 --pol VV --bbox -105,39,-103,41 --pixel-size 500
```

### Using External ATBD Parameters
Provide custom maps for any ATBD parameter in Step 4:
```powershell
python src\step4_final.py --tb <TB_file> --date 20150501 --Teff-npz <custom_Teff.npz> --tau-npz <custom_tau.npz> ...
```

## Citation

If you use this workflow, please cite the original SMAP data products:
- SMAP ATBD
- SMAP L3 Radiometer: https://nsidc.org/data/spl3smp
- SMAP L1C Radar: https://nsidc.org/data/spl1c_s0_lores

---
For questions or improvements, open an issue or contact the repository maintainer.
