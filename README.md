
# NISAR-SMAP Collocation Pipeline

This repository provides a robust workflow to collocate NISAR L2 GCOV backscatter (σ0) and SMAP L3 brightness temperature (TB) or soil moisture (SM) products onto a common grid, suitable for regional or global analysis.

## Prerequisites

- Python 3.9+ (recommended: 3.12)
- Windows (PowerShell) or Linux/Mac (bash)
- Earthdata credentials (for SMAP download)
- Recommended: conda or venv for environment management

## 1. Environment Setup

1. Clone the repository and open a terminal in the project root.
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

## 2. Download Data

### 2.1. Download SMAP L3 (Soil Moisture or TB)

- Set your Earthdata credentials as environment variables (or configure your `~/.netrc`):
	```powershell
	$env:EARTHDATA_USERNAME="your_username"
	$env:EARTHDATA_PASSWORD="your_password"
	```
- Download a SMAP L3 SM file (contains TB variables):
	```powershell
	python src\smap.py --date 2021-06-01
	```
	This will save the file in `data/raw/`.

### 2.2. Place NISAR GCOV File

- Copy your NISAR L2 GCOV HDF5 file (e.g. `NISAR_L2_PR_GCOV.h5`) into `data/raw/`.

## 3. (Recommended) Create a Regional UTM Template

If your NISAR scene is local (not global EASE2), create a UTM template to avoid reprojection NaNs:

```powershell
python src\make_utm_template.py --nisar data\raw\NISAR_L2_PR_GCOV.h5 --out data\processed\UTM_1km_template.tif
```

This generates a template GeoTIFF matching the NISAR CRS and extent.

## 4. Run the Collocation Pipeline (`step1.py`)

### 4.1. With Default (EASE2) Template (Global)

```powershell
python src\step1.py --date 2021-06-01 --pol V --smap-l3 data\raw\SMAP_L3_SM_P_20210601_R19240_001.h5 --nisar data\raw\NISAR_L2_PR_GCOV.h5 --tb-group AM --qa
```
- This will use a global EASE2 36km template. If your NISAR is local, most pixels will be NaN.

### 4.2. With Regional UTM Template (Recommended)

```powershell
python src\step1.py --date 2021-06-01 --pol V --smap-l3 data\raw\SMAP_L3_SM_P_20210601_R19240_001.h5 --nisar data\raw\NISAR_L2_PR_GCOV.h5 --tb-group AM --template data\processed\UTM_1km_template.tif --qa
```
- This will align both SMAP TB and NISAR σ0 to the same UTM grid, maximizing valid overlap.

#### Arguments:
- `--date`: Date of interest (YYYY-MM-DD)
- `--pol`: SMAP TB polarization (H or V, default V)
- `--smap-l3`: Path to SMAP L3 file
- `--nisar`: Path to NISAR GCOV file
- `--tb-group`: AM, PM, or AUTO (default AUTO). Use AM for morning pass.
- `--template`: Path to template GeoTIFF (use UTM for local scenes)
- `--qa`: Save debug GeoTIFFs for quick inspection

## 5. Outputs

- `data/interim/aligned_step1.npz`: Numpy archive with:
	- `TBc_2d`: Collocated SMAP TB (K)
	- `S_pp_dB`: Collocated NISAR σ0 (dB, HH polarization)
	- `S_pq_dB`: Cross-pol (HV), filled with NaN if not present
	- `crs_wkt`, `transform`, `height`, `width`: Grid metadata
- `data/interim/debug_TB.tif`, `debug_HH.tif`: QA GeoTIFFs for visualization

## 6. Notes and Troubleshooting

- If you see 100% NaNs in the output, check that your template CRS matches the NISAR scene (use UTM template for local scenes).
- Only HH polarization is available in the provided NISAR GCOV; cross-pol is filled with NaN.
- SMAP L3 SM files contain TB variables under `Soil_Moisture_Retrieval_Data_AM/PM` as `tb_h_corrected`/`tb_v_corrected`.
- Use `--tb-group AM` or `--tb-group PM` to select the desired SMAP pass.
- For other dates, repeat the download and processing steps with the new date.

## 7. Advanced

- To inspect available datasets in your HDF5 files:
	```powershell
	python src\inspect_h5.py data\raw\NISAR_L2_PR_GCOV.h5
	python src\inspect_smap_l3.py data\raw\SMAP_L3_SM_P_20210601_R19240_001.h5
	```
- To download a range of SMAP files:
	```powershell
	python src\smap.py --start 2021-06-01 --end 2021-06-04
	```

## 8. Citation
If you use this workflow, please cite the original NISAR and SMAP data products and this repository.

---
For questions or improvements, open an issue or contact the repository maintainer.
