# Rock Glacier Deformation Analysis - Complete Workflow Guide

## Overview

This repository contains a comprehensive workflow for measuring 3D rock glacier deformation using multi-temporal LiDAR DEMs. The workflow implements the "gold standard" approach: measuring horizontal displacement first (using COSI-Corr), then using that information to warp DEMs and isolate true vertical surface change.

## Scientific Background

### Why This Approach?

Simple DEM differencing on sloped terrain produces **apparent elevation changes** that mix:
1. Real surface lowering/thickening (what we want to measure)
2. Topographic effects from horizontal displacement (noise we need to remove)

For example, if a rock glacier flows 2m downhill on a 30° slope, simple differencing would show ~1m of elevation loss even if the surface didn't actually thin at all.

### The Solution: Lagrangian Back-Warping

1. **Measure horizontal displacement** (dx, dy) using image correlation
2. **Warp the target DEM** backwards by that displacement
3. **Difference the warped DEM** with the reference to get true vertical change

This is standard practice in glacier geodesy (Nuth & Kääb 2011, Dehecq et al. 2015).

---

## Repository Structure

```
rg_velocity/
├── WORKFLOW_GUIDE.md                          # This file
│
├── 00_generate_test_patches.py                # Utility: Crop DEMs to test patch
├── preprocess_dems_snowbird.py                # Step 1: DEM harmonization
├── 04_DEM_characterization.py                 # Step 2: Quality assessment
├── 03_horizontal_displacement_xdem.py         # Step 3: COSI-Corr wrapper
├── 05_3d_synthesis_xdem.py                    # Step 5: Warping & 3D vectors
├── vertical_displacement_analysis_FIXED.py    # Alternative: Vertical-only analysis
│
└── [Various diagnostic scripts]
```

---

## Complete Workflow

### Prerequisites

1. **Install Mamba** (faster than conda):
   ```bash
   conda install -n base -c conda-forge mamba
   ```

2. **Create the environment**:
   ```bash
   mamba create -n rock_glacier_env python=3.9 xdem gdal geopandas numpy matplotlib scipy -c conda-forge
   mamba activate rock_glacier_env
   ```

3. **Clone COSI-Corr** (external tool):
   ```bash
   cd "M:/My Drive/Rock Glaciers/Tools"
   git clone https://github.com/SaifAati/Geospatial-COSICorr3D.git
   ```

---

### Step 0: Generate Test Patches (Optional but Recommended)

**Why:** Work on a small area first to tune parameters and iterate quickly.

```bash
python 00_generate_test_patches.py
```

**What it does:**
- Reads `shapefiles/small_patch.shp`
- Crops all harmonized DEMs to that extent
- Saves to `preprocessed_dems/patches/`

**Output:**
```
preprocessed_dems/patches/
├── 2018_0p5m_upper_rg_dem_larger_roi_harmonized.tif
├── GadValleyRG_50cmDEM_2023_harmonized.TIF
├── GadValleyRG_50cmDEM_2024_harmonized.TIF
└── GadValleyRG_50cmDEM_2025_harmonized.TIF
```

**To use patches in subsequent steps:** Set `USE_TEST_PATCH = True` in scripts.

---

### Step 1: DEM Preprocessing

**Script:** `preprocess_dems_snowbird.py`

**What it does:**
- Co-registers DEMs to a stable reference
- Harmonizes grids (same extent, resolution, CRS)
- Generates hillshades for visualization

```bash
python preprocess_dems_snowbird.py
```

**Outputs:**
- `*_harmonized.tif` - Aligned DEMs
- `*_hs_harmonized.tif` - Hillshades (for Step 3)

---

### Step 2: DEM Quality Assessment

**Script:** `04_DEM_characterization.py`

**What it does:**
- Calculates roughness (local standard deviation)
- Generates hillshades
- Compares data quality across epochs

```bash
python 04_DEM_characterization.py
```

**Toggle test patch mode:**
```python
USE_TEST_PATCH = True  # In the script
```

**Outputs:**
- Hillshade images
- Roughness maps
- Comparison plots

**Interpretation:**
- High roughness = noisy data or blocky terrain
- Use this to identify problem areas

---

### Step 3: Horizontal Displacement (COSI-Corr)

**Script:** `03_horizontal_displacement_xdem.py`

**What it does:**
- Uses COSI-Corr frequency correlation to track features between hillshades
- Calculates East-West and North-South displacement

**Key Parameters to Tune:**
- `WINDOW_SIZE = 64` → Search window (64px × 0.5m = 32m window)
- `STEP_SIZE = 4` → Output resolution (4px × 0.5m = 2m grid)

**Run it:**
```bash
python 03_horizontal_displacement_xdem.py
```

**Toggle test patch:**
```python
USE_TEST_PATCH = True  # Uses shapefile clipping on-the-fly
```

**Outputs:**
```
results_horizontal_cosicorr/
├── correlation_EW.tif     # East-West displacement (meters)
├── correlation_NS.tif     # North-South displacement (meters)
└── correlation_SNR.tif    # Signal-to-noise ratio (quality)
```

**Troubleshooting:**
- Low SNR? → Increase `WINDOW_SIZE`
- Too coarse? → Decrease `STEP_SIZE`
- No matches? → Check hillshades have texture (not snow-covered)

---

### Step 5: 3D Synthesis & Warping

**Script:** `05_3d_synthesis_xdem.py`

**What it does:**
1. Loads displacement from Step 3
2. Aligns DEMs to displacement grid
3. **Warps** the target DEM backwards
4. Calculates corrected vertical change
5. Computes 3D magnitude

**Run it:**
```bash
python 05_3d_synthesis_xdem.py
```

**Outputs:**
```
results_3d_final/
├── vert_change_raw.tif              # Eulerian (includes topographic effects)
├── vert_change_corrected_melt.tif   # Lagrangian (TRUE vertical change)
├── magnitude_3d.tif                 # √(dx² + dy² + dz²)
├── displacement_EW.tif              # Copy of horizontal components
├── displacement_NS.tif
└── 3d_synthesis_diagnostic.png      # Comparison plots
```

**Interpretation:**
- **Raw vs Corrected:** The difference shows the topographic correction applied
- **Corrected Map (Red):** Surface lowering (melt/deflation)
- **Corrected Map (Blue):** Surface thickening (inflation)

---

## Alternative: Vertical-Only Analysis

**Script:** `vertical_displacement_analysis_FIXED.py`

**When to use:**
- If you only care about vertical displacement
- If horizontal displacement is minimal
- For initial exploration

**Features:**
- Multi-year pairwise comparisons
- Snow detection
- Stable area bias correction
- Time series visualization

```bash
python vertical_displacement_analysis_FIXED.py
```

**Toggle test patch:**
```python
USE_TEST_PATCH = True
```

---

## Test Patch Workflow Summary

**Quick iteration loop for parameter tuning:**

```bash
# One-time setup
python 00_generate_test_patches.py

# Set USE_TEST_PATCH = True in all scripts, then:
python 03_horizontal_displacement_xdem.py  # Fast on small area
python 05_3d_synthesis_xdem.py            # Review results

# Tune WINDOW_SIZE and STEP_SIZE, repeat

# Once satisfied, set USE_TEST_PATCH = False and run on full dataset
```

---

## Key Configuration Variables

### In `03_horizontal_displacement_xdem.py`:
```python
USE_TEST_PATCH = True/False
WINDOW_SIZE = 64    # Larger = more robust, but slower
STEP_SIZE = 4       # Smaller = denser output
```

### In `05_3d_synthesis_xdem.py`:
```python
VMAX_DH = 5.0      # Color scale for elevation change plots
VMAX_MAG = 10.0    # Color scale for magnitude plots
```

### In `vertical_displacement_analysis_FIXED.py`:
```python
USE_TEST_PATCH = True/False
COREG_METHOD = 'nuth_kaab'  # or 'icp', 'deramp'
DETECT_SNOW = True          # Enable/disable snow masking
```

### In `04_DEM_characterization.py`:
```python
USE_TEST_PATCH = True/False
```

---

## Expected Data Structure

```
M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/
│
├── Rasters/
│   ├── 2018_LIDAR/
│   │   └── 2018_0p5m_upper_rg_dem_larger_roi.tif
│   ├── 2023_lidar/
│   │   └── GadValleyRG_50cmDEM_2023.TIF
│   ├── 2024_lidar/
│   │   └── GadValleyRG_50cmDEM_2024.TIF
│   └── 2025_lidar/
│       └── GadValleyRG_50cmDEM_2025.TIF
│
├── shapefiles/
│   ├── stable_2.shp          # Stable ground mask
│   └── small_patch.shp       # Test area
│
└── Code/
    ├── preprocessed_dems/
    │   ├── *_harmonized.tif  # Output from Step 1
    │   └── patches/          # Output from Step 0
    └── rg_velocity/          # This repository
```

---

## Outputs Guide

### `results_horizontal_cosicorr/`
- **correlation_EW.tif**: Positive = eastward motion
- **correlation_NS.tif**: Positive = northward motion
- **correlation_SNR.tif**: Quality metric (0-1, higher is better)

### `results_3d_final/`
- **vert_change_raw.tif**: Simple DEM difference (Eulerian frame)
- **vert_change_corrected_melt.tif**: Flow-corrected elevation change (Lagrangian)
- **magnitude_3d.tif**: Total 3D displacement

### `results_multitemporal_*/`
- Pairwise comparison maps
- Time series plots
- Statistical summaries

---

## Troubleshooting

### COSI-Corr fails with "No module named geoCosiCorr3D"
**Solution:** Check that `COSICORR_SCRIPT` path is correct in `03_horizontal_displacement_xdem.py`

### "Patch shapefile not found"
**Solution:** Verify `PATCH_SHAPEFILE` path. Check Windows vs Linux path format.

### Low correlation (SNR < 0.5)
**Causes:**
- Snow cover changed between epochs
- Shadows changed (different sun angles)
- Window too small for feature size

**Solution:** Increase `WINDOW_SIZE` or use hillshades instead of DEMs

### Warping produces artifacts
**Causes:**
- Displacement grid doesn't match DEM grid
- Invalid displacement values (NaN)

**Solution:** Script handles this automatically via `xdem.reproject()`, but check intermediate outputs

### Memory errors
**Solution:** Use test patches first, or reduce DEM resolution

---

## References

- Nuth, C., & Kääb, A. (2011). Co-registration and bias corrections of satellite elevation data sets for quantifying glacier thickness change. *The Cryosphere*, 5(1), 271-290.
- Dehecq, A., et al. (2015). Deriving large-scale glacier velocities from a complete satellite archive: Application to the Pamir–Karakoram–Himalaya. *Remote Sensing of Environment*, 162, 55-66.
- COSI-Corr: https://github.com/SaifAati/Geospatial-COSICorr3D

---

## Contact & Support

For issues specific to this workflow, check the comments in each script.

For COSI-Corr issues, see: https://github.com/SaifAati/Geospatial-COSICorr3D/issues

---

## Version History

- **v1.0** (2025-11-22): Initial integrated workflow
  - Added test patch support across all scripts
  - Integrated xdem for consistency
  - Created 3D synthesis pipeline with warping
