# Rock Glacier Deformation Analysis - Complete Workflow Guide

## Table of Contents

1. [Overview](#overview)
2. [Scientific Background & Theory](#scientific-background--theory)
3. [Tools & Technologies](#tools--technologies)
4. [Complete Workflow](#complete-workflow)
5. [Error Analysis & Uncertainty](#error-analysis--uncertainty)
6. [Quality Control](#quality-control)
7. [Parameter Selection Guide](#parameter-selection-guide)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [References](#references)

---

## Overview

This repository contains a comprehensive workflow for measuring 3D rock glacier deformation using multi-temporal LiDAR DEMs. The workflow implements the "gold standard" approach: measuring horizontal displacement first (using COSI-Corr), then using that information to warp DEMs and isolate true vertical surface change.

### Key Features

- **3D Deformation Measurement**: Separate horizontal (dx, dy) and vertical (dz) components
- **Topographic Correction**: Lagrangian back-warping to remove slope-induced artifacts
- **Multi-Temporal Analysis**: Support for 4+ epochs (2018-2025 in our case)
- **Uncertainty Quantification**: Built-in error propagation and quality metrics
- **Test Patch Workflow**: Rapid parameter tuning on small subsets
- **Comprehensive QC**: Multiple diagnostic tools for quality assessment

### Repository Structure

```
rg_velocity/
├── WORKFLOW_GUIDE.md                          # This file
│
├── 00_generate_test_patches.py                # Utility: Crop DEMs to test patch
├── preprocess_dems_snowbird.py                # Preprocessing: DEM harmonization
├── 01_horizontal_displacement_cosicorr.py     # Step 1: Horizontal displacement (COSI-Corr)
├── 02_3d_synthesis_backwarp.py                # Step 2: 3D synthesis with back-warping
├── 03_dem_quality_assessment.py               # Step 3: DEM quality assessment
├── 04_vertical_displacement_xdem.py           # Step 4: Alternative vertical-only analysis
│
├── 01_DEM_Diagnostics.py                      # Diagnostic: DEM quality checks
├── 02_Stable_Area_Bias_Diagnostic.py          # Diagnostic: Bias analysis
├── 03_Noise_Source_Diagnostic.py              # Diagnostic: Noise characterization
└── [Other diagnostic scripts]
```

---

## Scientific Background & Theory

### The Challenge: Eulerian vs Lagrangian Frames

When measuring surface change on a moving, sloped surface, we must distinguish between two reference frames:

1. **Eulerian (Fixed Space)**: Measures change at a fixed location
   - Simple DEM differencing: `dH_eulerian = DEM₂(x,y) - DEM₁(x,y)`
   - Mixes real elevation change with topographic effects

2. **Lagrangian (Moving Material)**: Measures change following the material
   - Tracks the same physical parcel of material through time
   - Represents true surface lowering/thickening

### The Problem: Topographic Bias

On sloped terrain with horizontal displacement, Eulerian differencing creates **apparent elevation changes**:

```
For a slope of angle α and horizontal displacement d:
    dH_apparent = d × tan(α)
```

**Example:**
- Horizontal displacement: 2.0 m downslope
- Slope angle: 30°
- Apparent elevation change: 2.0 × tan(30°) ≈ **1.15 m**
- This is noise, not real thinning!

### The Solution: Lagrangian Back-Warping

#### Mathematical Formulation

Given:
- `DEM₁(x, y)` : Reference DEM at time t₁
- `DEM₂(x, y)` : Target DEM at time t₂
- `dx(x, y)` : East-West displacement field (meters)
- `dy(x, y)` : North-South displacement field (meters)

**Step 1: Calculate Eulerian difference (uncorrected)**
```
dH_eulerian(x, y) = DEM₂(x, y) - DEM₁(x, y)
```

**Step 2: Warp DEM₂ using displacement field**

Transform coordinates:
```
x' = x + dx(x, y) / pixel_size_x
y' = y + dy(x, y) / pixel_size_y
```

Interpolate warped DEM:
```
DEM₂_warped(x, y) = DEM₂(x', y')    [using bilinear interpolation]
```

**Step 3: Calculate Lagrangian difference (corrected)**
```
dH_lagrangian(x, y) = DEM₂_warped(x, y) - DEM₁(x, y)
```

This represents **true vertical surface change** (melt, inflation, compaction).

**Step 4: Calculate 3D magnitude**
```
Magnitude_3D(x, y) = √[dx² + dy² + dH_eulerian²]
```

### Physical Interpretation

| Component | Represents | Sign Convention |
|-----------|------------|-----------------|
| `dx` (EW) | Horizontal motion East(+) / West(-) | meters/year |
| `dy` (NS) | Horizontal motion North(+) / South(-) | meters/year |
| `dH_eulerian` | Apparent elevation change | meters/year |
| `dH_lagrangian` | True surface lowering/thickening | meters/year |
| `Magnitude_3D` | Total displacement vector length | meters/year |

**Key Insight**: `dH_eulerian - dH_lagrangian` = topographic correction applied

---

## Tools & Technologies

### Core Libraries

#### 1. **xdem** (eXtended Digital Elevation Model Analysis)
- **Version**: Latest from conda-forge
- **Purpose**: DEM manipulation, co-registration, terrain analysis
- **Key Functions Used**:
  - `xdem.DEM()`: Load and manipulate DEMs with spatial awareness
  - `dem.reproject()`: Align grids with automatic resampling
  - `xdem.coreg.NuthKaab()`: 3D co-registration using stable terrain
  - `dem.slope()`, `dem.aspect()`: Terrain derivatives
- **Advantages**:
  - Built on geoutils (handles CRS, transforms automatically)
  - Masked array support (proper NaN handling)
  - Co-registration methods designed for elevation data

#### 2. **COSI-Corr** (Co-registration of Optically Sensed Images and Correlation)
- **Repository**: https://github.com/SaifAati/Geospatial-COSICorr3D
- **Method**: Frequency-domain correlation (phase correlation)
- **Purpose**: Sub-pixel horizontal displacement tracking
- **Algorithm**:
  - Fourier transform-based image matching
  - Statistical peak detection in frequency space
  - Robustness iterations to remove outliers
- **Outputs**:
  - East-West displacement (meters)
  - North-South displacement (meters)
  - Signal-to-Noise Ratio (SNR) - quality metric
- **Best Used For**:
  - Orthoimages and hillshades (high texture)
  - Large displacements (> 0.5 pixels)
  - Areas with stable features (rocks, boulders)

**Why COSI-Corr over other methods?**
- **vs PIV (Particle Image Velocimetry)**: Better for sparse features
- **vs ICP (Iterative Closest Point)**: Handles deformation (not just rigid motion)
- **vs Optical Flow**: More robust to illumination changes
- **vs Feature Tracking**: Denser output, no manual point selection

#### 3. **GDAL/Rasterio** (Geospatial Data Abstraction Library)
- **Purpose**: Low-level raster I/O
- **Used For**:
  - Reading/writing GeoTIFFs
  - Coordinate transformations
  - Raster metadata handling
- **Note**: xdem wraps these for higher-level operations

#### 4. **GeoPandas** (Geospatial Python)
- **Purpose**: Vector data handling (shapefiles)
- **Used For**:
  - Stable area masks
  - Test patch boundaries
  - ROI clipping

#### 5. **SciPy** (Scientific Python)
- **Key Function**: `scipy.ndimage.map_coordinates()`
- **Purpose**: Bilinear interpolation for DEM warping
- **Advantages**: Fast C implementation, handles NaN gracefully

### Co-Registration Methods

#### Nuth-Kaab (2011) Method
- **Theory**: Uses terrain slope and aspect to estimate 3D shift
- **Equation**: `dH = a × cos(b - aspect) + c + tan(slope)`
- **Solves For**: `(dx, dy, dz)` shift vector
- **Best For**: Glacier/terrain with variable slopes
- **Implementation**: `xdem.coreg.NuthKaab()`

#### ICP (Iterative Closest Point)
- **Theory**: Minimizes point-to-plane distance iteratively
- **Best For**: Point clouds or when rotations suspected
- **Limitation**: Assumes rigid body motion

#### Deramp
- **Theory**: Fits polynomial surface to bias field
- **Best For**: Removing sensor artifacts or long-wavelength tilts
- **Degree**: Typically 1 (plane) or 2 (paraboloid)

---

## Complete Workflow

### Prerequisites

> **⚠️ WINDOWS USERS:** The instructions below use Linux/bash commands. For detailed **Windows-specific** installation instructions, including manual COSI-Corr download, see **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)**.

#### Environment Setup

1. **Install Mamba** (faster than conda):
   ```bash
   conda install -n base -c conda-forge mamba
   ```

2. **Create the environment**:
   ```bash
   # Linux/Mac:
   mamba create -n rock_glacier_env python=3.9 \
       xdem gdal rasterio geopandas numpy scipy matplotlib \
       scikit-image pandas jupyterlab -c conda-forge

   # Windows (use ^ for line continuation):
   mamba create -n rock_glacier_env python=3.9 ^
       xdem gdal rasterio geopandas numpy scipy matplotlib ^
       scikit-image pandas jupyterlab -c conda-forge

   mamba activate rock_glacier_env
   ```

3. **Clone COSI-Corr** (external tool):
   ```bash
   # Linux/Mac:
   cd "M:/My Drive/Rock Glaciers/Tools"
   git clone https://github.com/SaifAati/Geospatial-COSICorr3D.git

   # Windows: Download ZIP from GitHub (see WINDOWS_SETUP.md)
   ```

4. **Verify Installation**:
   ```bash
   python -c "import xdem; import rasterio; import geopandas; print('✓ All packages loaded')"
   ```

---

### Step 0: Generate Test Patches (Optional but Recommended)

**Script:** `00_generate_test_patches.py`

**Purpose:** Create small spatial subsets for rapid parameter tuning and algorithm validation.

**Why Use Test Patches?**
- **Speed**: 100x faster processing on 100m × 100m patch vs full DEM
- **Iteration**: Test multiple parameter combinations quickly
- **Debugging**: Easier to visualize and understand results
- **Memory**: Avoid OOM errors during development

**Run:**
```bash
python 00_generate_test_patches.py
```

**Configuration:**
- `PATCH_SHAPEFILE`: Path to polygon defining test area
- `INPUT_DIR`: Location of harmonized DEMs
- `OUTPUT_DIR`: Typically `preprocessed_dems/patches/`

**Outputs:**
```
preprocessed_dems/patches/
├── 2018_0p5m_upper_rg_dem_larger_roi_harmonized.tif
├── GadValleyRG_50cmDEM_2023_harmonized.TIF
├── GadValleyRG_50cmDEM_2024_harmonized.TIF
└── GadValleyRG_50cmDEM_2025_harmonized.TIF
```

**Best Practices for Test Patches:**
- Choose area with active deformation (not stable ground)
- Include variety of slopes and aspects
- Minimum size: 50m × 50m (100 × 100 pixels at 0.5m)
- Avoid edges (potential for edge effects)

---

### Step 1: DEM Preprocessing & Harmonization

**Script:** `preprocess_dems_snowbird.py`

**Purpose:** Co-register and align all DEMs to a common grid before analysis.

**Why This Step is Critical:**
- LiDAR acquisitions have different origins, rotations, and systematic biases
- Misalignment of ~0.5m can create false displacement signals
- Stable terrain should show zero elevation change

**Process:**

1. **Load Reference DEM** (e.g., 2018)
   - Define master grid (extent, resolution, CRS)

2. **Co-register Each Target DEM**
   - Uses Nuth-Kaab method by default
   - Calculates optimal (dx, dy, dz) shift using stable terrain
   - Applies shift to entire DEM

3. **Resample to Common Grid**
   - Ensures pixel-to-pixel correspondence
   - Uses bilinear interpolation
   - Maintains original resolution (0.5m)

4. **Generate Hillshades**
   - For visualization and feature tracking
   - Azimuth: 315°, Altitude: 45° (standard)

**Run:**
```bash
python preprocess_dems_snowbird.py
```

**Key Parameters:**
- `STABLE_AREA_SHAPEFILE`: Polygon of non-moving terrain
- `COREG_METHOD`: 'nuth_kaab' (default), 'icp', or 'deramp'
- `MASTER_DEM`: Reference epoch (typically earliest acquisition)

**Quality Checks:**
- Inspect stable area statistics (mean dH should be ~0 m)
- Check for systematic patterns (tilts, waves)
- Verify NMAD (Normalized Median Absolute Deviation) < 0.5 m

**Outputs:**
- `*_harmonized.tif` - Co-registered DEMs
- `*_hs_harmonized.tif` - Hillshades

**Typical Co-Registration Performance:**
| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Mean dH (stable) | -0.8 m | 0.02 m | < 0.1 m |
| Std dH (stable) | 0.6 m | 0.25 m | < 0.3 m |
| NMAD (stable) | 0.4 m | 0.18 m | < 0.2 m |

---

### Step 3: DEM Quality Assessment

**Script:** `03_dem_quality_assessment.py`

**Purpose:** Quantify data quality before deformation analysis.

**Metrics Calculated:**

#### 1. **Roughness (Local Standard Deviation)**
```python
roughness = focal_statistics(DEM, window=3x3, statistic='std')
```
- **Interpretation**:
  - Low (<0.1 m): Smooth snow or bare rock
  - Medium (0.1-0.3 m): Boulder fields, vegetated areas
  - High (>0.3 m): Very rough terrain or noisy data
- **Uses**: Identify poor quality regions, mask vegetation

#### 2. **Hillshade Quality**
- Visualize texture for feature tracking
- Check for:
  - Shadows (problematic for correlation)
  - Striping (LiDAR flight line artifacts)
  - Data gaps

#### 3. **Spatial Coverage**
- Percentage of valid pixels
- Identify gaps (vegetation, water)

**Run:**
```bash
python 03_dem_quality_assessment.py
```

**Toggle Test Patch:**
```python
USE_TEST_PATCH = True  # In script
```

**Outputs:**
- `dem_characterization/`
  - `*_hillshade.png` - Visualizations
  - `*_roughness.png` - Roughness maps
  - `dem_characterization_comparison.png` - Multi-epoch comparison
  - `roughness_distributions.png` - Histograms

**Interpretation Guide:**

| Observation | Likely Cause | Action |
|-------------|--------------|--------|
| Uniform high roughness | Vegetation | Mask area |
| Stripes in hillshade | Flight line mismatch | Check preprocessing |
| Patches of NaN | Water, shadows | Expected, exclude from analysis |
| Increasing roughness over time | Snow loss, degradation | Note in interpretation |

---

### Step 1: Horizontal Displacement (COSI-Corr)

**Script:** `01_horizontal_displacement_cosicorr.py`

**Purpose:** Calculate horizontal (dx, dy) displacement field using image correlation.

**Theory - Frequency Domain Correlation:**

COSI-Corr uses phase correlation in Fourier space:

1. Take FFT of reference and target windows
2. Calculate cross-power spectrum: `R = (F₁ × F₂*) / |F₁ × F₂*|`
3. Inverse FFT to get correlation surface
4. Peak location = displacement vector
5. Refine to sub-pixel using parabolic fitting

**Advantages:**
- Robust to illumination changes (uses phase, not amplitude)
- Sub-pixel accuracy (typically 0.1-0.2 pixels)
- Handles larger displacements than spatial methods

**Key Parameters:**

| Parameter | Value | Physical Meaning | Tuning |
|-----------|-------|------------------|--------|
| `WINDOW_SIZE` | 64 px | Search template size (32m at 0.5m res) | Larger = more robust, slower |
| `STEP_SIZE` | 4 px | Output grid spacing (2m at 0.5m res) | Smaller = denser output |
| `CORR_MIN` | 0.9 | Minimum correlation threshold | Higher = fewer but better matches |

**Parameter Selection Guide:**

**Window Size:**
- **Too Small** (< 32 px):
  - Not enough features → Low SNR
  - Ambiguous matches
- **Too Large** (> 128 px):
  - Averages out local deformation
  - Slower processing
  - Requires displacement to be consistent within window
- **Recommended**:
  - Homogeneous motion: 32-64 px
  - Variable motion: 64-96 px
  - Slow motion (<1m): 96-128 px

**Step Size:**
- **Trade-off**: Resolution vs computation time
- `STEP = 1`: Every pixel (very slow)
- `STEP = 2`: Good resolution, manageable
- `STEP = 4`: Balanced (used in this workflow)
- `STEP = 8`: Coarse, fast, good for initial tests

**Run:**
```bash
python 01_horizontal_displacement_cosicorr.py
```

**Configuration:**
```python
USE_TEST_PATCH = True  # Use clipping on-the-fly
IMG_REF = "path/to/2018_hs_harmonized.tif"
IMG_TARGET = "path/to/2023_hs_harmonized.tif"
WINDOW_SIZE = 64
STEP_SIZE = 4
```

**Outputs:**
```
results_horizontal_cosicorr/
├── correlation_EW.tif      # dx: East(+) / West(-), meters
├── correlation_NS.tif      # dy: North(+) / South(-), meters
└── correlation_SNR.tif     # Quality: 0 (poor) - 1 (excellent)
```

**Quality Assessment:**

**SNR Interpretation:**
- **SNR > 0.95**: Excellent match, high confidence
- **SNR 0.85-0.95**: Good match, usable
- **SNR 0.70-0.85**: Moderate match, check visually
- **SNR < 0.70**: Poor match, likely noise

**Visual QC Checklist:**
1. Plot displacement vectors on hillshade
2. Check for spatial coherence (flow should be smooth)
3. Verify stable areas show ~0 displacement
4. Look for outliers (isolated high values)

**Typical Results (Rock Glacier):**
- Mean horizontal velocity: 0.5-3.0 m/year
- Spatial pattern: Fastest at center, slower at margins
- Direction: Generally downslope
- SNR: >0.9 on well-textured surface

---

### Step 2: 3D Synthesis & Warping

**Script:** `02_3d_synthesis_backwarp.py`

**Purpose:** Combine horizontal displacement with elevation data to derive true vertical change.

**Process Overview:**

```
Input:
  ├─ Horizontal displacement (dx, dy) from Step 1
  ├─ Reference DEM (t₁)
  └─ Target DEM (t₂)

Processing:
  ├─ 1. Align DEMs to displacement grid
  ├─ 2. Convert displacement to pixel coordinates
  ├─ 3. Create warping coordinate arrays
  ├─ 4. Interpolate target DEM at displaced positions
  └─ 5. Calculate elevation differences

Output:
  ├─ dH_eulerian (raw difference)
  ├─ dH_lagrangian (flow-corrected)
  └─ Magnitude_3D (total displacement)
```

**Mathematical Detail:**

**1. Grid Alignment**
```python
# Ensure all rasters have identical:
# - Extent (xmin, ymin, xmax, ymax)
# - Resolution (pixel_size_x, pixel_size_y)
# - CRS (coordinate reference system)

dem_ref_aligned = dem_ref.reproject(displacement_grid)
dem_tar_aligned = dem_tar.reproject(displacement_grid)
```

**2. Convert Displacement to Pixels**
```python
dx_pixels = dx_meters / pixel_size_x
dy_pixels = dy_meters / abs(pixel_size_y)  # Note: Y often negative in rasters
```

**3. Create Coordinate Maps**
```python
# Original pixel indices
y_indices, x_indices = np.meshgrid(range(ny), range(nx), indexing='ij')

# Displaced positions (where each pixel came FROM)
y_displaced = y_indices + dy_pixels
x_displaced = x_indices + dx_pixels
```

**4. Warp Target DEM**
```python
# Bilinear interpolation at displaced coordinates
from scipy.ndimage import map_coordinates

dem_tar_warped = map_coordinates(
    dem_tar_aligned,
    [y_displaced, x_displaced],
    order=1,        # Bilinear
    mode='constant',
    cval=np.nan     # Fill value for out-of-bounds
)
```

**5. Calculate Differences**
```python
dH_eulerian = dem_tar_aligned - dem_ref_aligned
dH_lagrangian = dem_tar_warped - dem_ref_aligned
topographic_correction = dH_eulerian - dH_lagrangian
```

**Run:**
```bash
python 02_3d_synthesis_backwarp.py
```

**Outputs:**
```
results_3d_final/
├── vert_change_raw.tif                 # Eulerian (includes topographic effects)
├── vert_change_corrected_melt.tif      # Lagrangian (TRUE vertical change)
├── magnitude_3d.tif                    # √(dx² + dy² + dz²)
├── displacement_EW.tif                 # dx component
├── displacement_NS.tif                 # dy component
├── 3d_synthesis_diagnostic.png         # 6-panel comparison
└── displacement_histograms.png         # Distribution plots
```

**Interpreting Results:**

| Map | Physical Meaning | Expected Pattern (Rock Glacier) |
|-----|------------------|----------------------------------|
| `vert_change_raw` | Apparent elevation change at fixed locations | Dipole: loss upslope, gain downslope |
| `vert_change_corrected_melt` | True surface lowering/thickening | Uniform lowering (ice melt) or neutral |
| `topographic_correction` | Artifact removed by warping | Matches displacement × slope |
| `magnitude_3d` | Total 3D movement | Highest at center, lowest at margins |

**Example Interpretation:**
```
Observation: vert_change_raw shows -2.5m upslope, +2.0m downslope
             vert_change_corrected shows uniform -0.3m everywhere

Interpretation: Rock glacier flowed 2m downslope over the interval.
                On 30° slope, this creates apparent ±2.5m elevation changes.
                After correcting for flow, true surface lowered by 0.3m
                (ice melt from rock glacier interior).
```

---

## Error Analysis & Uncertainty

### Sources of Error

#### 1. **DEM Errors (σ_DEM)**

**Random Errors:**
- Point cloud density: σ ≈ 0.05-0.15 m (high-density LiDAR)
- Interpolation method: ±0.10 m (typical for TIN to raster)
- Roughness: Increases with surface complexity

**Systematic Errors:**
- Co-registration residual: ±0.05-0.20 m (after Nuth-Kaab)
- Atmospheric refraction: Typically negligible for airborne LiDAR
- Flight line mismatches: Up to ±0.30 m if not corrected

**Terrain-Dependent Errors:**
```
σ_terrain = σ_base / cos(slope)
```
- Flat terrain (0°): σ = 0.10 m
- Moderate slope (30°): σ = 0.12 m
- Steep slope (50°): σ = 0.16 m

#### 2. **Correlation Errors (σ_corr)**

**COSI-Corr Uncertainty:**
```
σ_corr ≈ (1 - SNR) × window_size / 4
```

| SNR | Window 64px | Uncertainty (0.5m res) |
|-----|-------------|------------------------|
| 0.95 | 3.2 px | ±1.6 m |
| 0.90 | 6.4 px | ±3.2 m |
| 0.85 | 9.6 px | ±4.8 m |

**Best Practices to Minimize:**
- Use hillshades (not DEMs) for tracking
- Increase window size in low-texture areas
- Mask snow-covered regions
- Filter by SNR threshold (>0.85)

#### 3. **Warping Errors (σ_warp)**

**Interpolation Error:**
```
σ_warp = σ_DEM × √(1 + displacement²/pixel_size²)
```

**Example:**
- DEM error: 0.10 m
- Displacement: 2.0 m
- Pixel size: 0.5 m
- Warping error: 0.10 × √(1 + 16) = **0.41 m**

**Implication:** Large displacements amplify uncertainty!

#### 4. **Temporal Decorrelation**

**Causes:**
- Snow cover changes
- Vegetation growth/removal
- Boulder rearrangement
- Shadow angle differences

**Quantification:**
```
Decorrelation_factor = 1 - (common_features / total_features)
```

**Mitigation:**
- Use acquisitions from similar seasons
- Prefer leaf-off conditions (late fall/early spring)
- Increase window size to average over changes

### Total Uncertainty Calculation

**For Horizontal Displacement:**
```
σ_horizontal = √(σ_corr² + σ_coregistration²)
```

**For Vertical Change (Lagrangian):**
```
σ_vertical = √(σ_DEM₁² + σ_DEM₂² + σ_warp² + (slope × σ_horizontal)²)
```

**For 3D Magnitude:**
```
σ_3D = √[(∂M/∂dx)²σ_dx² + (∂M/∂dy)²σ_dy² + (∂M/∂dz)²σ_dz²]

where M = √(dx² + dy² + dz²)
```

### Uncertainty Propagation Example

**Scenario:** Rock glacier, 5-year interval (2018-2023)

**Inputs:**
- DEM error: 0.15 m (each epoch)
- Correlation error: 0.20 m (SNR = 0.92)
- Co-registration: 0.08 m
- Slope: 25°
- Horizontal displacement: 2.5 m
- Vertical change: -0.4 m

**Calculations:**
```
σ_horizontal = √(0.20² + 0.08²) = 0.22 m

σ_warp = 0.15 × √(1 + (2.5/0.5)²) = 0.77 m

σ_vertical = √(0.15² + 0.15² + 0.77² + (tan(25°) × 0.22)²)
           = √(0.0225 + 0.0225 + 0.5929 + 0.0100)
           = 0.81 m

σ_3D = √[(2.5/2.6 × 0.22)² + (2.5/2.6 × 0.22)² + (0.4/2.6 × 0.81)²]
     = 0.31 m
```

**Annual Rates:**
```
Horizontal velocity: 2.5 m / 5 yr = 0.50 ± 0.04 m/yr
Vertical velocity: -0.4 m / 5 yr = -0.08 ± 0.16 m/yr  (⚠️ High uncertainty!)
3D velocity: 2.6 m / 5 yr = 0.52 ± 0.06 m/yr
```

**Interpretation:**
- Horizontal displacement is **well-resolved** (12 sigma)
- Vertical change is **marginally significant** (0.5 sigma)
- Conclusion: Confidently measure flow, but vertical change near detection limit

### Detection Limits

**Minimum Detectable Displacement:**
```
MDD = 3 × σ_total  (99% confidence)
```

**For Our Setup:**
- Horizontal: MDD = 3 × 0.22 m = **0.66 m** (over 5 years)
- Vertical: MDD = 3 × 0.81 m = **2.43 m** (over 5 years)

**Annual Detection Limits:**
- Horizontal: **0.13 m/year**
- Vertical: **0.49 m/year**

**Implication:** Rock glaciers moving <0.2 m/year may not be detectable!

---

## Quality Control

### Pre-Analysis Checks

#### 1. **DEM Alignment Verification**

**Stable Area Test:**
```bash
python 02_Stable_Area_Bias_Diagnostic.py
```

**Acceptance Criteria:**
| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Mean dH | < ±0.10 m | Minimal bias |
| Median dH | < ±0.05 m | Robust central tendency |
| NMAD | < 0.20 m | Good precision |
| Std dH | < 0.30 m | Acceptable variability |

**If Failed:**
- Re-run co-registration with stricter parameters
- Check stable area polygon (avoid edges, vegetation)
- Inspect for seasonal changes (snow, vegetation)

#### 2. **Correlation Quality (SNR Map)**

**Visual Inspection:**
```python
import rasterio
import matplotlib.pyplot as plt

with rasterio.open('results_horizontal_cosicorr/correlation_SNR.tif') as src:
    snr = src.read(1)

plt.imshow(snr, cmap='RdYlGn', vmin=0.7, vmax=1.0)
plt.colorbar(label='SNR')
plt.title('COSI-Corr Correlation Quality')
plt.show()
```

**Red Flags:**
- Patches of low SNR → Decorrelation (snow, shadow)
- Linear features of low SNR → Flight line boundaries
- Random speckles → Noisy data

**Statistical Checks:**
```python
print(f"Median SNR: {np.nanmedian(snr):.3f}")
print(f"% Pixels SNR > 0.85: {100 * np.sum(snr > 0.85) / snr.size:.1f}%")
```

**Acceptance:**
- Median SNR > 0.85
- At least 70% of pixels with SNR > 0.85

#### 3. **Displacement Coherence**

**Spatial Smoothness Test:**
```python
# Calculate local standard deviation of displacement
from scipy.ndimage import generic_filter

dx = load_displacement('correlation_EW.tif')
dx_variability = generic_filter(dx, np.nanstd, size=5)

# Expected: Low variability in active area (coherent flow)
print(f"Mean local std: {np.nanmean(dx_variability):.3f} m")
```

**Thresholds:**
- Coherent flow: σ_local < 0.5 m
- Noisy/chaotic: σ_local > 1.0 m

#### 4. **Stable Area Displacement Check**

**Should Be Zero:**
```python
stable_mask = load_stable_mask('shapefiles/stable_2.shp')
dx_stable = dx[stable_mask]
dy_stable = dy[stable_mask]

print(f"Stable area mean dx: {np.nanmean(dx_stable):.3f} m")
print(f"Stable area mean dy: {np.nanmean(dy_stable):.3f} m")
```

**Acceptance:**
- |mean| < 0.30 m
- std < 0.50 m

**If Violated:**
- Indicates residual co-registration error
- Apply bias correction before interpreting results

### Post-Analysis Validation

#### 1. **Mass Balance Check (Optional)**

For rock glaciers with known characteristics:
```
Integrated Volume Change = ∫∫ dH_lagrangian dA

Expected: Negative (mass loss) or neutral (equilibrium)
Unlikely: Large positive (mass gain) without accumulation
```

#### 2. **Flow Direction vs Topography**

**Test:**
```python
flow_direction = np.arctan2(dy, dx)  # radians
slope_aspect = calculate_aspect(dem)

# Flow should generally align with downslope
alignment = np.cos(flow_direction - slope_aspect)
print(f"Mean alignment: {np.nanmean(alignment):.2f}")  # Should be > 0.5
```

#### 3. **Temporal Consistency**

If multiple intervals available (e.g., 2018-2023, 2023-2025):
```python
velocity_1 = displacement_1 / time_1
velocity_2 = displacement_2 / time_2

# Should be similar (within uncertainties)
assert np.abs(velocity_1 - velocity_2) < 2 × σ_velocity
```

---

## Parameter Selection Guide

### COSI-Corr Parameters

#### Decision Tree for WINDOW_SIZE

```
START

Is displacement > 1 meter?
├─ YES: Window >= 64 px
└─ NO:  Window >= 96 px (need more context for small motion)

Is surface texture uniform (e.g., sandy)?
├─ YES: Window >= 96 px
└─ NO:  Window >= 64 px OK

Is displacement varying rapidly in space?
├─ YES: Window = 32-48 px (capture local deformation)
└─ NO:  Window = 64-96 px (average for stability)

Compute time constraint?
├─ FAST: Window = 64 px, Step = 8 px
└─ THOROUGH: Window = 96 px, Step = 2 px
```

#### STEP_SIZE Selection

```
Desired Output Resolution = STEP × Input_Resolution

Example: STEP=4, Input=0.5m → Output=2.0m grid spacing
```

**Recommendation:**
- Initial test: STEP = 8 (fast preview)
- Parameter tuning: STEP = 4 (balanced)
- Final run: STEP = 2 (high resolution)
- Publication: STEP = 1 (if computational resources allow)

### Co-Registration Method Selection

| Method | Use When | Pros | Cons |
|--------|----------|------|------|
| **Nuth-Kaab** | General purpose, variable terrain | Fast, robust, analytical solution | Assumes fixed bias |
| **ICP** | Suspected rotation or complex shift | Finds optimal alignment | Slow, can get stuck in local minima |
| **Deramp** | Sensor artifacts, tilt | Removes polynomial trends | Not physical (fitting artifact) |

**Workflow Recommendation:**
1. Start with Nuth-Kaab (default)
2. If stable area shows patterns (not random) → Add Deramp
3. If large rotation suspected (e.g., different sensor) → Try ICP

---

## Troubleshooting

### Common Issues

#### Issue 1: COSI-Corr Returns All NaN

**Symptoms:**
- Displacement files are empty or all NoData
- SNR is uniformly 0

**Causes & Solutions:**

| Cause | Check | Solution |
|-------|-------|----------|
| No texture in hillshades | Open hillshades, look for features | Use different illumination angle, or raw DEMs if very smooth |
| Window too small | Increase `WINDOW_SIZE` to 96 or 128 | |
| DEMs from different seasons | Check acquisition dates | Use same season acquisitions |
| Input files not overlapping | Check spatial extents | Ensure DEMs cover same area |
| COSI-Corr script not found | Verify `COSICORR_SCRIPT` path | Update path in config |

**Debug Command:**
```bash
# Test COSI-Corr directly
python /path/to/geoCosiCorr3D/scripts/correlation.py correlate \
    input1.tif input2.tif --window_size 128 --step 8 --output_dir test/
```

#### Issue 2: SNR is Low (<0.7)

**Cause Analysis:**

**Check 1: Visual Decorrelation**
```python
# Load and compare hillshades side-by-side
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(hillshade_2018, cmap='gray')
ax[0].set_title('2018')
ax[2].imshow(hillshade_2023, cmap='gray')
ax[1].set_title('2023')
plt.show()

# Look for: Snow cover changes, shadow differences, terrain changes
```

**Solutions by Cause:**
- **Snow:** Mask snow-covered areas, or use leaf-off acquisitions
- **Shadows:** Try different hillshade parameters (azimuth, altitude)
- **Vegetation:** Use earlier acquisitions before growth, or mask areas
- **Large displacement:** Increase `WINDOW_SIZE` to 128 or 196

#### Issue 3: Warping Creates Artifacts

**Symptoms:**
- Stripes or checkerboard patterns in `vert_change_corrected_melt.tif`
- Discontinuities along patch boundaries

**Causes:**
1. **Grid misalignment** between displacement and DEM
2. **Invalid displacement values** (NaN or Inf) causing interpolation issues
3. **Edge effects** from warping near boundaries

**Solutions:**

**Check Grid Alignment:**
```python
import rasterio

with rasterio.open('results_horizontal_cosicorr/correlation_EW.tif') as src_disp:
    disp_res = src_disp.res
    disp_bounds = src_disp.bounds

with rasterio.open('preprocessed_dems/2023_harmonized.tif') as src_dem:
    dem_res = src_dem.res
    dem_bounds = src_dem.bounds

print(f"Displacement: res={disp_res}, bounds={disp_bounds}")
print(f"DEM: res={dem_res}, bounds={dem_bounds}")

# Should be identical (or script handles with reproject)
```

**Filter Invalid Displacements:**
```python
# In 02_3d_synthesis_backwarp.py, add:
dx[np.isnan(dx)] = 0
dy[np.isnan(dy)] = 0
dx[np.abs(dx) > 20] = 0  # Flag unrealistic displacements
dy[np.abs(dy) > 20] = 0
```

**Mask Edges:**
```python
# Create buffer mask
from scipy.ndimage import binary_erosion

valid_mask = ~np.isnan(dx)
valid_mask_eroded = binary_erosion(valid_mask, iterations=5)

# Apply to results
dh_corrected[~valid_mask_eroded] = np.nan
```

#### Issue 4: "Memory Error" During Processing

**Solutions:**

1. **Use Test Patches**
   ```python
   USE_TEST_PATCH = True  # In scripts
   ```

2. **Process in Chunks**
   ```python
   # Manually divide DEM into tiles
   from rasterio.windows import Window

   for row in range(0, dem.height, 1000):
       for col in range(0, dem.width, 1000):
           window = Window(col, row, 1000, 1000)
           # Process chunk...
   ```

3. **Increase Virtual Memory**
   - Windows: Settings → System → About → Advanced system settings → Performance Settings → Advanced → Virtual memory
   - Linux: Increase swap space

4. **Reduce Precision**
   ```python
   dem = dem.astype(np.float32)  # Instead of float64
   ```

#### Issue 5: Vertical Change Looks Like Horizontal Flow

**Symptoms:**
- `vert_change_corrected_melt` still shows dipole pattern
- Correction seems ineffective

**Likely Cause:** Horizontal displacement and DEM are not from correct time intervals.

**Check:**
```python
# Ensure:
# - Displacement: 2018 → 2023
# - DEM_ref: 2018
# - DEM_tar: 2023

# NOT:
# - Displacement: 2018 → 2023
# - DEM_ref: 2018
# - DEM_tar: 2024  ← WRONG!
```

**Solution:** Match epochs exactly.

---

## Best Practices

### 1. Workflow Execution

**Iterative Approach:**
```
1. Generate test patch (00_generate_test_patches.py)
2. Run Step 3 on patch with default parameters
3. Inspect SNR, adjust WINDOW_SIZE
4. Repeat Step 3 until SNR > 0.85
5. Run Step 5 on patch, check results
6. If satisfied, switch USE_TEST_PATCH = False
7. Run full dataset
8. Quality control (check stable areas)
9. Calculate uncertainties
10. Interpret results
```

### 2. Data Management

**File Naming Convention:**
```
<site>_<resolution>_<epoch>_<processing_level>_<product>.tif

Examples:
GadValley_50cm_2018_L1_raw.tif
GadValley_50cm_2018_L2_harmonized.tif
GadValley_50cm_2018_L3_displacement_EW.tif
GadValley_50cm_2018-2023_L4_vert_corrected.tif
```

**Directory Structure:**
```
project/
├── 01_raw/          # Original LiDAR DEMs
├── 02_preprocessed/ # Harmonized DEMs
├── 03_displacement/ # COSI-Corr outputs
├── 04_3d_analysis/  # Final products
├── 05_qc/           # Quality control plots
└── 06_figures/      # Publication figures
```

### 3. Documentation

**Keep a Processing Log:**
```markdown
# Processing Log - Gad Valley Rock Glacier

## 2024-11-22: Initial Correlation Attempt
- Parameters: WINDOW=64, STEP=4
- Result: SNR = 0.78 (too low)
- Action: Increase WINDOW to 96

## 2024-11-22: Second Attempt
- Parameters: WINDOW=96, STEP=4
- Result: SNR = 0.91 (acceptable)
- Notes: Some low SNR in upper section (snow remnants)

## 2024-11-23: Full Dataset
- Used WINDOW=96, STEP=2 for final run
- Processing time: 3.2 hours
- Results stored in results_3d_final_20241123/
```

### 4. Version Control

**Git Workflow:**
```bash
# Branch for parameter experiments
git checkout -b experiment/window_size_tuning

# After finding good parameters
git add config.py results_summary.md
git commit -m "Optimized WINDOW_SIZE to 96px for SNR >0.9"

# Merge to main
git checkout main
git merge experiment/window_size_tuning
```

### 5. Reproducibility

**Create a Parameters File:**
```python
# analysis_config.yaml
workflow:
  name: "Gad Valley Rock Glacier 3D Deformation"
  version: "2024-11-22"

epochs:
  reference: "2018-09-15"
  target: "2023-09-20"

preprocessing:
  coreg_method: "nuth_kaab"
  stable_area: "shapefiles/stable_2.shp"

correlation:
  tool: "COSI-Corr"
  window_size: 96
  step_size: 2
  corr_min: 0.9

synthesis:
  interpolation: "bilinear"
  edge_buffer: 10  # pixels
```

---

## References

### Key Publications

#### Co-Registration & Bias Correction
- **Nuth, C., & Kääb, A. (2011).** Co-registration and bias corrections of satellite elevation data sets for quantifying glacier thickness change. *The Cryosphere*, 5(1), 271-290. [https://doi.org/10.5194/tc-5-271-2011](https://doi.org/10.5194/tc-5-271-2011)
  - **Key Contribution:** 3D co-registration using slope/aspect relationship
  - **Application:** All DEM alignment in this workflow

#### Image Correlation
- **Leprince, S., Barbot, S., Ayoub, F., & Avouac, J. P. (2007).** Automatic and precise orthorectification, coregistration, and subpixel correlation of satellite images, application to ground deformation measurements. *IEEE TGRS*, 45(6), 1529-1558.
  - **Key Contribution:** COSI-Corr algorithm development
  - **Application:** Step 3 of this workflow

- **Dehecq, A., et al. (2015).** Deriving large-scale glacier velocities from a complete satellite archive: Application to the Pamir–Karakoram–Himalaya. *Remote Sensing of Environment*, 162, 55-66.
  - **Key Contribution:** Large-scale correlation workflow
  - **Application:** Multi-temporal displacement tracking

#### Lagrangian Analysis
- **Berthier, E., et al. (2007).** Remote sensing estimates of glacier mass balances in the Himachal Pradesh (Western Himalaya, India). *Remote Sensing of Environment*, 108(3), 327-338.
  - **Key Contribution:** Separating vertical from horizontal effects
  - **Application:** Warping methodology (Step 5)

#### Rock Glacier Kinematics
- **Kääb, A., et al. (2021).** Sudden large-volume detachments of low-angle mountain glaciers – more frequent than thought? *The Cryosphere*, 15, 1751–1785.
  - **Context:** Rock glacier mechanics and failure modes

- **RGIK (2022).** Rock Glacier Kinematics Benchmark. [https://rockglacier.info](https://rockglacier.info)
  - **Resource:** Global rock glacier inventory and velocity database

### Software & Tools

#### Core Libraries
- **xdem**: [https://github.com/GlacioHack/xdem](https://github.com/GlacioHack/xdem)
  - Documentation: [https://xdem.readthedocs.io](https://xdem.readthedocs.io)

- **COSI-Corr**: [https://github.com/SaifAati/Geospatial-COSICorr3D](https://github.com/SaifAati/Geospatial-COSICorr3D)
  - Original: [http://www.tectonics.caltech.edu/slip_history/spot_coseis/index.html](http://www.tectonics.caltech.edu/slip_history/spot_coseis/index.html)

- **GDAL**: [https://gdal.org](https://gdal.org)
- **Rasterio**: [https://rasterio.readthedocs.io](https://rasterio.readthedocs.io)
- **GeoPandas**: [https://geopandas.org](https://geopandas.org)

#### Related Tools (Not Used but Worth Knowing)
- **GIV (PIV Toolbox)**: https://github.com/eguvep/givpiv
- **ImGRAFT**: https://github.com/grinsted/ImGRAFT
- **PyTrx**: https://github.com/PennyHow/PyTrx

### Uncertainty Quantification
- **Höhle, J., & Höhle, M. (2009).** Accuracy assessment of digital elevation models by means of robust statistical methods. *ISPRS Journal of Photogrammetry and Remote Sensing*, 64(4), 398-406.
  - **Key Metric:** NMAD (Normalized Median Absolute Deviation)

---

## Contact & Support

### Workflow Questions
Check the inline comments in each script for parameter explanations.

### COSI-Corr Issues
- GitHub Issues: [https://github.com/SaifAati/Geospatial-COSICorr3D/issues](https://github.com/SaifAati/Geospatial-COSICorr3D/issues)
- Contact: Saif Aati (developer)

### xdem Issues
- Documentation: [https://xdem.readthedocs.io](https://xdem.readthedocs.io)
- GitHub: [https://github.com/GlacioHack/xdem](https://github.com/GlacioHack/xdem)

---

## Version History

### v1.1 (2025-11-22)
- **Enhanced Documentation**:
  - Added comprehensive error analysis section
  - Detailed uncertainty quantification formulas
  - Expanded troubleshooting with solutions
  - Tool descriptions with theoretical background
- **Technical Additions**:
  - Parameter selection decision trees
  - Quality control metrics and thresholds
  - Best practices for reproducibility
  - Example calculations for uncertainty propagation

### v1.0 (2025-11-22)
- Initial integrated workflow
- Test patch support across all scripts
- Integrated xdem for consistency
- Created 3D synthesis pipeline with warping

---

## Appendix: Quick Reference

### Processing Checklist

```
☐ Environment setup (mamba, COSI-Corr clone)
☐ Generate test patches (00_generate_test_patches.py)
☐ Preprocess DEMs (preprocess_dems_snowbird.py)
☐ Check stable area statistics (< 0.1m bias)
☐ Characterize DEM quality (04_DEM_characterization.py)
☐ Run correlation on test patch (01_horizontal_displacement_cosicorr.py)
☐ Check SNR (> 0.85 median)
☐ Tune WINDOW_SIZE if needed
☐ Run 3D synthesis on test patch (02_3d_synthesis_backwarp.py)
☐ Verify results make physical sense
☐ Switch to full dataset (USE_TEST_PATCH = False)
☐ Re-run Steps 3 & 5
☐ Calculate uncertainties
☐ Generate figures
☐ Document processing log
```

### Command Summary

```bash
# Setup
mamba activate rock_glacier_env

# Quick Test (Patch)
python 00_generate_test_patches.py
python 01_horizontal_displacement_cosicorr.py  # USE_TEST_PATCH = True
python 02_3d_synthesis_backwarp.py

# Full Processing
python preprocess_dems_snowbird.py
python 03_dem_quality_assessment.py
python 01_horizontal_displacement_cosicorr.py  # USE_TEST_PATCH = False
python 02_3d_synthesis_backwarp.py

# Quality Control
python 02_Stable_Area_Bias_Diagnostic.py
```

### File Size Estimates

| Dataset | Test Patch (100×100m) | Full DEM (1×1 km) |
|---------|----------------------|-------------------|
| Input DEM | ~500 KB | ~50 MB |
| Hillshade | ~500 KB | ~50 MB |
| Displacement | ~100 KB | ~10 MB |
| Final Products | ~300 KB | ~30 MB |
| Processing Time (Step 3) | ~2 min | ~1-2 hours |

### Typical Parameter Values

| Parameter | Conservative | Balanced | Aggressive |
|-----------|--------------|----------|------------|
| WINDOW_SIZE | 128 px | 64 px | 32 px |
| STEP_SIZE | 8 px | 4 px | 1 px |
| CORR_MIN | 0.95 | 0.85 | 0.70 |
| Processing Time | Slow | Medium | Fast |
| Output Density | Coarse | Medium | Dense |

---

**End of Guide** - Last updated: 2025-11-22
