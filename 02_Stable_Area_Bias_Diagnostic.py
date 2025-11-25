"""
02_Stable_Area_Bias_Diagnostic.py
==================================

This script diagnoses the spatial pattern of vertical bias on stable ground
BEFORE any co-registration or bias correction is applied.

It helps answer:
1. Is the bias on stable ground random noise (good)?
2. Is there a spatial pattern (tilt, dome) in the bias (bad)?
3. How does the 2018-2023 bias compare to the 2023-2024 bias?
"""

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from rasterio.features import geometry_mask
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='geoutils')
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# ============================================================
# CONFIGURATION
# ============================================================

# Use the EXACT DEMs from your '04_vertical_displacement_xdem.py'
DEM_2018_ORIG = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_LIDAR/2018_0p5m_upper_rg_dem_larger_roi.tif"
DEM_2023_ORIG = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2023_lidar/2023_0p5m_4_imcorr_upper_rg_larger_roi.tif"
DEM_2024_HARM = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems/2024_0p5_4_imcorr__upper_rg_larger_roi_harmonized.tif"

STABLE_MASK_SHP = r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\shapefiles\stable_2.shp"

output_dir = Path("dem_diagnostics")
output_dir.mkdir(exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dem(path):
    """Load DEM and return data, metadata."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        
        # Handle nodata
        nodata_mask = np.isnan(data) | np.isinf(data) | (data < -1e10) | (data > 10000)
        if src.nodata is not None and np.isfinite(src.nodata):
            nodata_mask |= (data == src.nodata)
        
        data[nodata_mask] = np.nan
        
        metadata = {
            'shape': data.shape,
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds
        }
        
        return data, metadata

def load_stable_mask(shp_path, dem_meta):
    """Load, reproject, and rasterize the stable mask."""
    print(f"  Loading stable mask: {Path(shp_path).name}")
    stable_gdf = gpd.read_file(shp_path)
    
    if stable_gdf.crs != dem_meta['crs']:
        print(f"  Reprojecting mask from {stable_gdf.crs} to {dem_meta['crs']}")
        stable_gdf = stable_gdf.to_crs(dem_meta['crs'])
        
    mask = geometry_mask(
        stable_gdf.geometry,
        transform=dem_meta['transform'],
        invert=True,
        out_shape=dem_meta['shape']
    )
    print(f"  Mask pixels: {np.sum(mask)}")
    return mask

# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def run_bias_diagnostic():
    print("="*70)
    print("STABLE AREA BIAS DIAGNOSTIC")
    print("="*70)

    # Step 1: Load DEMs
    print("\nLoading DEMs...")
    dem_2018, meta = load_dem(DEM_2018_ORIG)
    dem_2023, _ = load_dem(DEM_2023_ORIG)
    dem_2024, _ = load_dem(DEM_2024_HARM)
    
    # Step 2: Load Stable Mask
    stable_mask = load_stable_mask(STABLE_MASK_SHP, meta)
    
    # Step 3: Calculate Raw Differences
    print("\nCalculating raw differences...")
    dh_18_23 = dem_2023 - dem_2018
    dh_23_24 = dem_2024 - dem_2023
    
    # Step 4: Extract Stable Ground Data
    mask_18_23 = stable_mask & np.isfinite(dh_18_23)
    dh_stable_18_23 = dh_18_23[mask_18_23]
    
    mask_23_24 = stable_mask & np.isfinite(dh_23_24)
    dh_stable_23_24 = dh_23_24[mask_23_24]
    
    # Step 5: Print Statistics
    print("\n" + "-"*70)
    print("Stable Ground Raw Statistics (Pre-Correction)")
    print(f"2018 -> 2023:")
    print(f"  Mean:   {np.mean(dh_stable_18_23):.4f} m")
    print(f"  Median: {np.median(dh_stable_18_23):.4f} m")
    print(f"  Std:    {np.std(dh_stable_18_23):.4f} m")
    print(f"  NMAD:   {1.4826 * np.median(np.abs(dh_stable_18_23 - np.median(dh_stable_18_23))):.4f} m")
    
    print(f"\n2023 -> 2024:")
    print(f"  Mean:   {np.mean(dh_stable_23_24):.4f} m")
    print(f"  Median: {np.median(dh_stable_23_24):.4f} m")
    print(f"  Std:    {np.std(dh_stable_23_24):.4f} m")
    print(f"  NMAD:   {1.4826 * np.median(np.abs(dh_stable_23_24 - np.median(dh_stable_23_24))):.4f} m")
    print("-" * 70)

    # Step 6: Create Diagnostic Plot
    print("\nGenerating diagnostic plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Histogram 2018-2023
    axes[0, 0].hist(dh_stable_18_23, bins=100, range=(-1, 1), density=True, alpha=0.7)
    axes[0, 0].axvline(np.mean(dh_stable_18_23), color='red', ls='--', label=f"Mean: {np.mean(dh_stable_18_23):.3f}")
    axes[0, 0].axvline(np.median(dh_stable_18_23), color='blue', ls=':', label=f"Median: {np.median(dh_stable_18_23):.3f}")
    axes[0, 0].set_title("2018-2023: Stable Ground Raw Bias")
    axes[0, 0].set_xlabel("Difference (m)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    
    # Plot 2: Histogram 2023-2024
    axes[0, 1].hist(dh_stable_23_24, bins=100, range=(-1, 1), density=True, alpha=0.7)
    axes[0, 1].axvline(np.mean(dh_stable_23_24), color='red', ls='--', label=f"Mean: {np.mean(dh_stable_23_24):.3f}")
    axes[0, 1].axvline(np.median(dh_stable_23_24), color='blue', ls=':', label=f"Median: {np.median(dh_stable_23_24):.3f}")
    axes[0, 1].set_title("2023-2024: Stable Ground Raw Bias")
    axes[0, 1].set_xlabel("Difference (m)")
    axes[0, 1].legend()
    
    # Prepare 2D maps
    dh_map_18_23 = dh_18_23.copy()
    dh_map_18_23[~mask_18_23] = np.nan
    
    dh_map_23_24 = dh_23_24.copy()
    dh_map_23_24[~mask_23_24] = np.nan
    
    vmax = np.nanpercentile(np.abs(dh_stable_23_24), 95)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    # Plot 3: 2D Bias Map 2018-2023
    im1 = axes[1, 0].imshow(dh_map_18_23, cmap='RdBu_r', norm=norm)
    axes[1, 0].set_title("2018-2023: Spatial Bias Pattern (Stable)")
    plt.colorbar(im1, ax=axes[1, 0], label="Difference (m)")
    
    # Plot 4: 2D Bias Map 2023-2024
    im2 = axes[1, 1].imshow(dh_map_23_24, cmap='RdBu_r', norm=norm)
    axes[1, 1].set_title("2023-2024: Spatial Bias Pattern (Stable)")
    plt.colorbar(im2, ax=axes[1, 1], label="Difference (m)")
    
    plt.tight_layout()
    plot_path = output_dir / 'stable_bias_diagnostic.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✓ Diagnostic plot saved to: {plot_path}")

if __name__ == "__main__":
    run_bias_diagnostic()