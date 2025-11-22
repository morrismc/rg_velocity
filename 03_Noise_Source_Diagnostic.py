"""
03_Noise_Source_Diagnostic.py
==============================

This script tests the hypothesis that the high noise in the 2024-2025
period is correlated with terrain complexity (roughness).

A strong correlation would suggest that the noise is not random,
but is a systematic artifact from the DEM generation process,
likely a difference in the bare-earth (terrain) filtering
parameters between the two surveys.
"""

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.features import geometry_mask
from scipy import ndimage, stats
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='geoutils')
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# ============================================================
# CONFIGURATION
# ============================================================

# Point to the two DEMs from the problematic period
DEM_2024_HARM = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems/GadValleyRG_50cmDEM_2024_harmonized.TIF"
DEM_2025_HARM = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems/GadValleyRG_50cmDEM_2025_harmonized.TIF"

STABLE_MASK_SHP = r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\shapefiles\stable_2.shp"

output_dir = Path("dem_diagnostics")
output_dir.mkdir(exist_ok=True)

# Window size for roughness calculation (3x3)
ROUGHNESS_WINDOW_SIZE = 3

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dem(path):
    """Load DEM and return data, metadata."""
    print(f"  Loading DEM: {Path(path).name}")
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
            'crs': src.crs
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

def run_noise_diagnostic():
    print("="*70)
    print("NOISE SOURCE DIAGNOSTIC (2024-2025)")
    print("="*70)

    # Step 1: Load Data
    print("\nLoading data...")
    dem_2024, meta = load_dem(DEM_2024_HARM)
    dem_2025, _ = load_dem(DEM_2025_HARM)
    stable_mask = load_stable_mask(STABLE_MASK_SHP, meta)
    
    # Step 2: Calculate Raw Difference (Noise)
    print("\nCalculating raw difference (noise)...")
    dh_raw = dem_2025 - dem_2024
    dh_abs = np.abs(dh_raw)
    
    # Step 3: Calculate Terrain Roughness
    # We use the standard deviation of elevation in a 3x3 window
    print(f"\nCalculating terrain roughness ({ROUGHNESS_WINDOW_SIZE}x{ROUGHNESS_WINDOW_SIZE} StdDev)...")
    roughness = ndimage.generic_filter(
        dem_2024, 
        np.std, 
        size=ROUGHNESS_WINDOW_SIZE, 
        mode='constant', 
        cval=np.nan
    )
    
    # Step 4: Prepare data for plotting
    print("\nPreparing data for plotting (on stable ground)...")
    
    # Create a common mask where all data is valid
    valid_mask = np.isfinite(dh_abs) & np.isfinite(roughness) & stable_mask
    
    if not np.any(valid_mask):
        print("❌ ERROR: No valid data found on stable mask. Cannot proceed.")
        return

    x_data = roughness[valid_mask]
    y_data = dh_abs[valid_mask]
    
    print(f"  Found {len(x_data)} valid data points on stable ground.")

    # Step 5: Calculate Correlation
    print("\nCalculating correlation...")
    # Use a sample to avoid memory issues if data is huge
    if len(x_data) > 100000:
        print(f"  Sampling 100,000 points for correlation...")
        sample_idx = np.random.choice(len(x_data), 100000, replace=False)
        x_sample = x_data[sample_idx]
        y_sample = y_data[sample_idx]
    else:
        x_sample = x_data
        y_sample = y_data

    corr, p_value = stats.pearsonr(x_sample, y_sample)
    print(f"  Pearson Correlation: {corr:.4f}")
    print(f"  p-value: {p_value:.2e}")

    # Step 6: Create Diagnostic Plot
    print("\nGenerating diagnostic plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    
    # Plot 1: Terrain Roughness Map
    vmax_rough = np.nanpercentile(roughness, 95)
    im1 = axes[0, 0].imshow(roughness, cmap='viridis', vmin=0, vmax=vmax_rough)
    axes[0, 0].set_title("Terrain Roughness (2024 DEM)")
    plt.colorbar(im1, ax=axes[0, 0], label="Elevation StdDev (m)")
    
    # Plot 2: Absolute Noise Map
    vmax_noise = np.nanpercentile(dh_abs, 95)
    im2 = axes[0, 1].imshow(dh_abs, cmap='inferno', vmin=0, vmax=vmax_noise)
    axes[0, 1].set_title("Absolute Noise |dh_2025 - dh_2024|")
    plt.colorbar(im2, ax=axes[0, 1], label="Absolute Difference (m)")
    
    # Plot 3: 2D Density Plot (Hexbin)
    # This shows where the majority of points fall
    im3 = axes[1, 0].hexbin(
        x_sample, 
        y_sample, 
        gridsize=100, 
        cmap='inferno', 
        mincnt=1,
        vmax=np.percentile(np.histogram2d(x_sample, y_sample, bins=100)[0], 98)
    )
    axes[1, 0].set_xlabel("Terrain Roughness (StdDev)")
    axes[1, 0].set_ylabel("Absolute Noise (m)")
    axes[1, 0].set_title("Noise vs. Roughness (Density Plot)")
    plt.colorbar(im3, ax=axes[1, 0], label="Point Count")
    
    # Plot 4: Binned Statistics (The clearest plot)
    # Bin the data by roughness and plot the median noise
    try:
        bins = np.linspace(np.nanmin(x_data), np.nanpercentile(x_data, 99), 30)
        binned_median_noise, bin_edges, _ = stats.binned_statistic(
            x_data, y_data, statistic='median', bins=bins
        )
        binned_p75_noise, _, _ = stats.binned_statistic(
            x_data, y_data, statistic=lambda x: np.percentile(x, 75), bins=bins
        )
        binned_p25_noise, _, _ = stats.binned_statistic(
            x_data, y_data, statistic=lambda x: np.percentile(x, 25), bins=bins
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        axes[1, 1].plot(bin_centers, binned_median_noise, 'o-', color='red', label='Median Noise')
        axes[1, 1].fill_between(
            bin_centers, 
            binned_p25_noise, 
            binned_p75_noise, 
            color='red', 
            alpha=0.2, 
            label='Interquartile Range'
        )
        axes[1, 1].set_xlabel("Binned Terrain Roughness (StdDev)")
        axes[1, 1].set_ylabel("Median Noise (m)")
        axes[1, 1].set_title("Median Noise vs. Roughness")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.5)
    except ValueError as e:
        print(f"  Could not create binned plot: {e}")
        axes[1, 1].text(0.5, 0.5, "Error creating binned plot", ha='center', va='center')

    
    fig.suptitle(
        f"Noise Source Diagnostic (2024-2025)\n"
        f"Pearson Correlation (on stable ground): {corr:.3f}",
        fontsize=16,
        fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plot_path = output_dir / '03_noise_source_diagnostic.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Diagnostic plot saved to: {plot_path}")

if __name__ == "__main__":
    run_noise_diagnostic()