"""
04_DEM_Characterization_Parallel.py
====================================

This script generates and compares key topographic metrics for the full
time series of DEMs (2018, 2023, 2024, 2025).

It is intended to help visualize differences in the underlying data quality,
particularly in terrain texture and noise, to inform discussions with
the lidar processing team.

This version is PARALLELIZED for speed and includes an additional
roughness distribution (PDF/CDF) comparison plot.

It produces four main outputs:
1.  A set of GeoTIFF files (one hillshade and one roughness map per year)
    saved in a new 'dem_characterization' directory.
2.  A comparison plot ('dem_characterization_comparison.png')
    showing all 8 maps for easy side-by-side evaluation.
3.  A distribution plot ('dem_roughness_distributions.png')
    comparing the PDF and CDF of roughness for all DEMs.
4.  All plots are also shown in the IDE.
"""

import numpy as np
import rasterio
from rasterio.profiles import Profile
from scipy import ndimage, stats
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from joblib import Parallel, delayed

# Suppress runtime warnings (e.g., from nan-slices)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

def generate_hillshade(dem, pixel_size, azimuth=315, altitude=45):
    """
    Generates a hillshade array from a DEM using numpy.
    """
    # Convert angles to radians
    az_rad = np.deg2rad(azimuth)
    alt_rad = np.deg2rad(altitude)
    
    # Calculate slope and aspect
    dy, dx = np.gradient(dem, pixel_size, pixel_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dx, dy)
    
    # Calculate illumination
    shaded = np.sin(alt_rad) * np.sin(slope_rad) + \
             np.cos(alt_rad) * np.cos(slope_rad) * \
             np.cos(az_rad - aspect_rad)
    
    # Scale to 0-255
    shaded_8bit = np.nan_to_num(shaded, nan=-2) # Use temp value
    
    # Scale valid pixels
    min_val, max_val = np.min(shaded_8bit[shaded_8bit > -2]), np.max(shaded_8bit)
    if (max_val - min_val) > 0:
        shaded_8bit[shaded_8bit > -2] = ((shaded[shaded_8bit > -2] - min_val) / (max_val - min_val) * 255)
    shaded_8bit[shaded_8bit == -2] = 0 # Set nodata to 0
    
    return shaded_8bit.astype(np.uint8)

def calculate_roughness(dem, window_size=3):
    """
    Calculates terrain roughness using the standard deviation
    in a moving window.
    """
    # Use generic_filter to apply np.std to a moving window
    # np.nanstd handles nodata values gracefully
    roughness = ndimage.generic_filter(
        dem, 
        np.nanstd, 
        size=window_size, 
        mode='constant', 
        cval=np.nan
    )
    return roughness.astype(np.float32)

def process_single_dem(year, dem_path, pixel_size, output_dir):
    """
    A self-contained function to process one DEM.
    This is designed to be called in parallel by joblib.
    """
    print(f"Processing: {year} ({dem_path.name})")
    
    if not dem_path.exists():
        print(f"  ⚠️ WARNING: File not found, skipping: {dem_path}")
        return {year: None}
        
    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float64)
            profile = src.profile.copy()
            
            # Handle NoData
            if src.nodata is not None:
                dem[dem == src.nodata] = np.nan
            dem[dem < -1000] = np.nan # Catch erroneous values
        
        # --- Generate Hillshade ---
        print(f"  Generating hillshade for {year}...")
        hillshade_array = generate_hillshade(dem, pixel_size)
        
        # Save Hillshade GeoTIFF
        hs_path = output_dir / f"{year}_hillshade.tif"
        hs_profile = profile
        hs_profile.update(dtype='uint8', nodata=0, count=1, compress='lzw')
        
        with rasterio.open(hs_path, 'w', **hs_profile) as dst:
            dst.write(hillshade_array, 1)
        print(f"  ✓ Saved: {hs_path.name}")

        # --- Calculate Roughness ---
        print(f"  Calculating roughness for {year}...")
        roughness_array = calculate_roughness(dem)
        
        # Save Roughness GeoTIFF
        rough_path = output_dir / f"{year}_roughness.tif"
        rough_profile = profile
        rough_profile.update(dtype='float32', nodata=np.nan, count=1, compress='lzw')
        
        with rasterio.open(rough_path, 'w', **rough_profile) as dst:
            dst.write(roughness_array, 1)
        print(f"  ✓ Saved: {rough_path.name}")
        
        # Return results
        return {year: {
            'hillshade': hillshade_array,
            'roughness': roughness_array
        }}
        
    except Exception as e:
        print(f"  ❌ ERROR processing {year}: {e}")
        return {year: None}

def plot_roughness_distributions(results, output_dir):
    """
    Creates a new plot comparing the PDF and CDF of roughness
    for all processed DEMs.
    """
    print("\n" + "="*60)
    print("GENERATING ROUGHNESS DISTRIBUTION PLOT")
    print("="*60)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    all_data = [] # To find overall x-limit
    
    for (year, data), color in zip(results.items(), colors):
        if data is None:
            continue
        
        rough_data = data['roughness']
        valid_data = rough_data[~np.isnan(rough_data)]
        all_data.append(valid_data)
        
        # --- PDF (Kernel Density Estimate) ---
        try:
            kde = stats.gaussian_kde(valid_data)
            x_range = np.linspace(0, np.percentile(valid_data, 99.5), 500)
            pdf = kde(x_range)
            axes[0].plot(x_range, pdf, label=f"{year}", color=color, linewidth=2)
        except Exception as e:
            print(f"  Could not generate PDF for {year}: {e}")

        # --- CDF (Cumulative Distribution) ---
        data_sorted = np.sort(valid_data)
        y_cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        axes[1].plot(data_sorted, y_cdf, label=f"{year}", color=color, linewidth=2)

    # --- Format PDF Plot ---
    axes[0].set_title("Probability Density Function (PDF) of Terrain Roughness", fontweight='bold')
    axes[0].set_ylabel("Density")
    axes[0].legend() # <-- This was the line with the SyntaxError
    axes[0].grid(True, alpha=0.5)
    
    # --- Format CDF Plot ---
    axes[1].set_title("Cumulative Distribution Function (CDF) of Terrain Roughness", fontweight='bold')
    axes[1].set_xlabel("Terrain Roughness (StdDev in 3x3 window)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].legend()
    axes[1].grid(True, alpha=0.5)

    # Set a consistent X-axis limit based on 99th percentile of all data
    if all_data:
        try:
            global_vmax = np.percentile(np.concatenate(all_data), 99)
            axes[0].set_xlim(0, global_vmax)
            axes[1].set_xlim(0, global_vmax)
        except Exception as e:
            print(f"Could not set xlims: {e}")

    plt.tight_layout()
    plot_path = output_dir / "dem_roughness_distributions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Distribution plot saved to: {plot_path.name}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"  Note: Could not show plot in IDE. {e}")
        
    plt.close()

def main():
    """
    Main function to run the characterization workflow.
    """

    # --- 1. Configuration ---
    BASE_DIR = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems")

    # --- 🆕 TEST PATCH TOGGLE 🆕 ---
    USE_TEST_PATCH = False  # Set to True to use small patches

    if USE_TEST_PATCH:
        DEMS_DIR = BASE_DIR / "patches"
        OUTPUT_DIR = Path("dem_characterization_patch")
        print("🔍 RUNNING IN TEST PATCH MODE")
    else:
        DEMS_DIR = BASE_DIR
        OUTPUT_DIR = Path("dem_characterization")
        print("🗺️ RUNNING ON FULL DATASET")

    # Define paths based on selected directory
    DEM_PATHS = {
        '2018': DEMS_DIR / "2018_0p5m_upper_rg_dem_larger_roi_harmonized.tif",
        '2023': DEMS_DIR / "GadValleyRG_50cmDEM_2023_harmonized.TIF",
        '2024': DEMS_DIR / "GadValleyRG_50cmDEM_2024_harmonized.TIF",
        '2025': DEMS_DIR / "GadValleyRG_50cmDEM_2025_harmonized.TIF"
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory set to: {OUTPUT_DIR.absolute()}\n")

    PIXEL_SIZE = 0.5  # 0.5m pixel size is constant
    
    # --- 2. Process DEMs in Parallel ---
    print("="*60)
    print("STARTING DEM CHARACTERIZATION (IN PARALLEL)")
    print("="*60)
    
    # Use n_jobs=-1 to use all available cores
    # Use n_jobs=1 to debug (run sequentially)
    n_jobs = -1 
    
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(process_single_dem)(year, path, PIXEL_SIZE, OUTPUT_DIR) 
        for year, path in DEM_PATHS.items()
    )

    # Combine the list of dictionaries from parallel processing
    # into a single, ordered dictionary
    results = {}
    for year in DEM_PATHS.keys(): # Iterate in order
        for res_dict in results_list:
            if year in res_dict:
                results[year] = res_dict[year]
                break

    # --- 3. Generate Comparison Plot ---
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOT")
    print("="*60)
    
    years = list(results.keys())
    if not years:
        print("No results to plot. Exiting.")
        return

    fig, axes = plt.subplots(
        nrows=2, 
        ncols=len(years), 
        figsize=(len(years) * 5, 10), 
        sharex=True, 
        sharey=True
    )

    for i, year in enumerate(years):
        if results[year] is None:
            axes[0, i].set_title(f"{year}\n(Failed to load)", fontweight='bold', color='red')
            axes[1, i].set_title(f"{year}\n(Failed to load)", fontweight='bold', color='red')
            continue

        # --- Top Row: Hillshades ---
        ax_hs = axes[0, i]
        ax_hs.imshow(results[year]['hillshade'], cmap='gray', vmin=0, vmax=255)
        ax_hs.set_title(f"{year} Hillshade", fontweight='bold')
        
        # --- Bottom Row: Roughness ---
        ax_rough = axes[1, i]
        rough_data = results[year]['roughness']
        
        vmax = np.nanpercentile(rough_data, 95)
        if vmax == 0: vmax = 1.0 # Handle flat/empty tiles
        
        im = ax_rough.imshow(
            rough_data, 
            cmap='inferno', 
            vmin=0, 
            vmax=vmax
        )
        
        ax_rough.set_title(f"{year} Roughness (StdDev)")
        
        cbar = fig.colorbar(im, ax=ax_rough, fraction=0.046, pad=0.04)
        cbar.set_label(f"Local StdDev (m)\nvmax={vmax:.2f}")

    # Set labels only on the outer plots
    for ax in axes[1, :]:
        ax.set_xlabel('Column (pixels)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Row (pixels)')
        
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "dem_characterization_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {plot_path.name}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"  Note: Could not show plot in IDE. {e}")
        
    plt.close()

    # --- 4. Generate Distribution Plot ---
    plot_roughness_distributions(results, OUTPUT_DIR)

    print("\n" + "="*60)
    print("CHARACTERIZATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()