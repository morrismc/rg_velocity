"""
00_generate_test_patches.py
===========================
Utility to crop harmonized DEMs to a specific test patch shapefile.
Uses rasterio for robust clipping (more reliable than geoutils for this task).

This script should be run ONCE after preprocessing to create smaller
test datasets for rapid iteration and parameter tuning.

Usage:
    python 00_generate_test_patches.py
"""

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
import numpy as np

# ================= CONFIGURATION =================

# 1. Input Directory (Where your harmonized DEMs live)
INPUT_DIR = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems")

# 2. The Patch Shapefile
PATCH_SHAPEFILE = Path(r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\shapefiles\small_patch.shp")

# 3. Output Directory for Patches
OUTPUT_DIR = INPUT_DIR / "patches"

# 4. List of files to process (The Harmonized filenames)
DEM_FILES = [
    "2018_0p5m_upper_rg_dem_larger_roi_harmonized.tif",
    "GadValleyRG_50cmDEM_2023_harmonized.TIF",
    "GadValleyRG_50cmDEM_2024_harmonized.TIF",
    "GadValleyRG_50cmDEM_2025_harmonized.TIF"
]

# Also process hillshades if they exist
HILLSHADE_FILES = [
    "2018_0p5m_hs_harmonized.tif",
    "2023_0p5m_hs_harmonized.tif",
    "2024_0p5m_hs_harmonized.tif",
    "2025_0p5m_hs_harmonized.tif"
]

# ================= MAIN LOGIC =================

def crop_raster_to_shapefile(input_path, shapefile_gdf, output_path):
    """
    Crop a raster to shapefile bounds using rasterio.

    Parameters:
    -----------
    input_path : Path
        Path to input raster
    shapefile_gdf : GeoDataFrame
        Loaded shapefile as geopandas GeoDataFrame
    output_path : Path
        Path to save cropped raster

    Returns:
    --------
    bool : True if successful, False otherwise
    """
    try:
        with rasterio.open(input_path) as src:
            # Get raster CRS
            raster_crs = src.crs

            # Reproject shapefile to match raster CRS if needed
            if shapefile_gdf.crs != raster_crs:
                print(f"    Reprojecting shapefile from {shapefile_gdf.crs} to {raster_crs}")
                shapefile_gdf = shapefile_gdf.to_crs(raster_crs)

            # Get geometries
            shapes = shapefile_gdf.geometry.values

            # Crop (mask) the raster
            # crop=True ensures the output raster is cropped to the bounds
            out_image, out_transform = mask(src, shapes, crop=True, all_touched=False)

            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw"  # Add compression to save space
            })

            # Check if we got valid data
            if out_image.shape[1] == 0 or out_image.shape[2] == 0:
                print(f"    ❌ Error: Crop resulted in empty raster (no overlap?)")
                return False

            # Check for valid pixels
            if np.all(np.isnan(out_image)) or np.all(out_image == src.nodata):
                print(f"    ⚠️ Warning: Cropped raster contains only NoData")

            # Save
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f"    ✅ Cropped: {out_image.shape[1]}×{out_image.shape[2]} pixels")
            return True

    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_patches():
    print("="*60)
    print("GENERATING TEST PATCHES (using rasterio)")
    print("="*60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Output Directory: {OUTPUT_DIR}")

    # Load shapefile once
    if not PATCH_SHAPEFILE.exists():
        print(f"❌ Error: Patch shapefile not found at {PATCH_SHAPEFILE}")
        print("\nPlease create a polygon shapefile defining your test area.")
        print("You can use QGIS or ArcGIS to draw a small rectangle over your area of interest.")
        return

    print(f"\nLoading crop geometry: {PATCH_SHAPEFILE.name}")
    try:
        shapefile_gdf = gpd.read_file(PATCH_SHAPEFILE)
        print(f"  CRS: {shapefile_gdf.crs}")
        print(f"  Bounds: {shapefile_gdf.total_bounds}")
        print(f"  Features: {len(shapefile_gdf)}")

        if len(shapefile_gdf) == 0:
            print("  ❌ Error: Shapefile is empty!")
            return

    except Exception as e:
        print(f"  ❌ Error loading shapefile: {e}")
        return

    # Process DEMs
    print("\n--- Processing DEMs ---")
    success_count = 0
    for filename in DEM_FILES:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename

        print(f"\n{filename}")

        if not input_path.exists():
            print(f"  ⚠️ File not found: {input_path}")
            continue

        # Get file size for reporting
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        print(f"  Input size: {file_size_mb:.1f} MB")

        if crop_raster_to_shapefile(input_path, shapefile_gdf, output_path):
            # Report output size
            if output_path.exists():
                out_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  Output size: {out_size_mb:.1f} MB")
            success_count += 1

    # Process Hillshades
    print("\n--- Processing Hillshades ---")
    for filename in HILLSHADE_FILES:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename

        print(f"\n{filename}")

        if not input_path.exists():
            print(f"  ⚠️ File not found (skipping)")
            continue

        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        print(f"  Input size: {file_size_mb:.1f} MB")

        if crop_raster_to_shapefile(input_path, shapefile_gdf, output_path):
            if output_path.exists():
                out_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  Output size: {out_size_mb:.1f} MB")
            success_count += 1

    print("\n" + "="*60)
    print("PATCH GENERATION COMPLETE")
    print("="*60)
    print(f"Successfully processed: {success_count} files")
    print(f"Patches saved to: {OUTPUT_DIR.absolute()}")

    if success_count > 0:
        print("\n✅ Next steps:")
        print("   1. Set USE_TEST_PATCH = True in analysis scripts")
        print("   2. Run: python 03_horizontal_displacement_xdem.py")
    else:
        print("\n⚠️ No files were successfully processed. Check:")
        print("   1. Input files exist in preprocessed_dems/")
        print("   2. Shapefile overlaps with raster extent")
        print("   3. Shapefile CRS is compatible")

if __name__ == "__main__":
    generate_patches()
