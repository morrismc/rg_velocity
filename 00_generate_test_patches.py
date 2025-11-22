"""
00_generate_test_patches.py
===========================
Utility to crop harmonized DEMs to a specific test patch shapefile.
Uses xdem/geoutils for efficient processing.

This script should be run ONCE after preprocessing to create smaller
test datasets for rapid iteration and parameter tuning.

Usage:
    python 00_generate_test_patches.py
"""

import geoutils as gu
from pathlib import Path
import matplotlib.pyplot as plt

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

def generate_patches():
    print("="*60)
    print("GENERATING TEST PATCHES")
    print("="*60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output Directory: {OUTPUT_DIR}")

    if not PATCH_SHAPEFILE.exists():
        print(f"❌ Error: Patch shapefile not found at {PATCH_SHAPEFILE}")
        return

    # Load the crop geometry once
    print(f"Loading crop geometry: {PATCH_SHAPEFILE.name}")
    crop_geom = gu.Vector(str(PATCH_SHAPEFILE))

    # Display crop geometry info
    print(f"  Bounds: {crop_geom.bounds}")
    print(f"  CRS: {crop_geom.crs}")

    # Process DEMs
    print("\n--- Processing DEMs ---")
    for filename in DEM_FILES:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename  # Keep same filename, different folder

        print(f"\n{filename}")

        if not input_path.exists():
            print(f"  ⚠️ File not found: {input_path}")
            continue

        try:
            # Load Raster
            r = gu.Raster(str(input_path))

            print(f"  Original size: {r.shape}")

            # Crop
            print("  ✂️ Cropping...")
            r_cropped = r.crop(crop_geom)

            print(f"  Cropped size: {r_cropped.shape}")

            # Save
            r_cropped.save(str(output_path))
            print(f"  ✅ Saved: patches/{output_path.name}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Process Hillshades
    print("\n--- Processing Hillshades ---")
    for filename in HILLSHADE_FILES:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename

        print(f"\n{filename}")

        if not input_path.exists():
            print(f"  ⚠️ File not found (skipping): {input_path}")
            continue

        try:
            # Load Raster
            r = gu.Raster(str(input_path))

            print(f"  Original size: {r.shape}")

            # Crop
            print("  ✂️ Cropping...")
            r_cropped = r.crop(crop_geom)

            print(f"  Cropped size: {r_cropped.shape}")

            # Save
            r_cropped.save(str(output_path))
            print(f"  ✅ Saved: patches/{output_path.name}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "="*60)
    print("PATCH GENERATION COMPLETE")
    print(f"All patches saved to: {OUTPUT_DIR.absolute()}")
    print("="*60)

if __name__ == "__main__":
    generate_patches()
