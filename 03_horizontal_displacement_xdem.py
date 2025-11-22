"""
03_horizontal_displacement_xdem.py
==================================
Step 3: Horizontal Displacement using COSI-Corr.
Uses xdem/geoutils for efficient pre-processing (clipping).

This script wraps the COSI-Corr frequency correlation tool to calculate
horizontal (X, Y) displacement vectors between two epochs.

Prerequisites:
    - COSI-Corr repository cloned to Tools directory
    - Harmonized hillshades (from preprocessing step)
    - Test patches generated (from 00_generate_test_patches.py) if using test mode
    - 'rock_glacier_env' conda environment activated

Usage:
    python 03_horizontal_displacement_xdem.py
"""

import sys
import subprocess
from pathlib import Path
import geoutils as gu  # geoutils is the base library for xdem
import geopandas as gpd

# ================= CONFIGURATION =================

# Input Hillshades (Harmonized from preprocessing step)
# NOTE: Use HILLSHADES not DEMs for best feature tracking
IMG_REF = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_LIDAR/2018_0p5m_hs_harmonized.tif"
IMG_TARGET = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2023_lidar/2023_0p5m_hs_harmonized.tif"

# Test Patch Settings
USE_TEST_PATCH = True
PATCH_SHAPEFILE = r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\shapefiles\small_patch.shp"

# Alternative: If you already generated patches with 00_generate_test_patches.py,
# you can point directly to them instead:
# USE_PREGENERATED_PATCHES = True
# IMG_REF = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems/patches/2018_0p5m_hs_harmonized.tif"
# IMG_TARGET = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems/patches/2023_0p5m_hs_harmonized.tif"

# Output & Tools
OUTPUT_DIR = Path("results_horizontal_cosicorr")
COSICORR_SCRIPT = r"M:/My Drive/Rock Glaciers/Tools/Geospatial-COSICorr3D/geoCosiCorr3D/scripts/correlation.py"

# COSI-Corr Parameters (Tuning)
# Window Size: Area to search for features (64px * 0.5m = 32m window)
# Step Size: Resolution of output grid (4px * 0.5m = 2m resolution)
WINDOW_SIZE = 64
STEP_SIZE = 4

# Advanced parameters (uncomment to customize)
# CORR_MIN = 0.9  # Minimum correlation threshold (0-1)
# ROBUSTNESS = 4  # Number of robustness iterations

# ================= LOGIC =================

def prepare_inputs(ref_path, tar_path, output_dir):
    """Prepares input files, cropping them if a patch is defined."""
    if not USE_TEST_PATCH:
        return ref_path, tar_path

    print(f"  ✂️ Clipping inputs to test patch...")
    if not Path(PATCH_SHAPEFILE).exists():
        raise FileNotFoundError(f"Patch shapefile not found: {PATCH_SHAPEFILE}")

    # Load inputs using geoutils (lazy loading)
    ref = gu.Raster(ref_path)
    tar = gu.Raster(tar_path)
    crop_geom = gu.Vector(PATCH_SHAPEFILE)

    # Crop both (inplace=False to return new objects)
    # This automatically handles the bounding box of your shapefile
    ref_crop = ref.crop(crop_geom)
    tar_crop = tar.crop(crop_geom)

    # Save temporary clips for COSI-Corr to read
    ref_out = output_dir / "temp_ref_clip.tif"
    tar_out = output_dir / "temp_tar_clip.tif"

    ref_crop.save(ref_out)
    tar_crop.save(tar_out)

    print(f"     Created temps: {ref_out.name}, {tar_out.name}")
    return str(ref_out), str(tar_out)

def run_process():
    # Ensure output directory exists first
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- COSI-CORR with XDEM (Test Mode: {USE_TEST_PATCH}) ---")

    try:
        # 1. Prepare Files (Clip if needed)
        ref_input, tar_input = prepare_inputs(IMG_REF, IMG_TARGET, OUTPUT_DIR)

        # 2. Build Command
        # We use sys.executable to force using the CURRENT active python environment
        cmd = [
            sys.executable,
            COSICORR_SCRIPT,
            "correlate",
            ref_input,
            tar_input,
            "--window_size", str(WINDOW_SIZE),
            "--step", str(STEP_SIZE),
            "--output_dir", str(OUTPUT_DIR)
        ]

        # Add optional parameters if defined
        # if 'CORR_MIN' in globals():
        #     cmd.extend(["--corr_min", str(CORR_MIN)])
        # if 'ROBUSTNESS' in globals():
        #     cmd.extend(["--robustness", str(ROBUSTNESS)])

        # 3. Run
        print(f"\n🚀 Launching COSI-Corr...")
        print(f"   Window: {WINDOW_SIZE}px, Step: {STEP_SIZE}px")
        print(f"   Command: {' '.join(cmd)}")

        # subprocess.run will wait for the external script to finish
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Print COSI-Corr output
        print("\n--- COSI-Corr Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- COSI-Corr Warnings ---")
            print(result.stderr)

        print(f"\n✅ Complete. Results in: {OUTPUT_DIR.absolute()}")
        print("\n--- Output Files ---")
        for output_file in OUTPUT_DIR.glob("*.tif"):
            print(f"   {output_file.name}")

        print("\n--- Next Step ---")
        print("   Run: python 04_3d_synthesis_xdem.py")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ COSI-Corr Execution Error")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("\nStdout:")
            print(e.stdout)
        if e.stderr:
            print("\nStderr:")
            print(e.stderr)
    except FileNotFoundError as e:
        print(f"\n❌ File Not Found: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that COSICORR_SCRIPT path is correct")
        print("  2. Verify input hillshade files exist")
        print("  3. Ensure patch shapefile exists (if USE_TEST_PATCH=True)")
    except Exception as e:
        print(f"\n❌ Script Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_process()
