"""
02_3d_synthesis_backwarp.py
===========================
Step 2: 3D Vector Synthesis & Vertical Flow Correction.
Automatically aligns DEMs to the displacement grid from Step 1.

This script implements the "warping" methodology to separate true vertical
surface change (melt/inflation) from apparent elevation changes caused by
horizontal displacement on sloped terrain.

Theory:
    - Simple DEM differencing (Eulerian) shows apparent elevation changes
      that mix real surface lowering with topographic effects
    - By "back-warping" the target DEM using the horizontal displacement field,
      we can isolate the true vertical component (Lagrangian)
    - This is the "gold standard" for rock glacier deformation analysis

Prerequisites:
    - Completed Step 1 (horizontal displacement from COSI-Corr)
    - Harmonized DEMs (elevation, not hillshades)

Usage:
    python 02_3d_synthesis_backwarp.py
"""

import numpy as np
import xdem
import geoutils as gu
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path

# ================= CONFIGURATION =================

# The Full Elevation Models (DEMs, NOT Hillshades)
DEM_REF_PATH = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems/2018_0p5m_upper_rg_dem_larger_roi_harmonized.tif"
DEM_TAR_PATH = r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems/GadValleyRG_50cmDEM_2023_harmonized.TIF"

# The Inputs from Step 3
# These are automatically found if you didn't change the output folder
DISP_DIR = Path("results_horizontal_cosicorr")
EW_PATH = DISP_DIR / "correlation_EW.tif"
NS_PATH = DISP_DIR / "correlation_NS.tif"

# Output
OUTPUT_DIR = Path("results_3d_final")

# Visualization settings
VMAX_DH = 5.0  # Maximum scale for elevation change plots (meters)
VMAX_MAG = 10.0  # Maximum scale for 3D magnitude plot (meters)

# ================= LOGIC =================

def run_synthesis():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("3D SYNTHESIS WITH XDEM")
    print("="*60)

    if not EW_PATH.exists():
        print(f"❌ Error: Run Step 3 first. Missing {EW_PATH}")
        print(f"   Expected path: {EW_PATH.absolute()}")
        return

    # 1. Load Displacement Maps (The 'Master' Grid)
    # We use this grid because COSI-Corr output might be cropped or coarser
    print("\n1. Loading Displacement Maps...")
    try:
        dx_map = gu.Raster(str(EW_PATH))
        dy_map = gu.Raster(str(NS_PATH))
        print(f"   ✅ Displacement grid shape: {dx_map.shape}")
        print(f"   Resolution: {dx_map.res[0]:.2f}m")
    except Exception as e:
        print(f"   ❌ Error loading displacement: {e}")
        return

    # 2. Load & Align DEMs
    print("\n2. Loading and Aligning DEMs...")
    try:
        # Load lazily
        dem_ref_raw = xdem.DEM(str(DEM_REF_PATH))
        dem_tar_raw = xdem.DEM(str(DEM_TAR_PATH))

        print(f"   Reference DEM: {dem_ref_raw.shape}")
        print(f"   Target DEM: {dem_tar_raw.shape}")

        # Align both DEMs to the Displacement Map's grid
        # This handles clipping, resampling, and resolution matching in one line
        print("   Reprojecting to displacement grid...")
        dem_ref = dem_ref_raw.reproject(dx_map, resampling='bilinear')
        dem_tar = dem_tar_raw.reproject(dx_map, resampling='bilinear')

        print(f"   ✅ Matched Grid Shape: {dem_ref.shape}")

    except FileNotFoundError as e:
        print(f"   ❌ DEM file not found: {e}")
        return
    except Exception as e:
        print(f"   ❌ Error aligning DEMs: {e}")
        return

    # 3. Prepare Data for Warping
    print("\n3. Preparing arrays for warping...")
    # xdem uses MaskedArrays. We fill NaNs for scipy processing.
    dem_tar_data = dem_tar.data.filled(np.nan).squeeze()
    dx_data = dx_map.data.filled(np.nan).squeeze()
    dy_data = dy_map.data.filled(np.nan).squeeze()

    # Statistics
    valid_disp = ~(np.isnan(dx_data) | np.isnan(dy_data))
    if not np.any(valid_disp):
        print("   ❌ No valid displacement data found!")
        return

    horiz_mag = np.sqrt(dx_data**2 + dy_data**2)
    print(f"   Horizontal displacement statistics:")
    print(f"     Mean: {np.nanmean(horiz_mag):.3f}m")
    print(f"     Max: {np.nanmax(horiz_mag):.3f}m")
    print(f"     Valid pixels: {np.sum(valid_disp):,}")

    # 4. Lagrangian Back-Warping
    print("\n4. Performing Back-Warping Correction...")
    print("   (This removes topographic effects from elevation change)")

    y_grid, x_grid = np.indices(dem_tar_data.shape)

    # Convert displacement (meters) to pixels
    res = dx_map.res
    dx_px = dx_data / res[0]
    dy_px = dy_data / -res[1]  # Negative because typical rasters count Y down

    # Map coordinates: Where did this pixel come from?
    # New = Old + Displacement
    coords = np.array([y_grid + dy_px, x_grid + dx_px])

    # Warp
    # We warp the TARGET (2023) back to the REFERENCE (2018) geometry
    dem_tar_warped_data = ndimage.map_coordinates(
        dem_tar_data,
        coords,
        order=1,  # Bilinear interpolation
        mode='constant',
        cval=np.nan
    )

    print(f"   ✅ Warped {np.sum(~np.isnan(dem_tar_warped_data)):,} pixels")

    # 5. Calculate Differences
    print("\n5. Calculating elevation changes...")

    # Re-wrap numpy arrays into xdem objects for easy saving/plotting
    dem_tar_warped = dem_ref.copy(new_data=dem_tar_warped_data)

    # Eulerian (Raw) Difference
    dh_raw = dem_tar - dem_ref

    # Lagrangian (Corrected) Difference (Melt/Inflation)
    dh_corrected = dem_tar_warped - dem_ref

    # 3D Magnitude
    mag_3d_data = np.sqrt(dx_data**2 + dy_data**2 + dh_raw.data.filled(np.nan).squeeze()**2)
    mag_3d = dem_ref.copy(new_data=mag_3d_data)

    # Statistics
    print(f"\n   Elevation Change Statistics:")
    print(f"   Raw (Eulerian):")
    print(f"     Mean: {np.nanmean(dh_raw.data):.3f}m")
    print(f"     Std: {np.nanstd(dh_raw.data):.3f}m")
    print(f"   Corrected (Lagrangian):")
    print(f"     Mean: {np.nanmean(dh_corrected.data):.3f}m")
    print(f"     Std: {np.nanstd(dh_corrected.data):.3f}m")
    print(f"   3D Magnitude:")
    print(f"     Mean: {np.nanmean(mag_3d_data):.3f}m")
    print(f"     Max: {np.nanmax(mag_3d_data):.3f}m")

    # 6. Save Results
    print(f"\n6. Saving results to {OUTPUT_DIR.absolute()}...")
    try:
        dh_raw.save(OUTPUT_DIR / "vert_change_raw.tif")
        dh_corrected.save(OUTPUT_DIR / "vert_change_corrected_melt.tif")
        mag_3d.save(OUTPUT_DIR / "magnitude_3d.tif")

        # Also save the horizontal components for reference
        dx_map.save(OUTPUT_DIR / "displacement_EW.tif")
        dy_map.save(OUTPUT_DIR / "displacement_NS.tif")

        print("   ✅ Saved:")
        print("      - vert_change_raw.tif (Eulerian)")
        print("      - vert_change_corrected_melt.tif (Lagrangian)")
        print("      - magnitude_3d.tif")
        print("      - displacement_EW.tif")
        print("      - displacement_NS.tif")
    except Exception as e:
        print(f"   ❌ Error saving results: {e}")
        return

    # 7. Quick Visualization using xdem
    print("\n7. Generating diagnostic plots...")
    try:
        fig, ax = plt.subplots(2, 3, figsize=(18, 12))

        # Top row: Raw differences and displacement
        dh_raw.plot(ax=ax[0, 0], cmap="RdBu_r", vmin=-VMAX_DH, vmax=VMAX_DH,
                    cbar_title="Elevation Change (m)")
        ax[0, 0].set_title("Raw Vertical Change (Eulerian)\n(Includes topographic effects)",
                          fontweight='bold')

        dh_corrected.plot(ax=ax[0, 1], cmap="RdBu_r", vmin=-VMAX_DH, vmax=VMAX_DH,
                         cbar_title="Elevation Change (m)")
        ax[0, 1].set_title("Corrected Vertical Change (Lagrangian)\n(True melt/inflation)",
                          fontweight='bold')

        # Difference between raw and corrected (the topographic correction applied)
        correction = dh_raw - dh_corrected
        correction.plot(ax=ax[0, 2], cmap="PuOr", vmin=-VMAX_DH/2, vmax=VMAX_DH/2,
                       cbar_title="Correction (m)")
        ax[0, 2].set_title("Topographic Correction Applied\n(Raw - Corrected)",
                          fontweight='bold')

        # Bottom row: Displacement components and magnitude
        dx_map.plot(ax=ax[1, 0], cmap="RdBu_r", vmin=-VMAX_MAG/2, vmax=VMAX_MAG/2,
                   cbar_title="Displacement (m)")
        ax[1, 0].set_title("East-West Displacement", fontweight='bold')

        dy_map.plot(ax=ax[1, 1], cmap="RdBu_r", vmin=-VMAX_MAG/2, vmax=VMAX_MAG/2,
                   cbar_title="Displacement (m)")
        ax[1, 1].set_title("North-South Displacement", fontweight='bold')

        mag_3d.plot(ax=ax[1, 2], cmap="viridis", vmin=0, vmax=VMAX_MAG,
                   cbar_title="3D Magnitude (m)")
        ax[1, 2].set_title("3D Displacement Magnitude\n√(dx² + dy² + dz²)",
                          fontweight='bold')

        plt.tight_layout()
        plot_path = OUTPUT_DIR / "3d_synthesis_diagnostic.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {plot_path.name}")
        plt.close()

        # Histogram comparison
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Elevation change histogram
        ax[0].hist(dh_raw.data.compressed(), bins=50, alpha=0.6, label='Raw (Eulerian)',
                  density=True, color='blue')
        ax[0].hist(dh_corrected.data.compressed(), bins=50, alpha=0.6,
                  label='Corrected (Lagrangian)', density=True, color='red')
        ax[0].axvline(0, color='black', linestyle='--', linewidth=1)
        ax[0].set_xlabel('Elevation Change (m)')
        ax[0].set_ylabel('Density')
        ax[0].set_title('Distribution of Vertical Displacement')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # 3D magnitude histogram
        ax[1].hist(horiz_mag[valid_disp].flatten(), bins=50, alpha=0.6,
                  label='Horizontal', color='green', density=True)
        ax[1].hist(mag_3d_data[~np.isnan(mag_3d_data)], bins=50, alpha=0.6,
                  label='3D Total', color='orange', density=True)
        ax[1].set_xlabel('Displacement Magnitude (m)')
        ax[1].set_ylabel('Density')
        ax[1].set_title('Distribution of Displacement Magnitude')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        hist_path = OUTPUT_DIR / "displacement_histograms.png"
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {hist_path.name}")
        plt.close()

    except Exception as e:
        print(f"   ⚠️ Warning: Could not create plots: {e}")

    print("\n" + "="*60)
    print("✅ 3D SYNTHESIS COMPLETE")
    print("="*60)
    print(f"\nOutputs in: {OUTPUT_DIR.absolute()}")
    print("\nKey Results:")
    print(f"  - vert_change_corrected_melt.tif: True vertical surface change")
    print(f"  - magnitude_3d.tif: Total 3D displacement magnitude")
    print("\nInterpretation:")
    print("  - Red in 'Corrected' map = Surface lowering (melt/deflation)")
    print("  - Blue in 'Corrected' map = Surface thickening (inflation)")
    print("  - Compare 'Raw' vs 'Corrected' to see topographic correction effect")

if __name__ == "__main__":
    run_synthesis()
