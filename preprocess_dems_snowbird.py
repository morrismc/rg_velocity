"""
Standalone DEM Preprocessor for Snowbird Rock Glacier
======================================================

This script validates and harmonizes your 2018, 2023, 2024, and 2025 DEMs
before running displacement analysis.

Run this FIRST, then use the harmonized DEMs in your displacement analysis.

Usage:
------
    python preprocess_dems_snowbird.py

Output:
-------
    preprocessed_dems/
        ├── 2018_0p5m_upper_rg_dem_larger_roi_harmonized.tif
        ├── GadValleyRG_50cmDEM_2023_harmonized.tif
        ├── GadValleyRG_50cmDEM_2024_harmonized.tif
        ├── GadValleyRG_50cmDEM_2025_harmonized.tif
        ├── dem_validation_report.txt
        └── dem_coverage.png
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import OrderedDict


def validate_and_harmonize_dems():
    """Main preprocessing function."""
    
    # ==================== USER CONFIGURATION ====================
   # Define your DEM files
    dem_files = [
        {
            'name': '2018',
            'path': r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_LIDAR/2018_0p5m_upper_rg_dem_larger_roi.tif",
            'date': '2018-09-01'
        },
        {
            'name': '2023',
            'path': r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Corrected_DEMs/GadValleyRG_50cmDEM_2023.TIF",
            'date': '2023-09-01'
        },
        {
            'name': '2024',
            'path': r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Corrected_DEMs/GadValleyRG_50cmDEM_2024.TIF",
            'date': '2024-09-01'
        },
        {
            'name': '2025',
            'path': r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Corrected_DEMs/GadValleyRG_50cmDEM_2025.TIF",
            'date': '2025-09-01'
        }
    ]
    
    # Stable area shapefile
    stable_shapefile = r"M:\My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/shapefiles/stable_2.shp"
    
    # Output directory
    output_dir = Path("preprocessed_dems")
    output_dir.mkdir(exist_ok=True)
    
    # Reference DEM (all others will be matched to this)
    REFERENCE_NAME = '2018'
    
    # Resampling method
    RESAMPLING_METHOD = Resampling.bilinear
    
    # ============================================================
    
    print("="*70)
    print("DEM PREPROCESSING FOR SNOWBIRD ROCK GLACIER")
    print("="*70)
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Step 1: Load and validate DEMs
    print("\n" + "="*70)
    print("STEP 1: LOADING AND VALIDATING DEMS")
    print("="*70)
    
    dems = OrderedDict()
    
    for dem_info in dem_files:
        name = dem_info['name']
        path = Path(dem_info['path'])
        
        print(f"\n{name}:")
        print(f"  File: {path.name}")
        
        if not path.exists():
            print(f"  ❌ ERROR: File not found!")
            print(f"  Path checked: {path.absolute()}")
            return False
        
        try:
            with rasterio.open(path) as src:
                metadata = {
                    'name': name,
                    'path': path,
                    'shape': (src.height, src.width),
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'resolution': (src.transform[0], abs(src.transform[4])),
                    'nodata': src.nodata
                }
                
                # Read and validate data
                data = src.read(1).astype(np.float64)
                nodata_mask = np.isnan(data) | np.isinf(data) | (data < -1e10) | (data > 10000)
                if src.nodata is not None and np.isfinite(src.nodata):
                    nodata_mask |= (data == src.nodata)
                
                data[nodata_mask] = np.nan
                valid_data = data[~np.isnan(data)]
                
                metadata['valid_pixels'] = len(valid_data)
                metadata['nodata_pixels'] = np.sum(nodata_mask)
                metadata['nodata_percent'] = 100 * np.sum(nodata_mask) / data.size
                
                if len(valid_data) > 0:
                    metadata['min'] = float(np.min(valid_data))
                    metadata['max'] = float(np.max(valid_data))
                    metadata['mean'] = float(np.mean(valid_data))
                else:
                    metadata['min'] = np.nan
                    metadata['max'] = np.nan
                    metadata['mean'] = np.nan
                
                dems[name] = metadata
                
                print(f"  ✓ Shape: {metadata['shape']}")
                print(f"  ✓ Resolution: {metadata['resolution'][0]:.3f} m")
                print(f"  ✓ Valid range: [{metadata['min']:.1f}, {metadata['max']:.1f}] m")
                print(f"  ✓ Valid pixels: {metadata['valid_pixels']:,} ({100-metadata['nodata_percent']:.1f}%)")
                
        except Exception as e:
            print(f"  ❌ ERROR loading DEM: {e}")
            return False
    
    # Step 2: Check spatial consistency
    print("\n" + "="*70)
    print("STEP 2: CHECKING SPATIAL CONSISTENCY")
    print("="*70)
    
    reference = dems[REFERENCE_NAME]
    print(f"\nUsing '{REFERENCE_NAME}' as reference")
    
    needs_harmonization = False
    issues = []
    
    for name, dem in dems.items():
        if name == REFERENCE_NAME:
            continue
        
        print(f"\n{name} vs {REFERENCE_NAME}:")
        
        # Check CRS
        if dem['crs'] != reference['crs']:
            print(f"  ❌ CRS mismatch")
            issues.append(f"{name}: CRS mismatch")
            needs_harmonization = True
        else:
            print(f"  ✓ CRS matches")
        
        # Check resolution
        res_diff = abs(dem['resolution'][0] - reference['resolution'][0])
        if res_diff > 0.001:
            print(f"  ❌ Resolution mismatch ({res_diff:.4f} m difference)")
            issues.append(f"{name}: Resolution mismatch")
            needs_harmonization = True
        else:
            print(f"  ✓ Resolution matches")
        
        # Check shape
        if dem['shape'] != reference['shape']:
            print(f"  ❌ Shape mismatch: {dem['shape']} vs {reference['shape']}")
            issues.append(f"{name}: Shape mismatch")
            needs_harmonization = True
        else:
            print(f"  ✓ Shape matches")
        
        # Check transform
        if dem['transform'] != reference['transform']:
            print(f"  ❌ Transform mismatch (pixel grids don't align)")
            issues.append(f"{name}: Transform mismatch")
            needs_harmonization = True
        else:
            print(f"  ✓ Transform matches")
    
    # Step 3: Check stable area coverage
    print("\n" + "="*70)
    print("STEP 3: CHECKING STABLE AREA COVERAGE")
    print("="*70)
    
    stable_path = Path(stable_shapefile)
    if stable_path.exists():
        try:
            stable_gdf = gpd.read_file(stable_path)
            print(f"\n✓ Loaded stable areas: {len(stable_gdf)} polygon(s)")
            
            for name, dem in dems.items():
                if stable_gdf.crs != dem['crs']:
                    stable_reproj = stable_gdf.to_crs(dem['crs'])
                else:
                    stable_reproj = stable_gdf
                
                # Rough coverage estimate
                stable_area = stable_reproj.geometry.area.sum()
                pixel_area = dem['resolution'][0] * dem['resolution'][1]
                stable_pixels = int(stable_area / pixel_area)
                total_pixels = dem['shape'][0] * dem['shape'][1]
                coverage_pct = 100 * stable_pixels / total_pixels
                
                print(f"  {name}: ~{stable_pixels:,} stable pixels ({coverage_pct:.1f}%)")
                
                if coverage_pct < 0.1:
                    print(f"    ⚠️ WARNING: Very low stable area coverage!")
        
        except Exception as e:
            print(f"⚠️ Could not validate stable areas: {e}")
    else:
        print(f"⚠️ Stable area shapefile not found: {stable_path}")
    
    # Step 4: Harmonize if needed
    if not needs_harmonization:
        print("\n" + "="*70)
        print("✅ ALL DEMS ARE ALREADY CONSISTENT!")
        print("="*70)
        print("No harmonization needed. You can use the original DEMs.")
        
        # Still copy to output for consistency
        print("\nCopying DEMs to output directory for consistency...")
        for name, dem in dems.items():
            src_path = dem['path']
            dst_path = output_dir / f"{src_path.stem}_harmonized.tif"
            
            with rasterio.open(src_path) as src:
                data = src.read()
                profile = src.profile.copy()
            
            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(data)
            
            print(f"  ✓ {name}: {dst_path.name}")
        
    else:
        print("\n" + "="*70)
        print("⚠️ SPATIAL INCONSISTENCIES DETECTED")
        print("="*70)
        print(f"Issues found: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\n" + "="*70)
        print("HARMONIZING DEMS TO COMMON GRID")
        print("="*70)
        print(f"Reference: {REFERENCE_NAME}")
        print(f"Resampling: {RESAMPLING_METHOD.name}\n")
        
        for name, dem in dems.items():
            src_path = dem['path']
            # --- UPDATED: Use stem to get original filename ---
            dst_path = output_dir / f"{src_path.stem}_harmonized.tif"
            
            print(f"{name}:")
            
            if name == REFERENCE_NAME:
                # Just copy reference
                print(f"  (reference - copying)")
                with rasterio.open(src_path) as src:
                    data = src.read()
                    profile = src.profile.copy()
                
                with rasterio.open(dst_path, 'w', **profile) as dst:
                    dst.write(data)
                
                print(f"  ✓ Saved: {dst_path.name}")
                continue
            
            try:
                with rasterio.open(src_path) as src:
                    # Read source
                    src_data = src.read(1).astype(np.float64)
                    src_nodata = src.nodata
                    if src_nodata is None:
                        src_nodata = -9999.0 # Assign a nodata value
                    
                    # Clean nodata
                    src_data[np.isnan(src_data)] = src_nodata
                    src_data[np.isinf(src_data)] = src_nodata
                    src_data[src_data < -1e10] = src_nodata
                    src_data[src_data > 10000] = src_nodata
                    
                    # Create output array
                    dst_data = np.full(
                        reference['shape'],
                        src_nodata,
                        dtype=np.float64
                    )
                    
                    # Reproject
                    reproject(
                        source=src_data,
                        destination=dst_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=src_nodata,
                        dst_transform=reference['transform'],
                        dst_crs=reference['crs'],
                        dst_nodata=src_nodata,
                        resampling=RESAMPLING_METHOD
                    )
                    
                    # Replace nodata with NaN
                    dst_data[dst_data == src_nodata] = np.nan
                    
                    # Statistics
                    valid_mask = np.isfinite(dst_data)
                    n_valid = np.sum(valid_mask)
                    pct_valid = 100 * n_valid / dst_data.size
                    
                    print(f"  ✓ Reprojected to {reference['shape']}")
                    print(f"  ✓ Valid pixels: {n_valid:,} ({pct_valid:.1f}%)")
                    
                    if n_valid > 0:
                        valid_data = dst_data[valid_mask]
                        print(f"  ✓ Range: [{np.min(valid_data):.1f}, {np.max(valid_data):.1f}] m")
                    
                    # Save
                    profile = {
                        'driver': 'GTiff',
                        'height': reference['shape'][0],
                        'width': reference['shape'][1],
                        'count': 1,
                        'dtype': 'float32',
                        'crs': reference['crs'],
                        'transform': reference['transform'],
                        'nodata': np.nan,
                        'compress': 'lzw'
                    }
                    
                    with rasterio.open(dst_path, 'w', **profile) as dst:
                        dst.write(dst_data.astype(np.float32), 1)
                    
                    print(f"  ✓ Saved: {dst_path.name}")
                    
            except Exception as e:
                print(f"  ❌ ERROR harmonizing: {e}")
                return False
    
    # Step 5: Create visualization
    print("\n" + "="*70)
    print("CREATING COVERAGE VISUALIZATION")
    print("="*70)
    
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(dems)))
        
        # Calculate overall extent
        all_bounds = [dem['bounds'] for dem in dems.values()]
        min_x = min(b.left for b in all_bounds)
        max_x = max(b.right for b in all_bounds)
        min_y = min(b.bottom for b in all_bounds)
        max_y = max(b.top for b in all_bounds)
        
        # Add 5% margin
        margin_x = (max_x - min_x) * 0.05
        margin_y = (max_y - min_y) * 0.05
        
        for idx, (name, dem) in enumerate(dems.items()):
            bounds = dem['bounds']
            rect = Rectangle(
                (bounds.left, bounds.bottom),
                bounds.right - bounds.left,
                bounds.top - bounds.bottom,
                fill=False,
                edgecolor=colors[idx],
                linewidth=3,
                label=name
            )
            ax.add_patch(rect)
            
            # Label
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            ax.text(center_x, center_y, name,
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Set explicit axis limits to prevent matplotlib issues
        ax.set_xlim(min_x - margin_x, max_x + margin_x)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)
        
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_title('DEM Spatial Coverage - Snowbird Rock Glacier',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plot_path = output_dir / 'dem_coverage.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {plot_path.name}")
        
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
        print("   (This doesn't affect harmonization - DEMs are still valid)")
    
    # Step 6: Generate report
    print("\n" + "="*70)
    print("GENERATING VALIDATION REPORT")
    print("="*70)
    
    report_path = output_dir / 'dem_validation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DEM VALIDATION REPORT - SNOWBIRD ROCK GLACIER\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference DEM: {REFERENCE_NAME}\n")
        f.write(f"Resampling method: {RESAMPLING_METHOD.name}\n\n")
        
        f.write("DEMS PROCESSED:\n")
        f.write("-"*70 + "\n")
        for name, dem in dems.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Original file: {dem['path'].name}\n")
            f.write(f"  Shape: {dem['shape']}\n")
            f.write(f"  Resolution: {dem['resolution'][0]:.3f} m\n")
            f.write(f"  CRS: {dem['crs']}\n")
            f.write(f"  Valid range: [{dem['min']:.1f}, {dem['max']:.1f}] m\n")
            f.write(f"  Valid pixels: {dem['valid_pixels']:,} ({100-dem['nodata_percent']:.1f}%)\n")
        
        if needs_harmonization:
            f.write("\n\nHARMONIZATION PERFORMED:\n")
            f.write("-"*70 + "\n")
            f.write(f"Reason: Spatial inconsistencies detected\n")
            f.write(f"Issues resolved: {len(issues)}\n")
            for issue in issues:
                f.write(f"  - {issue}\n")
            f.write(f"\nAll DEMs reprojected to match {REFERENCE_NAME} grid\n")
        else:
            f.write("\n\nHARMONIZATION STATUS:\n")
            f.write("-"*70 + "\n")
            f.write("Not required - all DEMs already spatially consistent\n")
        
        f.write("\n\nOUTPUT FILES:\n")
        f.write("-"*70 + "\n")
        f.write(f"Location: {output_dir.absolute()}\n\n")
        for name, dem in dems.items():
            harmonized_name = f"{dem['path'].stem}_harmonized.tif"
            f.write(f"  {harmonized_name}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Saved: {report_path.name}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE")
    print("="*70)
    print(f"\n✓ Harmonized DEMs saved to: {output_dir.absolute()}")
    print(f"✓ All DEMs now have consistent shape: {reference['shape']}")
    print(f"✓ All DEMs aligned to common grid")
    print(f"\nGenerated files:")
    for name, dem in dems.items():
        harmonized_name = f"{dem['path'].stem}_harmonized.tif"
        print(f"  ✓ {harmonized_name}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review: dem_validation_report.txt")
    if (output_dir / 'dem_coverage.png').exists():
        print("2. View: dem_coverage.png")
    print("\n3. Update your displacement analysis scripts with these harmonized DEMs:")
    print("\n   (Example for vertical_displacement_analysis_multi-year.py)")
    print("   dem_epochs = [")
    for dem_info in dem_files:
        name = dem_info['name']
        date = dem_info['date']
        harmonized_name = f"{Path(dem_info['path']).stem}_harmonized.tif"
        print(f"       ('{name}', r\"{output_dir / harmonized_name}\", '{date}'),")
    print("   ]")
    print("\n4. Run your displacement analysis scripts.")
    
    return True


if __name__ == "__main__":
    try:
        success = validate_and_harmonize_dems()
        if success:
            print("\n🎉 Success! Ready for displacement analysis.")
        else:
            print("\n❌ Preprocessing failed. Check errors above.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()