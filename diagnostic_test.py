"""
Diagnostic script to identify where the rock glacier code is hanging
"""
import time
import numpy as np
import rasterio
from pathlib import Path

def test_file_access():
    """Test if files exist and are accessible"""
    print("=" * 60)
    print("FILE ACCESS TEST")
    print("=" * 60)
    
    files = {
        "DEM1": r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_LIDAR/2018_0p5m_upper_rg_dem_larger_roi.tif",
        "HS1": r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_2023_hs_large/2018_0p5m_hs_large.tif",
        "DEM2": r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2023_lidar/2023_0p5m_4_imcorr_upper_rg_larger_roi.tif",
        "HS2": r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_2023_hs_large/2023_0p5m_hs_large.tif",
        "Stable": r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\shapefiles\stable_2.shp"
    }
    
    for name, path in files.items():
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / (1024**2)
            print(f"✓ {name:8s}: EXISTS ({size_mb:.1f} MB) - {path}")
        else:
            print(f"✗ {name:8s}: NOT FOUND - {path}")
    
    return files

def test_raster_loading(file_path, max_wait_seconds=10):
    """Test loading a single raster with timeout"""
    print(f"\n{'=' * 60}")
    print(f"LOADING TEST: {Path(file_path).name}")
    print(f"{'=' * 60}")
    
    start_time = time.time()
    
    try:
        print(f"  Opening file...")
        with rasterio.open(file_path) as src:
            elapsed = time.time() - start_time
            print(f"  ✓ File opened in {elapsed:.2f}s")
            
            print(f"  File info:")
            print(f"    - Shape: {src.shape}")
            print(f"    - Dtype: {src.dtypes[0]}")
            print(f"    - CRS: {src.crs}")
            print(f"    - Bounds: {src.bounds}")
            
            # Test reading metadata only (fast)
            print(f"\n  Reading array...")
            read_start = time.time()
            data = src.read(1)
            read_elapsed = time.time() - read_start
            
            print(f"  ✓ Array read in {read_elapsed:.2f}s")
            print(f"  Array stats:")
            print(f"    - Min: {np.nanmin(data):.2f}")
            print(f"    - Max: {np.nanmax(data):.2f}")
            print(f"    - Mean: {np.nanmean(data):.2f}")
            print(f"    - NaN count: {np.sum(np.isnan(data))}")
            
            total_elapsed = time.time() - start_time
            print(f"\n  Total time: {total_elapsed:.2f}s")
            
            if total_elapsed > max_wait_seconds:
                print(f"  ⚠️  WARNING: Load time exceeded {max_wait_seconds}s threshold")
            
            return True, total_elapsed
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ ERROR after {elapsed:.2f}s: {e}")
        return False, elapsed

def test_ssim_performance():
    """Test if SSIM is available and check performance"""
    print(f"\n{'=' * 60}")
    print(f"SSIM PERFORMANCE TEST")
    print(f"{'=' * 60}")
    
    try:
        from skimage.metrics import structural_similarity as ssim
        print("✓ scikit-image is installed")
        
        # Test on small arrays
        print("\nTesting SSIM on 1000x1000 arrays...")
        img1 = np.random.rand(1000, 1000)
        img2 = np.random.rand(1000, 1000)
        
        start = time.time()
        score = ssim(img1, img2, data_range=1.0)
        elapsed = time.time() - start
        
        print(f"  ✓ SSIM completed in {elapsed:.2f}s")
        print(f"  Similarity score: {score:.3f}")
        
        # Estimate time for full resolution
        print("\nEstimating time for 5000x5000 arrays...")
        scale_factor = (5000/1000)**2
        estimated_time = elapsed * scale_factor
        print(f"  Estimated time: {estimated_time:.2f}s ({estimated_time/60:.1f} minutes)")
        
        if estimated_time > 60:
            print(f"  ⚠️  WARNING: Full-resolution SSIM would take >{estimated_time/60:.1f} minutes")
            print(f"     Recommendation: Use downsampling (factor=4 or higher)")
        
        return True
        
    except ImportError:
        print("✗ scikit-image is NOT installed")
        print("  Install with: pip install scikit-image")
        return False

def main():
    print("\n" + "=" * 60)
    print("ROCK GLACIER TRACKING - DIAGNOSTIC TEST")
    print("=" * 60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Test 1: File access
    files = test_file_access()
    
    # Test 2: SSIM availability
    test_ssim_performance()
    
    # Test 3: Load each file
    load_times = {}
    for name, path in files.items():
        if Path(path).exists() and name != "Stable":  # Skip shapefile
            success, elapsed = test_raster_loading(path, max_wait_seconds=30)
            load_times[name] = elapsed
            
            if not success:
                print(f"\n❌ CRITICAL: Failed to load {name}")
                break
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if load_times:
        print("\nLoad times:")
        for name, elapsed in load_times.items():
            status = "✓" if elapsed < 10 else "⚠️"
            print(f"  {status} {name:8s}: {elapsed:.2f}s")
        
        total_load_time = sum(load_times.values())
        print(f"\nTotal data loading time: {total_load_time:.2f}s")
        
        if total_load_time > 60:
            print(f"⚠️  WARNING: Data loading is slow (>{total_load_time/60:.1f} minutes)")
            print("   Possible causes:")
            print("   - Files are on a network drive")
            print("   - Files are very large")
            print("   - Disk I/O is slow")
            print("\n   Recommendations:")
            print("   - Copy files to local SSD")
            print("   - Use compressed formats (COG)")
            print("   - Work with downsampled data first")
    
    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()