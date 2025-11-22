"""
Minimal test script - Run this to find where code hangs
This script tests each component individually with clear progress markers
"""

import time
import numpy as np
from pathlib import Path

def print_section(title):
    """Print a visible section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)
    time.sleep(0.1)  # Ensure it prints before any blocking operation

# =============================================================================
# TEST 1: File Access
# =============================================================================
print_section("TEST 1: Checking if files exist...")

dem1_file = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_LIDAR/2018_0p5m_upper_rg_dem_larger_roi.tif")
print(f"DEM1 exists: {dem1_file.exists()}")

if not dem1_file.exists():
    print("\n❌ ERROR: Cannot find files on M: drive")
    print("   Is Google Drive mounted/synced?")
    exit(1)

# =============================================================================
# TEST 2: Import Rasterio
# =============================================================================
print_section("TEST 2: Importing rasterio...")
try:
    import rasterio
    print("✓ rasterio imported successfully")
except ImportError as e:
    print(f"✗ rasterio import failed: {e}")
    exit(1)

# =============================================================================
# TEST 3: Open File (This might be where it hangs!)
# =============================================================================
print_section("TEST 3: Opening raster file...")
print("⏳ This is where network drives often hang...")
print(f"Opening: {dem1_file.name}")

start = time.time()
try:
    with rasterio.open(dem1_file) as src:
        elapsed = time.time() - start
        print(f"✓ File opened in {elapsed:.2f} seconds")
        print(f"  Shape: {src.shape}")
        print(f"  CRS: {src.crs}")
        
        if elapsed > 30:
            print(f"\n⚠️  WARNING: Opening took {elapsed:.1f}s (very slow!)")
            print("   This suggests network drive issues")
except Exception as e:
    elapsed = time.time() - start
    print(f"✗ Failed after {elapsed:.2f}s: {e}")
    exit(1)

# =============================================================================
# TEST 4: Read Data (This might also hang!)
# =============================================================================
print_section("TEST 4: Reading raster data...")
print("⏳ Loading array into memory...")

start = time.time()
try:
    with rasterio.open(dem1_file) as src:
        data = src.read(1)
        elapsed = time.time() - start
        print(f"✓ Data loaded in {elapsed:.2f} seconds")
        print(f"  Array shape: {data.shape}")
        print(f"  Array size: {data.nbytes / (1024**2):.1f} MB")
        print(f"  Data range: [{np.nanmin(data):.1f}, {np.nanmax(data):.1f}]")
        
        if elapsed > 60:
            print(f"\n⚠️  WARNING: Loading took {elapsed:.1f}s (very slow!)")
            print("   Recommendation: Copy files to local SSD")
except Exception as e:
    elapsed = time.time() - start
    print(f"✗ Failed after {elapsed:.2f}s: {e}")
    exit(1)

# =============================================================================
# TEST 5: Import scikit-image (for SSIM)
# =============================================================================
print_section("TEST 5: Testing scikit-image import...")
try:
    from skimage.metrics import structural_similarity as ssim
    print("✓ scikit-image available")
    
    # Quick SSIM test
    print("\n⏳ Testing SSIM performance on small arrays...")
    img1 = np.random.rand(1000, 1000)
    img2 = np.random.rand(1000, 1000)
    
    start = time.time()
    score = ssim(img1, img2, data_range=1.0)
    elapsed = time.time() - start
    
    print(f"✓ SSIM test completed in {elapsed:.2f}s")
    
    # Extrapolate to full size
    if data.shape[0] > 1000:
        full_time_estimate = elapsed * (data.shape[0]/1000)**2
        print(f"\n  Estimated time for full {data.shape} image: {full_time_estimate:.1f}s")
        if full_time_estimate > 60:
            print(f"  ⚠️  Full-size SSIM would take ~{full_time_estimate/60:.1f} minutes!")
            print(f"     Use downsample_factor=4 or higher")
    
except ImportError:
    print("✗ scikit-image not installed (SSIM will be skipped)")

# =============================================================================
# TEST 6: Import other dependencies
# =============================================================================
print_section("TEST 6: Checking other dependencies...")

dependencies = [
    ('geopandas', 'geopandas'),
    ('xdem', 'xdem'),
    ('openpiv.pyprocess', 'openpiv'),
]

for module_name, package_name in dependencies:
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
    except ImportError:
        print(f"✗ {package_name} - Not installed")

# =============================================================================
# SUMMARY
# =============================================================================
print_section("SUMMARY")

print("""
If the script hung at any of the above steps, that's your problem!

Common hang points:
1. TEST 3 (Opening file) → Network drive issue
2. TEST 4 (Reading data) → Network drive or memory issue  
3. TEST 5 (SSIM test)    → Computational bottleneck

Solutions:
• If hung at TEST 3/4: Copy files to local SSD
• If hung at TEST 5: Use downsample_factor=4+ in check_hillshade_quality()
• If all tests passed: The hang is likely in your original code structure

Next steps:
1. Note which test hung (if any)
2. Check your console output from the original script
3. Compare to the section headers in the fixed script
4. Use the fixed script with proper variable ordering

The fixed script includes:
  ✓ Proper execution order
  ✓ No duplicate loading
  ✓ Downsampled SSIM
  ✓ Better progress reporting
""")

print("="*70)
print("  Test complete! ")
print("="*70)