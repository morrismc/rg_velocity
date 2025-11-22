"""
DEM Quality Diagnostic Script
==============================

This script helps diagnose:
1. Is the grid pattern in the original 2024 DEM or introduced by harmonization?
2. Should we use original DEMs instead of harmonized?
3. What's causing the apparent worsening of results?

Run this to understand your data before proceeding with analysis.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ============================================================
# CONFIGURATION
# ============================================================

# Original DEMs (before harmonization)
original_dems = {
    '2018': r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_LIDAR/2018_0p5m_upper_rg_dem_larger_roi.tif",
    '2023': r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2023_lidar/2023_0p5m_4_imcorr_upper_rg_larger_roi.tif",
    '2024': r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2024_lidar/2024_0p5_4_imcorr__upper_rg_larger_roi.tif"
}

# Harmonized DEMs (after preprocessing)
harmonized_dems = {
    '2018': r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems/2018_0p5m_upper_rg_dem_larger_roi_harmonized.tif",
    '2023': r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems/2023_0p5m_4_imcorr_upper_rg_larger_roi_harmonized.tif",
    '2024': r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Code/preprocessed_dems/2024_0p5_4_imcorr__upper_rg_larger_roi_harmonized.tif"
}

output_dir = Path("dem_diagnostics")
output_dir.mkdir(exist_ok=True)


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


def detect_grid_pattern(data, window_size=50):
    """
    Detect grid/checkerboard patterns using Fourier analysis.
    Returns pattern strength (0-1, higher = more pattern).
    """
    # Get valid region
    valid = np.isfinite(data)
    if not np.any(valid):
        return 0.0, None
    
    # Extract center region for analysis
    center_y = data.shape[0] // 2
    center_x = data.shape[1] // 2
    half_win = window_size // 2
    
    window = data[center_y-half_win:center_y+half_win, 
                  center_x-half_win:center_x+half_win].copy()
    
    # Remove NaNs
    if np.any(~np.isfinite(window)):
        window = np.where(np.isfinite(window), window, np.nanmean(window))
    
    # Detrend
    from scipy.signal import detrend
    window_detrend = detrend(detrend(window, axis=0), axis=1)
    
    # FFT
    fft = np.fft.fft2(window_detrend)
    power = np.abs(fft)**2
    
    # Check for peaks at high frequencies (grid pattern signature)
    # Grid patterns show up as peaks in upper-right quadrant
    h, w = power.shape
    high_freq_region = power[h//4:h//2, w//4:w//2]
    low_freq_region = power[1:h//4, 1:w//4]
    
    high_freq_power = np.mean(high_freq_region)
    low_freq_power = np.mean(low_freq_region)
    
    # Pattern strength: ratio of high to low frequency power
    if low_freq_power > 0:
        pattern_strength = high_freq_power / low_freq_power
    else:
        pattern_strength = 0.0
    
    return pattern_strength, power


def calculate_local_variance(data, window_size=3):
    """Calculate local variance to detect high-frequency noise."""
    from scipy.ndimage import uniform_filter
    
    # --- FIX for RuntimeWarning: Mean of empty slice ---
    # Check for valid data *first* to avoid filtering an all-NaN array
    valid_input = np.isfinite(data)
    if not np.any(valid_input):
        return np.nan
    # --- END FIX ---

    local_mean = uniform_filter(data, size=window_size, mode='constant', cval=np.nan)
    local_var = uniform_filter(data**2, size=window_size, mode='constant', cval=np.nan) - local_mean**2
    
    # --- FIX ---
    # Create the valid mask *from the result*, which handles NaNs
    # from both the input and the filter edges.
    valid_output = np.isfinite(local_var)
    if not np.any(valid_output):
        return np.nan
    return np.nanmean(local_var[valid_output])
    # --- END FIX ---


def compare_dems(dem1, dem2, name1, name2):
    """
    Compare two DEMs and return statistics.
    Handles shape mismatches by using overlapping region only.
    """
    # Check if shapes match
    if dem1.shape != dem2.shape:
        print(f"  ⚠️ Shape mismatch: {dem1.shape} vs {dem2.shape}")
        print(f"     Using overlapping region for comparison")
        
        # Find overlapping region
        min_rows = min(dem1.shape[0], dem2.shape[0])
        min_cols = min(dem1.shape[1], dem2.shape[1])
        
        # Crop both to overlap
        dem1_crop = dem1[:min_rows, :min_cols]
        dem2_crop = dem2[:min_rows, :min_cols]
        
        # Only compare where both are valid
        valid = np.isfinite(dem1_crop) & np.isfinite(dem2_crop)
        
        if not np.any(valid):
            return None
        
        diff = dem2_crop[valid] - dem1_crop[valid]
        n_overlap = min_rows * min_cols
        pct_overlap = 100 * n_overlap / max(dem1.size, dem2.size)
        
        print(f"     Overlap region: {min_rows}×{min_cols} ({pct_overlap:.1f}% of larger)")
    else:
        # Same shape - compare directly
        valid = np.isfinite(dem1) & np.isfinite(dem2)
        
        if not np.any(valid):
            return None
        
        diff = dem2[valid] - dem1[valid]
    
    stats_dict = {
        'mean': np.mean(diff),
        'std': np.std(diff),
        'rmse': np.sqrt(np.mean(diff**2)),
        'mad': np.median(np.abs(diff - np.median(diff))),
        'nmad': 1.4826 * np.median(np.abs(diff - np.median(diff))),
        'min': np.min(diff),
        'max': np.max(diff),
        'n_pixels': np.sum(valid)
    }
    
    return stats_dict


# ============================================================
# RUN DIAGNOSTICS
# ============================================================

print("="*70)
print("DEM QUALITY DIAGNOSTICS")
print("="*70)

# Load all DEMs
print("\nLoading DEMs...")
original_data = {}
harmonized_data = {}

for name in ['2018', '2023', '2024']:
    print(f"\n{name}:")
    
    # Original
    try:
        orig_dem, orig_meta = load_dem(original_dems[name])
        original_data[name] = (orig_dem, orig_meta)
        print(f"  Original: {orig_meta['shape']}")
    except Exception as e:
        print(f"  ✗ Original failed: {e}")
        original_data[name] = None
    
    # Harmonized
    try:
        harm_dem, harm_meta = load_dem(harmonized_dems[name])
        harmonized_data[name] = (harm_dem, harm_meta)
        print(f"  Harmonized: {harm_meta['shape']}")
    except Exception as e:
        print(f"  ✗ Harmonized failed: {e}")
        harmonized_data[name] = None


# ============================================================
# DIAGNOSTIC 1: Check for grid patterns
# ============================================================

print("\n" + "="*70)
print("DIAGNOSTIC 1: GRID PATTERN DETECTION")
print("="*70)

print("\nAnalyzing for checkerboard/grid artifacts...")
print("\nPattern Strength (higher = more grid pattern):")
print("-"*70)

for name in ['2018', '2023', '2024']:
    print(f"\n{name}:")
    
    # Original
    if original_data[name] is not None:
        dem, _ = original_data[name]
        pattern, power = detect_grid_pattern(dem)
        var = calculate_local_variance(dem)
        print(f"  Original:    Pattern={pattern:.4f}, Local_Var={var:.6f}")
    
    # Harmonized
    if harmonized_data[name] is not None:
        dem, _ = harmonized_data[name]
        pattern, power = detect_grid_pattern(dem)
        var = calculate_local_variance(dem)
        print(f"  Harmonized:  Pattern={pattern:.4f}, Local_Var={var:.6f}")

print("\nInterpretation:")
print("  Pattern < 0.01: No grid artifact")
print("  Pattern 0.01-0.05: Mild grid artifact")
print("  Pattern > 0.05: Significant grid artifact")


# ============================================================
# DIAGNOSTIC 2: Original vs Harmonized comparison
# ============================================================

print("\n" + "="*70)
print("DIAGNOSTIC 2: HARMONIZATION IMPACT")
print("="*70)

print("\nComparing original vs harmonized DEMs...")
print("(Should be near-zero for 2018 & 2023, small for 2024)")
print("-"*70)

for name in ['2018', '2023', '2024']:
    print(f"\n{name}:")
    
    if original_data[name] is not None and harmonized_data[name] is not None:
        orig_dem, _ = original_data[name]
        harm_dem, _ = harmonized_data[name]
        
        stats_dict = compare_dems(orig_dem, harm_dem, 'original', 'harmonized')
        
        if stats_dict:
            print(f"  RMSE: {stats_dict['rmse']:.4f} m")
            print(f"  Mean: {stats_dict['mean']:.4f} m")
            print(f"  Std:  {stats_dict['std']:.4f} m")
            print(f"  NMAD: {stats_dict['nmad']:.4f} m")
            
            # Flag issues
            if name in ['2018', '2023'] and stats_dict['rmse'] > 0.1:
                print(f"  ⚠️ WARNING: High RMSE for {name} (should be ~0 since it didn't need harmonization)")
            elif name == '2024' and stats_dict['rmse'] > 0.2:
                print(f"  ⚠️ WARNING: High RMSE for 2024 harmonization")


# ============================================================
# DIAGNOSTIC 3: Shape and transform consistency
# ============================================================

print("\n" + "="*70)
print("DIAGNOSTIC 3: SPATIAL CONSISTENCY")
print("="*70)

print("\nOriginal DEMs:")
for name in ['2018', '2023', '2024']:
    if original_data[name] is not None:
        _, meta = original_data[name]
        print(f"  {name}: Shape={meta['shape']}, Transform={(meta['transform'][0], meta['transform'][4])}")

print("\nHarmonized DEMs:")
for name in ['2018', '2023', '2024']:
    if harmonized_data[name] is not None:
        _, meta = harmonized_data[name]
        print(f"  {name}: Shape={meta['shape']}, Transform={(meta['transform'][0], meta['transform'][4])}")


# ============================================================
# DIAGNOSTIC 4: Visual comparison
# ============================================================

print("\n" + "="*70)
print("DIAGNOSTIC 4: VISUAL COMPARISON")
print("="*70)
print("\nGenerating comparison plots...")

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for idx, name in enumerate(['2018', '2023', '2024']):
    # Original
    if original_data[name] is not None:
        dem, _ = original_data[name]
        valid = np.isfinite(dem)
        im = axes[idx, 0].imshow(dem, cmap='terrain', vmin=3000, vmax=3350)
        axes[idx, 0].set_title(f'{name} Original')
        plt.colorbar(im, ax=axes[idx, 0])
    
    # Harmonized
    if harmonized_data[name] is not None:
        dem, _ = harmonized_data[name]
        valid = np.isfinite(dem)
        im = axes[idx, 1].imshow(dem, cmap='terrain', vmin=3000, vmax=3350)
        axes[idx, 1].set_title(f'{name} Harmonized')
        plt.colorbar(im, ax=axes[idx, 1])
    
    # Difference (handling shape mismatch)
    if original_data[name] is not None and harmonized_data[name] is not None:
        orig_dem, _ = original_data[name]
        harm_dem, _ = harmonized_data[name]
        
        # Handle shape mismatch
        if orig_dem.shape != harm_dem.shape:
            min_rows = min(orig_dem.shape[0], harm_dem.shape[0])
            min_cols = min(orig_dem.shape[1], harm_dem.shape[1])
            orig_dem = orig_dem[:min_rows, :min_cols]
            harm_dem = harm_dem[:min_rows, :min_cols]
        
        diff = harm_dem - orig_dem
        valid = np.isfinite(diff)
        
        if np.any(valid):
            vmax = np.percentile(np.abs(diff[valid]), 95)
            
            # --- FIX for ValueError ---
            # If vmax is 0 (no difference), set a small default
            # to prevent TwoSlopeNorm from failing.
            if vmax == 0:
                vmax = 0.01 # 1 cm default range
            # --- END FIX ---

            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = axes[idx, 2].imshow(diff, cmap='RdBu_r', norm=norm)
            axes[idx, 2].set_title(f'{name} Diff (Harm - Orig)')
            plt.colorbar(im, ax=axes[idx, 2], label='Difference (m)')

plt.tight_layout()
plt.savefig(output_dir / 'dem_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {output_dir / 'dem_comparison.png'}")
plt.close()


# ============================================================
# DIAGNOSTIC 5: Detailed 2024 Grid Pattern Analysis
# ============================================================

print("\n" + "="*70)
print("DIAGNOSTIC 5: 2024 GRID PATTERN ANALYSIS")
print("="*70)
print("\nDetailed analysis of 2024 DEM grid artifact...")

if original_data['2024'] is not None and harmonized_data['2024'] is not None:
    orig_2024, _ = original_data['2024']
    harm_2024, _ = harmonized_data['2024']
    
    # Extract same-size regions for comparison
    min_rows = min(orig_2024.shape[0], harm_2024.shape[0])
    min_cols = min(orig_2024.shape[1], harm_2024.shape[1])
    orig_2024_crop = orig_2024[:min_rows, :min_cols]
    harm_2024_crop = harm_2024[:min_rows, :min_cols]
    
    # Create detailed comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: DEMs
    im1 = axes[0, 0].imshow(orig_2024_crop, cmap='terrain', vmin=3000, vmax=3350)
    axes[0, 0].set_title('2024 Original DEM', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(harm_2024_crop, cmap='terrain', vmin=3000, vmax=3350)
    axes[0, 1].set_title('2024 Harmonized DEM', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Difference
    diff_2024 = harm_2024_crop - orig_2024_crop
    valid_diff = np.isfinite(diff_2024)
    if np.any(valid_diff):
        vmax_diff = np.percentile(np.abs(diff_2024[valid_diff]), 95)
        
        # --- FIX for ValueError (safety check) ---
        if vmax_diff == 0:
            vmax_diff = 0.01
        # --- END FIX ---

        from matplotlib.colors import TwoSlopeNorm
        norm_diff = TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)
        im3 = axes[0, 2].imshow(diff_2024, cmap='RdBu_r', norm=norm_diff)
        axes[0, 2].set_title('Difference (Harmonized - Original)', fontsize=12, fontweight='bold')
        plt.colorbar(im3, ax=axes[0, 2], label='Elevation change (m)')
    
    # Row 2: Zoomed sections to see grid pattern
    # Choose center region
    center_y, center_x = min_rows // 2, min_cols // 2
    zoom_size = 100
    y1, y2 = center_y - zoom_size, center_y + zoom_size
    x1, x2 = center_x - zoom_size, center_x + zoom_size
    
    orig_zoom = orig_2024_crop[y1:y2, x1:x2]
    harm_zoom = harm_2024_crop[y1:y2, x1:x2]
    
    im4 = axes[1, 0].imshow(orig_zoom, cmap='terrain', vmin=3000, vmax=3350)
    axes[1, 0].set_title('Original (Zoomed Center)', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(harm_zoom, cmap='terrain', vmin=3000, vmax=3350)
    axes[1, 1].set_title('Harmonized (Zoomed Center)', fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Histogram comparison
    orig_valid = orig_2024_crop[np.isfinite(orig_2024_crop)]
    harm_valid = harm_2024_crop[np.isfinite(harm_2024_crop)]
    
    axes[1, 2].hist(orig_valid.flatten(), bins=50, alpha=0.5, label='Original', color='blue')
    axes[1, 2].hist(harm_valid.flatten(), bins=50, alpha=0.5, label='Harmonized', color='red')
    axes[1, 2].set_xlabel('Elevation (m)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Elevation Distribution', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '2024_grid_analysis.png', dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_dir / '2024_grid_analysis.png'}")
    plt.close()
else:
    print("  ⚠️ Cannot analyze 2024 - data not available")


# ============================================================
# RECOMMENDATION
# ============================================================

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

# Check if 2018 and 2023 originally matched
if original_data['2018'] is not None and original_data['2023'] is not None:
    _, meta2018 = original_data['2018']
    _, meta2023 = original_data['2023']
    
    if (meta2018['shape'] == meta2023['shape'] and 
        meta2018['transform'] == meta2023['transform']):
        print("\n✓ 2018 and 2023 originally had matching shape and transform")
        print("\n🎯 RECOMMENDED APPROACH:")
        print("  1. Use ORIGINAL 2018 and 2023 DEMs (no harmonization needed)")
        print("  2. Use HARMONIZED 2024 DEM (to match 2018/2023 grid)")
        print("\n  This avoids unnecessary resampling of 2018 and 2023.")
    else:
        print("\n2018 and 2023 had different grids - harmonization was necessary")

# Check 2024 grid pattern
if original_data['2024'] is not None:
    dem, _ = original_data['2024']
    pattern, _ = detect_grid_pattern(dem)
    
    if pattern > 0.05:
        print("\n⚠️  GRID PATTERN IN ORIGINAL 2024 DEM:")
        print(f"  Pattern strength: {pattern:.4f}")
        print("\n  This pattern exists in your source data, not from harmonization.")
        print("  Options:")
        print("    A. Accept it (if amplitude is small)")
        print("    B. Contact data provider about processing")
        print("    C. Apply spatial filtering (may reduce accuracy)")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print(f"\nResults saved to: {output_dir.absolute()}")