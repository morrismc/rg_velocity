"""
CORRECTED: Multi-Temporal Vertical Displacement Analysis
=========================================================

Critical fix: Use ORIGINAL 2018 & 2023 (already matched), 
             HARMONIZED 2024 only (to match their grid)

This avoids unnecessary resampling of 2018 and 2023 which may have
introduced artifacts in the previous approach.

(v3: Added robust statistical filter for bias correction and plt.show())
"""

import numpy as np
import rasterio
import geopandas as gpd
import xdem
from xdem.coreg import NuthKaab, ICP, Deramp
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from pathlib import Path
import warnings
from datetime import datetime
from collections import OrderedDict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='geoutils')
warnings.filterwarnings('ignore', message=".*'partition' will ignore the 'mask'.*")
warnings.filterwarnings('ignore', category=DeprecationWarning, module='xdem')


class MultiTemporalDisplacementAnalyzer:
    """
    Analyzes vertical displacement across multiple time periods.
    """
    
    def __init__(self, stable_area_shapefile):
        """Initialize the analyzer."""
        self.stable_shapefile = stable_area_shapefile
        if not Path(stable_area_shapefile).exists():
            raise FileNotFoundError(f"Stable area shapefile not found: {stable_area_shapefile}")
        
        self.epochs = OrderedDict()
        self.dates = OrderedDict()
        self.stable_mask = None
        self.transform = None
        self.crs = None
        self.snow_masks = {}
        
    def add_epoch(self, epoch_name, dem_path, date=None, nodata_threshold=-1e10):
        """Add a DEM epoch to the analysis."""
        print(f"\nAdding epoch: {epoch_name}")
        print(f"  Loading: {Path(dem_path).name}")
        
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float64)
            transform = src.transform
            crs = src.crs
            nodata_val = src.nodata
            
            print(f"  Shape: {dem.shape}")
        
        # Identify nodata pixels
        nodata_mask = (
            np.isnan(dem) | 
            np.isinf(dem) |
            (dem < nodata_threshold) |
            (dem > 10000)
        )
        
        if nodata_val is not None and np.isfinite(nodata_val):
            nodata_mask |= (dem == nodata_val)
        
        dem[nodata_mask] = np.nan
        
        n_nodata = np.sum(nodata_mask)
        pct_nodata = 100 * n_nodata / dem.size
        print(f"  Nodata: {n_nodata} pixels ({pct_nodata:.1f}%)")
        
        if not np.all(nodata_mask):
            valid_dem = dem[~nodata_mask]
            print(f"  Valid range: [{np.min(valid_dem):.1f}, {np.max(valid_dem):.1f}] m")
        
        self.epochs[epoch_name] = {
            'dem': dem,
            'nodata_mask': nodata_mask
        }
        
        if date is not None:
            if isinstance(date, str):
                self.dates[epoch_name] = datetime.strptime(date, '%Y-%m-%d')
            else:
                self.dates[epoch_name] = date
        
        if self.transform is None:
            self.transform = transform
            self.crs = crs
        else:
            if transform != self.transform:
                print(f"  ⚠️ WARNING: Transform differs from first epoch")
            if crs != self.crs:
                print(f"  ⚠️ WARNING: CRS differs from first epoch")
    
    def create_stable_mask(self):
        """Create boolean mask from stable ground shapefile."""
        if not self.epochs:
            raise ValueError("Must add at least one epoch before creating stable mask")
        
        print("\n--- CREATING STABLE MASK ---")
        
        first_epoch = list(self.epochs.values())[0]
        reference_shape = first_epoch['dem'].shape
        
        stable_gdf = gpd.read_file(self.stable_shapefile)
        print(f"  Loaded {len(stable_gdf)} stable area polygon(s)")
        
        if stable_gdf.crs != self.crs:
            print(f"  Reprojecting from {stable_gdf.crs} to {self.crs}")
            stable_gdf = stable_gdf.to_crs(self.crs)
        
        self.stable_mask = geometry_mask(
            stable_gdf.geometry,
            transform=self.transform,
            invert=True,
            out_shape=reference_shape
        )
        
        n_stable = np.sum(self.stable_mask)
        pct_stable = 100 * n_stable / self.stable_mask.size
        
        if n_stable == 0:
            raise ValueError("Stable mask is empty! Check shapefile overlap.")
        
        print(f"  Stable pixels: {n_stable} ({pct_stable:.1f}%)")
    
    def detect_snow_change(self, dh, threshold=0.3):
        """Detect potential snow cover changes."""
        from scipy import ndimage
        
        large_changes = np.abs(dh) > threshold
        structure = ndimage.generate_binary_structure(2, 2)
        large_changes_filtered = ndimage.binary_opening(
            large_changes, 
            structure=structure, 
            iterations=2
        )
        snow_mask = ndimage.binary_dilation(
            large_changes_filtered,
            structure=structure,
            iterations=1
        )
        
        return snow_mask
    
    def calculate_pairwise_displacement(self, epoch1_name, epoch2_name, 
                                       coreg_method='nuth_kaab',
                                       filter_outliers=True,
                                       detect_snow=True,
                                       apply_bias_correction=True):
        """Calculate displacement between two epochs with bias correction."""
        print(f"\n--- CALCULATING DISPLACEMENT: {epoch1_name} → {epoch2_name} ---")
        
        if epoch1_name not in self.epochs or epoch2_name not in self.epochs:
            raise ValueError(f"Epochs not found")
        
        dem1 = self.epochs[epoch1_name]['dem']
        dem2 = self.epochs[epoch2_name]['dem']
        
        # Calculate time delta
        if epoch1_name in self.dates and epoch2_name in self.dates:
            dt = (self.dates[epoch2_name] - self.dates[epoch1_name]).days / 365.25
            print(f"  Time interval: {dt:.2f} years")
        else:
            try:
                year1 = float(epoch1_name)
                year2 = float(epoch2_name)
                dt = year2 - year1
                print(f"  Time interval: {dt:.1f} years (from epoch names)")
            except:
                dt = 1.0
                print(f"  ⚠️ WARNING: Cannot determine time interval, using 1.0 year")
        
        # Create xdem DEM objects
        dem1_xdem = xdem.DEM.from_array(
            data=dem1,
            transform=self.transform,
            crs=self.crs,
            nodata=np.nan
        )
        
        dem2_xdem = xdem.DEM.from_array(
            data=dem2,
            transform=self.transform,
            crs=self.crs,
            nodata=np.nan
        )
        
        # Co-registration
        print(f"  Co-registration method: {coreg_method}")
        if coreg_method == 'none':
            dem2_aligned = dem2_xdem
            print(f"  Skipping co-registration")
        elif coreg_method == 'nuth_kaab':
            try:
                coreg = xdem.coreg.NuthKaab()
                coreg.fit(dem1_xdem, dem2_xdem, inlier_mask=self.stable_mask, verbose=False)
                dem2_aligned = coreg.apply(dem2_xdem)
                print(f"  ✓ Nuth-Kaab successful")
            except Exception as e:
                print(f"  ✗ Failed: {e}, skipping co-registration")
                dem2_aligned = dem2_xdem
        else:  # icp
            try:
                coreg = xdem.coreg.ICP()
                coreg.fit(dem1_xdem, dem2_xdem, inlier_mask=self.stable_mask, verbose=False)
                dem2_aligned = coreg.apply(dem2_xdem)
                print(f"  ✓ ICP successful")
            except Exception as e:
                print(f"  ✗ Failed: {e}, trying Nuth-Kaab instead")
                coreg = xdem.coreg.NuthKaab()
                coreg.fit(dem1_xdem, dem2_xdem, inlier_mask=self.stable_mask, verbose=False)
                dem2_aligned = coreg.apply(dem2_xdem)
        
        # Calculate difference (post-NuthKaab)
        dh = dem2_aligned - dem1_xdem
        dh_array = dh.data.copy()
        valid = np.isfinite(dh_array)

        # [ ROBUST STATISTICAL FILTERING BLOCK ]
        
        # 1. Detect high-change areas (snow/trees) for VISUALIZATION ONLY
        #    This is no longer used for the correction.
        snow_mask = None
        if detect_snow:
            print("  Detecting potential snow/vegetation changes (for visualization)...")
            snow_mask = self.detect_snow_change(dh_array, threshold=0.3)
            pair_key = f"{epoch1_name}_{epoch2_name}"
            self.snow_masks[pair_key] = snow_mask
        
        # 2. Create the "Clean" Stable Mask using a robust statistical filter
        
        # Get all valid pixels in the user's shapefile (the 'contaminated' mask)
        contaminated_stable_mask = self.stable_mask & valid
        clean_stable_mask = np.zeros_like(contaminated_stable_mask) # Start with an empty mask
        
        if np.any(contaminated_stable_mask):
            # Get all raw elevation differences from the contaminated mask
            stable_dh_raw = dh_array[contaminated_stable_mask]
            
            # Calculate robust stats (median and NMAD)
            median_bias = np.nanmedian(stable_dh_raw)
            nmad_bias = 1.4826 * np.nanmedian(np.abs(stable_dh_raw - median_bias))
            
            # Handle cases with zero variance
            if nmad_bias == 0:
                nmad_bias = 1.4826 * np.nanstd(stable_dh_raw) # Fallback to std
            if nmad_bias == 0:
                nmad_bias = 0.1 # Fallback to 10cm if still zero
            
            print(f"  Robust stats on raw stable mask: Median={median_bias:.3f}, NMAD={nmad_bias:.3f}")
            
            # Define our filter: only pixels within 2x NMAD of the median
            # This automatically rejects the long tail of snow/trees
            lower_bound = median_bias - 2.0 * nmad_bias
            upper_bound = median_bias + 2.0 * nmad_bias
            
            print(f"  Filtering stable mask to range: [{lower_bound:.3f}, {upper_bound:.3f}]")
            
            # Create a mask of pixels *within* the contaminated mask that pass the filter
            clean_pixels_in_mask = (stable_dh_raw >= lower_bound) & (stable_dh_raw <= upper_bound)
            
            # Now, update the full-resolution clean_stable_mask
            clean_stable_mask[contaminated_stable_mask] = clean_pixels_in_mask

            n_clean = np.sum(clean_stable_mask)
            n_raw = np.sum(contaminated_stable_mask)
            pct_kept = 100 * n_clean / n_raw if n_raw > 0 else 0
            print(f"  Statistical filter kept {n_clean}/{n_raw} stable pixels ({pct_kept:.1f}%)")

        else:
            print("  ⚠️ No valid pixels in initial stable mask.")
        
        # 3. Quality metrics and bias correction using the STATISTICALLY CLEAN mask
        if np.any(clean_stable_mask):
            # Calculate pre-correction stats (from the Nuth-Kaab aligned dh)
            stable_dh_pre = dh_array[clean_stable_mask]
            mean_bias_pre = np.nanmean(stable_dh_pre)
            std_bias_pre = np.nanstd(stable_dh_pre)
            nmad_pre = 1.4826 * np.nanmedian(np.abs(stable_dh_pre - median_bias))
            
            print(f"  Stable area quality (using STATISTICALLY CLEAN mask, post-NuthKaab):")
            print(f"    Pixels used: {np.sum(clean_stable_mask)}")
            print(f"    Mean: {mean_bias_pre:.3f} m")
            print(f"    Std:  {std_bias_pre:.3f} m")
            print(f"    NMAD: {nmad_pre:.3f} m")
            
            # Apply bias correction
            if apply_bias_correction:
                print(f"  ⚙️  Applying 1st-degree polynomial deramping (tilt correction)...")
                
                # Use poly_order=1: It's the right model for simple tilts.
                deramp = Deramp(poly_order=1)
                
                # Fit Deramp to the two aligned DEMs, using our new clean mask
                deramp.fit(
                    reference_elev=dem1_xdem, 
                    to_be_aligned_elev=dem2_aligned, 
                    inlier_mask=clean_stable_mask
                )
                
                # Apply the deramping correction
                dem2_final_aligned = deramp.apply(dem2_aligned)
                
                # Recalculate the final difference array
                dh_final = dem2_final_aligned - dem1_xdem
                dh_array = dh_final.data.copy()

                # Recalculate stats *after* deramping for reporting
                stable_dh_corrected = dh_array[clean_stable_mask]
                mean_bias_after = np.nanmean(stable_dh_corrected)
                std_bias_after = np.nanstd(stable_dh_corrected)
                nmad_after = 1.4826 * np.nanmedian(np.abs(stable_dh_corrected - np.nanmedian(stable_dh_corrected)))
                
                print(f"  ✓ Stable area mean after correction: {mean_bias_after:.4f} m")
                print(f"  ✓ NMAD after correction: {nmad_after:.3f} m")
                
                # Save the final stats
                mean_bias = mean_bias_after
                std_bias = std_bias_after
                nmad = nmad_after
                
            else:
                print(f"  ⚠️ Bias correction disabled")
                mean_bias = mean_bias_pre
                std_bias = std_bias_pre
                nmad = nmad_pre
        else:
            mean_bias = std_bias = nmad = np.nan
            print(f"  ⚠️ WARNING: No valid stable area pixels after cleaning!")
            print(f"  Check stable mask and snow mask overlap.")
        
        # [ END OF REPLACEMENT BLOCK ]
        
        # 4. Filter outliers
        dh_filtered = dh_array.copy()
        outlier_mask = np.zeros_like(dh_array, dtype=bool)
        
        if filter_outliers:
            # Re-calculate valid mask on the *final* corrected dh_array
            valid = np.isfinite(dh_array)
            
            q1, q3 = np.nanpercentile(dh_array[valid], [25, 75])
            iqr = q3 - q1
            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr
            
            outlier_mask = (dh_array < lower) | (dh_array > upper)
            n_outliers = np.sum(outlier_mask & valid)
            pct_outliers = 100 * n_outliers / np.sum(valid) if np.any(valid) else 0
            
            dh_filtered[outlier_mask] = np.nan
            print(f"  Outliers removed: {n_outliers} ({pct_outliers:.1f}%)")
        
        # 5. Calculate velocity
        velocity = dh_filtered / dt
        
        return {
            'dh': dh_filtered,
            'velocity': velocity,
            'time_delta': dt,
            'valid_mask': valid & ~outlier_mask,
            'outlier_mask': outlier_mask,
            'snow_mask': snow_mask,
            'quality': {
                'mean_bias': mean_bias,
                'std_bias': std_bias,
                'nmad': nmad,
                'coreg_method': coreg_method,
                'bias_corrected': apply_bias_correction
            }
        }
    
    def calculate_all_pairwise(self, coreg_method='nuth_kaab', 
                               filter_outliers=True, 
                               detect_snow=True,
                               apply_bias_correction=True):
        """Calculate displacement for all consecutive epoch pairs."""
        print("\n" + "="*60)
        print("CALCULATING ALL PAIRWISE DISPLACEMENTS")
        print("="*60)
        
        results = OrderedDict()
        epoch_names = list(self.epochs.keys())
        
        for i in range(len(epoch_names) - 1):
            epoch1 = epoch_names[i]
            epoch2 = epoch_names[i + 1]
            pair_key = f"{epoch1}_{epoch2}"
            
            results[pair_key] = self.calculate_pairwise_displacement(
                epoch1, epoch2,
                coreg_method=coreg_method,
                filter_outliers=filter_outliers,
                detect_snow=detect_snow,
                apply_bias_correction=apply_bias_correction
            )
        
        return results
    
    def visualize_pairwise_comparison(self, results, output_dir, show_snow_mask=True):
        """Create comprehensive visualization."""
        print("\n--- CREATING PAIRWISE COMPARISON PLOTS ---")
        
        n_pairs = len(results)
        fig, axes = plt.subplots(2, n_pairs, figsize=(6*n_pairs, 12))
        if n_pairs == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (pair_key, result) in enumerate(results.items()):
            dh = result['dh']
            velocity = result['velocity']
            dt = result['time_delta']
            
            dh_copy = np.array(dh, dtype=np.float64)
            valid_dh = dh_copy[np.isfinite(dh_copy)]
            
            if len(valid_dh) > 0:
                vmax = np.percentile(np.abs(valid_dh), 95)
                if vmax == 0: vmax = 0.5 # Handle no variance case
            else:
                vmax = 0.5
            
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            
            # Top row: Displacement
            im1 = axes[0, idx].imshow(dh, cmap='RdBu_r', norm=norm)
            axes[0, idx].set_title(f'{pair_key.replace("_", " → ")}\n'
                                  f'Displacement (m), Δt={dt:.2f}yr',
                                  fontsize=11, fontweight='bold')
            
            # Stable areas overlay
            if self.stable_mask is not None:
                axes[0, idx].contour(
                    self.stable_mask, 
                    levels=[0.5], 
                    colors='white', 
                    linewidths=5,
                    linestyles='-',
                    alpha=0.8
                )
                axes[0, idx].contour(
                    self.stable_mask, 
                    levels=[0.5], 
                    colors='lime', 
                    linewidths=3,
                    linestyles='-',
                    alpha=1.0
                )
            
            # Snow mask overlay - ENHANCED
            if show_snow_mask and result['snow_mask'] is not None:
                snow_mask = result['snow_mask']
                
                # Thick contour lines
                try:
                    axes[0, idx].contour(
                        snow_mask.astype(float), 
                        levels=[0.5], 
                        colors='cyan', 
                        linewidths=4,
                        linestyles='--',
                        alpha=1.0
                    )
                except:
                    pass
                
                # Filled regions for visibility
                try:
                    snow_masked = np.ma.masked_where(~snow_mask, np.ones_like(dh))
                    axes[0, idx].contourf(
                        snow_masked,
                        levels=[0.5, 1.5],
                        colors=['cyan'],
                        alpha=0.2
                    )
                except:
                    pass
            
            plt.colorbar(im1, ax=axes[0, idx], label='Subsidence ← | → Uplift (m)')
            axes[0, idx].set_xlabel('Column (pixels)')
            axes[0, idx].set_ylabel('Row (pixels)')
            
            # Bottom row: Velocity
            vel_copy = np.array(velocity, dtype=np.float64)
            valid_vel = vel_copy[np.isfinite(vel_copy)]
            
            if len(valid_vel) > 0:
                vmax_vel = np.percentile(np.abs(valid_vel), 95)
                if vmax_vel == 0: vmax_vel = 0.1 # Handle no variance case
            else:
                vmax_vel = 0.1
            
            norm_vel = TwoSlopeNorm(vmin=-vmax_vel, vcenter=0, vmax=vmax_vel)
            im2 = axes[1, idx].imshow(velocity, cmap='RdBu_r', norm=norm_vel)
            axes[1, idx].set_title(f'Velocity (m/year)', fontsize=11, fontweight='bold')
            plt.colorbar(im2, ax=axes[1, idx], label='Subsidence ← | → Uplift (m/yr)')
            axes[1, idx].set_xlabel('Column (pixels)')
            axes[1, idx].set_ylabel('Row (pixels)')
        
        # Legend
        legend_elements = [
            Patch(facecolor='lime', edgecolor='white', linewidth=2, label='Stable areas')
        ]
        if show_snow_mask:
            legend_elements.append(
                Patch(facecolor='cyan', edgecolor='cyan', linewidth=2, 
                      label='Potential snow', linestyle='--', alpha=0.6)
            )
        
        fig.legend(handles=legend_elements, loc='upper center', 
                  ncol=len(legend_elements), fontsize=11, 
                  bbox_to_anchor=(0.5, 0.98))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / 'pairwise_displacements.png', dpi=200, bbox_inches='tight')
        print(f"  Saved: pairwise_displacements.png")
        
        # --- NEW CODE ---
        # Show the plot in the IDE
        try:
            plt.show()
        except Exception as e:
            print(f"  Note: Could not show plot in IDE. {e}")
        # --- END NEW ---
        
        plt.close()
    
    def create_time_series_plot(self, results, output_dir):
        """Create time series plot."""
        print("  Creating time series plot...")
        
        pairs = []
        mean_dh = []
        median_dh = []
        std_dh = []
        stable_mean = []
        stable_std = []
        time_intervals = []
        
        for pair_key, result in results.items():
            pairs.append(pair_key.replace('_', ' → '))
            
            # --- START MODIFICATION ---
            # 1. Get all valid pixels on the ROCK GLACIER (moving ground)
            #    We exclude the stable mask from our final stats.
            valid_moving_mask = np.isfinite(result['dh']) & ~self.stable_mask
            
            # 2. Get the snow mask for this pair
            snow_mask = result.get('snow_mask') # Use .get() for safety
            
            # 3. Create a "clean" mask by removing snow pixels
            if snow_mask is not None:
                clean_moving_mask = valid_moving_mask & ~snow_mask
            else:
                clean_moving_mask = valid_moving_mask
            
            # 4. Calculate stats ONLY on the clean, moving pixels
            if np.any(clean_moving_mask):
                dh_valid = result['dh'][clean_moving_mask]
            else:
                # Fallback in case mask is empty (to avoid crash)
                dh_valid = np.array([np.nan]) 
            # --- END MODIFICATION ---
            
            mean_dh.append(np.nanmean(dh_valid))
            median_dh.append(np.nanmedian(dh_valid))
            std_dh.append(np.nanstd(dh_valid))
            
            stable_mean.append(result['quality']['mean_bias'])
            stable_std.append(result['quality']['std_bias'])
            time_intervals.append(result['time_delta'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        x = np.arange(len(pairs))
        
        # 1. Mean displacement (NOW ROBUST)
        axes[0, 0].bar(x, mean_dh, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Epoch Pair')
        axes[0, 0].set_ylabel('Mean Displacement (m)')
        axes[0, 0].set_title('Mean Vertical Displacement by Period\n(Moving ground, snow filtered)', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(pairs, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Median with error bars (NOW ROBUST)
        axes[0, 1].errorbar(x, median_dh, yerr=std_dh, fmt='o-', 
                           linewidth=2, markersize=8, capsize=5,
                           color='darkgreen', label='Median ± Std')
        axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Epoch Pair')
        axes[0, 1].set_ylabel('Displacement (m)')
        axes[0, 1].set_title('Median Displacement with Variability\n(Moving ground, snow filtered)', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(pairs, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Stable area quality (This plot remains unchanged, as it should)
        axes[1, 0].bar(x, stable_mean, alpha=0.7, color='lightgreen', 
                      edgecolor='black')
        axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].axhline(0.05, color='orange', linestyle=':', linewidth=1, alpha=0.7)
        axes[1, 0].axhline(-0.05, color='orange', linestyle=':', linewidth=1, alpha=0.7)
        axes[1, 0].set_xlabel('Epoch Pair')
        axes[1, 0].set_ylabel('Mean Displacement (m)')
        axes[1, 0].set_title('Stable Area Quality (After Correction)', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(pairs, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].text(0.02, 0.98, 'Target: |mean| < 0.05m', 
                       transform=axes[1, 0].transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 4. Time intervals and velocities (NOW ROBUST)
        velocities = [m/dt if dt > 0 else 0 for m, dt in zip(median_dh, time_intervals)]
        
        ax4a = axes[1, 1]
        ax4b = ax4a.twinx()
        
        ax4a.bar(x, time_intervals, alpha=0.5, color='lightcoral', 
                edgecolor='black', label='Time interval (years)')
        ax4b.plot(x, velocities, 'o-', linewidth=2, markersize=8, 
                 color='darkblue', label='Velocity (m/yr)')
        ax4b.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        ax4a.set_xlabel('Epoch Pair')
        ax4a.set_ylabel('Time Interval (years)', color='darkred')
        ax4b.set_ylabel('Velocity (m/year)', color='darkblue')
        ax4a.set_title('Time Intervals and Velocities\n(Based on robust median)', fontweight='bold')
        ax4a.set_xticks(x)
        ax4a.set_xticklabels(pairs, rotation=45, ha='right')
        ax4a.tick_params(axis='y', labelcolor='darkred')
        ax4b.tick_params(axis='y', labelcolor='darkblue')
        ax4a.grid(True, alpha=0.3, axis='x')
        
        lines1, labels1 = ax4a.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4a.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'time_series_summary.png', dpi=200, bbox_inches='tight')
        print(f"  Saved: time_series_summary.png")

        # --- NEW CODE ---
        # Show the plot in the IDE
        try:
            plt.show()
        except Exception as e:
            print(f"  Note: Could not show plot in IDE. {e}")
        # --- END NEW ---

        plt.close()

# ======================
# EXAMPLE USAGE
# ======================
# ======================
# EXAMPLE USAGE
# ======================
if __name__ == "__main__":
    
    # --- User Inputs ---
    stable_file = Path(r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\shapefiles\stable_2.shp")
    
    # Define your DEM epochs
    # ** THESE MUST POINT TO THE HARMONIZED OUTPUTS **
    dem_epochs = [
        ('2018', r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\preprocessed_dems\2018_0p5m_upper_rg_dem_larger_roi_harmonized.tif", '2018-09-01'),
        ('2023', r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\preprocessed_dems\GadValleyRG_50cmDEM_2023_harmonized.TIF", '2023-09-01'),
        ('2024', r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\preprocessed_dems\GadValleyRG_50cmDEM_2024_harmonized.TIF", '2024-09-01'),
        ('2025', r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\preprocessed_dems\GadValleyRG_50cmDEM_2025_harmonized.TIF", '2025-09-01'),
    ]
    
    # Analysis parameters
   
   
    
    # Analysis parameters
    COREG_METHOD = 'nuth_kaab'
    FILTER_OUTLIERS = True
    DETECT_SNOW = True
    APPLY_BIAS_CORRECTION = True
    
    try:
        output_dir = Path("results_multitemporal_corrected")
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}\n")
        
        print("="*60)
        print("MULTI-TEMPORAL VERTICAL DISPLACEMENT ANALYSIS")
        print("Using ORIGINAL 2018/2023, HARMONIZED 2024 only")
        print("="*60)
        analyzer = MultiTemporalDisplacementAnalyzer(stable_area_shapefile=stable_file)
        
        for epoch_name, dem_path, date in dem_epochs:
            analyzer.add_epoch(epoch_name, dem_path, date)
        
        analyzer.create_stable_mask()
        
        results = analyzer.calculate_all_pairwise(
            coreg_method=COREG_METHOD,
            filter_outliers=FILTER_OUTLIERS,
            detect_snow=DETECT_SNOW,
            apply_bias_correction=APPLY_BIAS_CORRECTION
        )
        
        analyzer.visualize_pairwise_comparison(results, output_dir, show_snow_mask=DETECT_SNOW)
        analyzer.create_time_series_plot(results, output_dir)
        
        output_file = output_dir / "multitemporal_results.npz"
        save_data = {
            'stable_mask': analyzer.stable_mask,
            'transform': analyzer.transform,
            'epoch_names': list(analyzer.epochs.keys())
        }
        
        for pair_key, result in results.items():
            save_data[f'{pair_key}_dh'] = result['dh']
            save_data[f'{pair_key}_velocity'] = result['velocity']
            save_data[f'{pair_key}_time_delta'] = result['time_delta']
            if result['snow_mask'] is not None:
                save_data[f'{pair_key}_snow_mask'] = result['snow_mask']
        
        np.savez_compressed(output_file, **save_data)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"\nProcessed {len(dem_epochs)} epochs")
        print(f"Generated {len(results)} pairwise comparisons")
        
        print(f"\nQuality Summary (After Bias Correction):")
        print("-" * 60)
        for pair_key, result in results.items():
            quality = result['quality']
            print(f"{pair_key}:")
            print(f"  Mean bias: {quality['mean_bias']:.4f} m")
            print(f"  NMAD: {quality['nmad']:.3f} m")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()