"""
Vertical Displacement Analysis for Rock Glaciers
=================================================

This script focuses exclusively on calculating vertical (Z) displacement
using DEM differencing with xdem. It includes comprehensive quality checks,
calibration using stable ground, and diagnostic visualizations.

Key features:
- Robust nodata detection and handling
- DEM co-registration using stable areas
- Statistical outlier removal
- Quality diagnostic plots
- Result visualization maps
"""

import numpy as np
import rasterio
import geopandas as gpd
import xdem
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='geoutils')
warnings.filterwarnings('ignore', message=".*'partition' will ignore the 'mask'.*")
warnings.filterwarnings('ignore', category=DeprecationWarning, module='xdem')


class VerticalDisplacementAnalyzer:
    """
    Analyzes vertical displacement between two DEMs with calibration
    and comprehensive diagnostics.
    """
    
    def __init__(self, stable_area_shapefile):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        stable_area_shapefile : str or Path
            Path to shapefile defining stable (bedrock) areas for calibration
        """
        self.stable_shapefile = stable_area_shapefile
        if not Path(stable_area_shapefile).exists():
            raise FileNotFoundError(f"Stable area shapefile not found: {stable_area_shapefile}")
        
        self.stable_mask = None
        self.dem1_xdem = None
        self.dem2_xdem = None
        self.transform = None
        self.crs = None
        
    def load_dem(self, dem_path, nodata_threshold=-1e10):
        """
        Load a DEM with robust nodata handling.
        
        The nodata_threshold is set to catch extreme negative values that
        often indicate nodata in float32 rasters.
        
        Parameters:
        -----------
        dem_path : str or Path
            Path to DEM raster file
        nodata_threshold : float
            Values below this are considered nodata (default: -1e10)
            
        Returns:
        --------
        tuple : (dem_array, transform, crs, nodata_value)
        """
        print(f"Loading DEM: {Path(dem_path).name}")
        
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float64)  # Use float64 for precision
            transform = src.transform
            crs = src.crs
            nodata_val = src.nodata
            
            print(f"  Raw DEM range: [{np.min(dem):.2e}, {np.max(dem):.2e}]")
            print(f"  Declared nodata value: {nodata_val}")
            
        # Identify nodata pixels using multiple criteria
        nodata_mask = (
            np.isnan(dem) | 
            np.isinf(dem) |
            (dem < nodata_threshold) |  # Catch extreme negative values
            (dem > 10000)  # Catch unreasonably high values
        )
        
        # Also check if declared nodata value exists
        if nodata_val is not None and np.isfinite(nodata_val):
            nodata_mask |= (dem == nodata_val)
        
        # Set nodata pixels to NaN
        dem[nodata_mask] = np.nan
        
        n_nodata = np.sum(nodata_mask)
        pct_nodata = 100 * n_nodata / dem.size
        
        print(f"  Nodata pixels: {n_nodata} ({pct_nodata:.1f}%)")
        
        if np.all(nodata_mask):
            raise ValueError("All pixels are nodata! Check DEM file and nodata threshold.")
        
        valid_dem = dem[~nodata_mask]
        print(f"  Valid elevation range: [{np.min(valid_dem):.1f}, {np.max(valid_dem):.1f}] m")
        print(f"  Valid elevation mean: {np.mean(valid_dem):.1f} m")
        
        return dem, transform, crs, nodata_val
    
    def create_stable_mask(self, reference_shape, reference_transform, reference_crs):
        """
        Create boolean mask from stable ground shapefile.
        
        Parameters:
        -----------
        reference_shape : tuple
            Shape (rows, cols) of reference DEM
        reference_transform : affine.Affine
            Geotransform of reference DEM
        reference_crs : rasterio.crs.CRS
            CRS of reference DEM
        """
        print("\nCreating stable ground mask...")
        
        stable_gdf = gpd.read_file(self.stable_shapefile)
        print(f"  Loaded {len(stable_gdf)} stable area polygon(s)")
        
        # Reproject if necessary
        if stable_gdf.crs != reference_crs:
            print(f"  Reprojecting from {stable_gdf.crs} to {reference_crs}")
            stable_gdf = stable_gdf.to_crs(reference_crs)
        
        # Create raster mask
        self.stable_mask = geometry_mask(
            stable_gdf.geometry,
            transform=reference_transform,
            invert=True,
            out_shape=reference_shape
        )
        
        n_stable = np.sum(self.stable_mask)
        pct_stable = 100 * n_stable / self.stable_mask.size
        
        if n_stable == 0:
            raise ValueError("Stable mask is empty! Check shapefile overlap with DEM.")
        
        print(f"  Stable pixels: {n_stable} ({pct_stable:.1f}%)")
        
    def check_dem_alignment(self, dem1, dem2, output_dir):
        """
        Check if DEMs are well-aligned before differencing.
        
        Parameters:
        -----------
        dem1, dem2 : np.ndarray
            The two DEM arrays to compare
        output_dir : Path
            Directory to save diagnostic plots
        """
        print("\n--- DEM ALIGNMENT CHECK ---")
        
        # Calculate initial difference (before co-registration)
        valid_mask = np.isfinite(dem1) & np.isfinite(dem2)
        initial_diff = dem2 - dem1
        
        # Statistics on stable areas
        if self.stable_mask is not None:
            stable_valid = valid_mask & self.stable_mask
            stable_diff = initial_diff[stable_valid]
            
            print(f"  Stable area difference (before co-registration):")
            print(f"    Mean: {np.mean(stable_diff):.3f} m")
            print(f"    Std:  {np.std(stable_diff):.3f} m")
            print(f"    Median: {np.median(stable_diff):.3f} m")
            
            if abs(np.mean(stable_diff)) > 0.5:
                print(f"  ⚠️  WARNING: Large bias detected! Co-registration recommended.")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # DEM 1
        im0 = axes[0, 0].imshow(dem1, cmap='terrain', vmin=np.nanpercentile(dem1, 2), 
                                 vmax=np.nanpercentile(dem1, 98))
        axes[0, 0].set_title('DEM 1 (Earlier)')
        plt.colorbar(im0, ax=axes[0, 0], label='Elevation (m)')
        
        # DEM 2
        im1 = axes[0, 1].imshow(dem2, cmap='terrain', vmin=np.nanpercentile(dem2, 2), 
                                 vmax=np.nanpercentile(dem2, 98))
        axes[0, 1].set_title('DEM 2 (Later)')
        plt.colorbar(im1, ax=axes[0, 1], label='Elevation (m)')
        
        # Initial difference
        diff_abs_max = np.nanpercentile(np.abs(initial_diff), 95)
        im2 = axes[1, 0].imshow(initial_diff, cmap='RdBu_r', 
                                 vmin=-diff_abs_max, vmax=diff_abs_max)
        axes[1, 0].set_title('Initial Difference (DEM2 - DEM1)')
        plt.colorbar(im2, ax=axes[1, 0], label='Elevation change (m)')
        
        # Highlight stable areas
        if self.stable_mask is not None:
            stable_overlay = np.ma.masked_where(~self.stable_mask, 
                                                 np.ones_like(dem1))
            axes[1, 0].imshow(stable_overlay, cmap='Greens', alpha=0.3)
        
        # Histogram of differences
        axes[1, 1].hist(initial_diff[valid_mask], bins=100, alpha=0.7, 
                        label='All areas', edgecolor='black')
        if self.stable_mask is not None:
            axes[1, 1].hist(stable_diff, bins=50, alpha=0.7, 
                           label='Stable areas', edgecolor='black')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Elevation difference (m)')
        axes[1, 1].set_ylabel('Pixel count')
        axes[1, 1].set_title('Distribution of Elevation Differences')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(-diff_abs_max, diff_abs_max)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dem_alignment_check.png', dpi=150)
        print(f"  Saved: dem_alignment_check.png")
        plt.close()
    
    def calculate_vertical_displacement(self, dem1, dem2, method='nuth_kaab'):
        """
        Calculate vertical displacement with co-registration.
        
        Parameters:
        -----------
        dem1, dem2 : np.ndarray
            The "before" and "after" DEMs
        method : str
            Co-registration method: 'icp', 'nuth_kaab', or 'none'
            
        Returns:
        --------
        np.ndarray : Vertical displacement (dh) in meters
        """
        print("\n--- CALCULATING VERTICAL DISPLACEMENT ---")
        
        if self.stable_mask is None:
            raise ValueError("Must create stable mask before calculating displacement")
        
        # Create xdem DEM objects
        print(f"  Creating xdem DEM objects...")
        self.dem1_xdem = xdem.DEM.from_array(
            data=dem1,
            transform=self.transform,
            crs=self.crs,
            nodata=np.nan
        )
        
        self.dem2_xdem = xdem.DEM.from_array(
            data=dem2,
            transform=self.transform,
            crs=self.crs,
            nodata=np.nan
        )
        
        # Co-registration
        if method == 'none':
            print("  Skipping co-registration (user requested)")
            dem2_aligned = self.dem2_xdem
            
        elif method == 'icp':
            print("  Attempting ICP co-registration...")
            try:
                coreg = xdem.coreg.ICP()
                coreg.fit(
                    self.dem1_xdem,
                    self.dem2_xdem,
                    inlier_mask=self.stable_mask,
                    verbose=False
                )
                dem2_aligned = coreg.apply(self.dem2_xdem)
                print("  ✓ ICP successful")
            except Exception as e:
                print(f"  ✗ ICP failed: {e}")
                print("  Falling back to Nuth-Kaab...")
                method = 'nuth_kaab'
        
        if method == 'nuth_kaab':
            print("  Attempting Nuth-Kaab co-registration...")
            try:
                coreg = xdem.coreg.NuthKaab()
                coreg.fit(
                    self.dem1_xdem,
                    self.dem2_xdem,
                    inlier_mask=self.stable_mask,
                    verbose=False
                )
                dem2_aligned = coreg.apply(self.dem2_xdem)
                print("  ✓ Nuth-Kaab successful")
            except Exception as e:
                print(f"  ✗ Nuth-Kaab failed: {e}")
                print("  Proceeding without co-registration")
                dem2_aligned = self.dem2_xdem
        
        # Calculate difference
        print("  Computing elevation difference...")
        dh = dem2_aligned - self.dem1_xdem
        dh_array = dh.data.copy()
        
        # Check calibration quality on stable areas
        valid_mask = np.isfinite(dh_array)
        stable_valid = valid_mask & self.stable_mask
        
        if np.any(stable_valid):
            stable_dh = dh_array[stable_valid]
            print(f"\n  Co-registration quality (stable areas):")
            print(f"    Mean dh: {np.mean(stable_dh):.3f} m")
            print(f"    Std dh:  {np.std(stable_dh):.3f} m")
            # Calculate NMAD manually to avoid deprecation warning
            nmad = 1.4826 * np.median(np.abs(stable_dh - np.median(stable_dh)))
            print(f"    NMAD:    {nmad:.3f} m")
            
            if abs(np.mean(stable_dh)) > 0.2:
                print(f"  ⚠️  WARNING: Residual bias > 0.2m. Consider different co-registration method.")
        
        return dh_array
    
    def filter_outliers(self, dh, method='iqr', threshold=3.0):
        """
        Remove statistical outliers from vertical displacement.
        
        Parameters:
        -----------
        dh : np.ndarray
            Vertical displacement array
        method : str
            'iqr' for interquartile range or 'mad' for median absolute deviation
        threshold : float
            Multiplier for outlier detection (default: 3.0)
            
        Returns:
        --------
        np.ndarray : Filtered displacement array
        """
        print(f"\n  Filtering outliers (method={method}, threshold={threshold})...")
        
        dh_filtered = dh.copy()
        valid = np.isfinite(dh)
        
        if method == 'iqr':
            q1, q3 = np.nanpercentile(dh[valid], [25, 75])
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            
        elif method == 'mad':
            median = np.nanmedian(dh[valid])
            mad = np.nanmedian(np.abs(dh[valid] - median))
            lower = median - threshold * 1.4826 * mad  # 1.4826 converts MAD to std equivalent
            upper = median + threshold * 1.4826 * mad
        
        print(f"    Outlier bounds: [{lower:.2f}, {upper:.2f}] m")
        
        outliers = (dh < lower) | (dh > upper)
        n_outliers = np.sum(outliers & valid)
        pct_outliers = 100 * n_outliers / np.sum(valid)
        
        dh_filtered[outliers] = np.nan
        
        print(f"    Removed {n_outliers} outliers ({pct_outliers:.1f}% of valid pixels)")
        
        return dh_filtered
    
    def visualize_results(self, dh, output_dir, time_delta_years=1.0):
        """
        Create comprehensive visualization of vertical displacement results.
        
        Parameters:
        -----------
        dh : np.ndarray
            Vertical displacement in meters
        output_dir : Path
            Directory to save plots
        time_delta_years : float
            Time between DEMs in years (for velocity calculation)
        """
        print("\n--- CREATING RESULT VISUALIZATIONS ---")
        
        # Calculate velocity - ensure writable copy to avoid read-only errors
        velocity = np.array(dh / time_delta_years, dtype=np.float64)
        
        # Statistics
        valid = np.isfinite(dh)
        print(f"\n  Vertical displacement statistics:")
        print(f"    Valid pixels: {np.sum(valid)} ({100*np.sum(valid)/dh.size:.1f}%)")
        print(f"    Range: [{np.nanmin(dh):.2f}, {np.nanmax(dh):.2f}] m")
        print(f"    Mean: {np.nanmean(dh):.2f} m")
        print(f"    Median: {np.nanmedian(dh):.2f} m")
        print(f"    Std: {np.nanstd(dh):.2f} m")
        
        # Create visualizations as separate plots
        self._create_displacement_maps(dh, velocity, output_dir, time_delta_years)
        self._create_distribution_plots(dh, velocity, output_dir, time_delta_years)
        
        print(f"  Visualizations complete!")
    
    def _create_displacement_maps(self, dh, velocity, output_dir, time_delta_years):
        """
        Create maps of vertical displacement and velocity.
        
        Parameters:
        -----------
        dh : np.ndarray
            Vertical displacement in meters
        velocity : np.ndarray  
            Vertical velocity in m/year
        output_dir : Path
            Directory to save plots
        time_delta_years : float
            Time interval in years
        """
        print("  Creating displacement maps...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Vertical displacement map
        # Use writable copy for percentile calculation
        dh_copy = np.array(dh, dtype=np.float64)
        vmax = np.nanpercentile(np.abs(dh_copy[np.isfinite(dh_copy)]), 95)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im1 = axes[0].imshow(dh, cmap='RdBu_r', norm=norm)
        axes[0].set_title(f'Vertical Displacement (m)\n{time_delta_years:.1f} year interval', 
                      fontsize=12, fontweight='bold')
        
        # Overlay stable areas
        if self.stable_mask is not None:
            stable_overlay = np.ma.masked_where(~self.stable_mask, np.ones_like(dh))
            axes[0].contour(stable_overlay, levels=[0.5], colors='lime', linewidths=2, 
                       linestyles='--', alpha=0.8)
            axes[0].plot([], [], 'lime', linestyle='--', linewidth=2, label='Stable areas')
            axes[0].legend(loc='upper right')
        
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('Subsidence ← | → Uplift (m)', fontsize=10)
        axes[0].set_xlabel('Column (pixels)')
        axes[0].set_ylabel('Row (pixels)')
        
        # 2. Vertical velocity map
        vel_copy = np.array(velocity, dtype=np.float64)
        vmax_vel = np.nanpercentile(np.abs(vel_copy[np.isfinite(vel_copy)]), 95)
        norm_vel = TwoSlopeNorm(vmin=-vmax_vel, vcenter=0, vmax=vmax_vel)
        im2 = axes[1].imshow(velocity, cmap='RdBu_r', norm=norm_vel)
        axes[1].set_title(f'Vertical Velocity (m/year)', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label('Subsidence ← | → Uplift (m/yr)', fontsize=10)
        axes[1].set_xlabel('Column (pixels)')
        axes[1].set_ylabel('Row (pixels)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'vertical_displacement_maps.png', dpi=200, bbox_inches='tight')
        print(f"    Saved: vertical_displacement_maps.png")
        plt.close()
    
    def _create_distribution_plots(self, dh, velocity, output_dir, time_delta_years):
        """
        Create distribution and statistical plots.
        
        Parameters:
        -----------
        dh : np.ndarray
            Vertical displacement in meters
        velocity : np.ndarray
            Vertical velocity in m/year
        output_dir : Path
            Directory to save plots
        time_delta_years : float
            Time interval in years
        """
        print("  Creating distribution plots...")
        
        valid = np.isfinite(dh)
        dh_valid = dh[valid]
        vel_valid = velocity[valid]
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Histogram of displacement
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(dh_valid, bins=100, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero change')
        ax1.axvline(np.mean(dh_valid), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(dh_valid):.3f}m')
        ax1.set_xlabel('Vertical displacement (m)')
        ax1.set_ylabel('Pixel count')
        ax1.set_title('Distribution of Displacement')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram of velocity
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(vel_valid, bins=100, edgecolor='black', alpha=0.7, color='orange')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.axvline(np.mean(vel_valid), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(vel_valid):.3f}m/yr')
        ax2.set_xlabel('Vertical velocity (m/year)')
        ax2.set_ylabel('Pixel count')
        ax2.set_title('Distribution of Velocity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        ax3 = fig.add_subplot(gs[0, 2])
        sorted_dh = np.sort(dh_valid)
        cumulative = np.arange(1, len(sorted_dh) + 1) / len(sorted_dh) * 100
        ax3.plot(sorted_dh, cumulative, linewidth=2)
        ax3.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Vertical displacement (m)')
        ax3.set_ylabel('Cumulative percentage (%)')
        ax3.set_title('Cumulative Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot comparison
        ax4 = fig.add_subplot(gs[1, 0])
        if self.stable_mask is not None:
            stable_dh = dh[valid & self.stable_mask]
            moving_dh = dh[valid & ~self.stable_mask]
            
            data_to_plot = [stable_dh, moving_dh]
            tick_labels = ['Stable\nareas', 'Glacier\nareas']
            
            bp = ax4.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True,
                            showfliers=False)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            ax4.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax4.set_ylabel('Vertical displacement (m)')
            ax4.set_title('Stable vs. Moving Areas')
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'No stable mask\navailable', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Stable vs. Moving Areas')
        
        # 5. 2D histogram (displacement vs elevation if we had elevation)
        ax5 = fig.add_subplot(gs[1, 1])
        # Create 2D histogram of spatial distribution of displacement
        hist_2d, xedges, yedges = np.histogram2d(
            np.repeat(np.arange(dh.shape[0]), dh.shape[1])[valid.flatten()],
            np.tile(np.arange(dh.shape[1]), dh.shape[0])[valid.flatten()],
            bins=50,
            weights=dh.flatten()[valid.flatten()]
        )
        counts, _, _ = np.histogram2d(
            np.repeat(np.arange(dh.shape[0]), dh.shape[1])[valid.flatten()],
            np.tile(np.arange(dh.shape[1]), dh.shape[0])[valid.flatten()],
            bins=50
        )
        hist_2d = np.divide(hist_2d, counts, where=counts>0, out=np.zeros_like(hist_2d))
        
        im = ax5.imshow(hist_2d, origin='lower', aspect='auto', cmap='RdBu_r',
                       vmin=-0.5, vmax=0.5)
        ax5.set_xlabel('Column bin')
        ax5.set_ylabel('Row bin')
        ax5.set_title('Spatial Average of Displacement')
        plt.colorbar(im, ax=ax5, label='Mean dh (m)')
        
        # 6. Summary statistics table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        stats_text = f"""
SUMMARY STATISTICS
{'='*30}

Valid pixels: {np.sum(valid):,}
Coverage: {100*np.sum(valid)/dh.size:.1f}%

Displacement (m):
  Range: [{np.nanmin(dh):.3f}, {np.nanmax(dh):.3f}]
  Mean: {np.nanmean(dh):.3f}
  Median: {np.nanmedian(dh):.3f}
  Std: {np.nanstd(dh):.3f}
  
Velocity (m/yr):
  Range: [{np.nanmin(velocity):.3f}, {np.nanmax(velocity):.3f}]
  Mean: {np.nanmean(velocity):.3f}
  Median: {np.nanmedian(velocity):.3f}
        """
        
        if self.stable_mask is not None:
            stable_valid = valid & self.stable_mask
            if np.any(stable_valid):
                stats_text += f"""
Stable areas:
  Mean dh: {np.nanmean(dh[stable_valid]):.3f} m
  Std dh: {np.nanstd(dh[stable_valid]):.3f} m
  N pixels: {np.sum(stable_valid):,}
            """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 7. Time series visualization (placeholder for single epoch)
        ax7 = fig.add_subplot(gs[2, :])
        # Show profile across the glacier
        center_col = dh.shape[1] // 2
        profile_dh = dh[:, center_col]
        profile_rows = np.arange(len(profile_dh))
        
        ax7.plot(profile_rows[np.isfinite(profile_dh)], 
                profile_dh[np.isfinite(profile_dh)], 
                'o-', markersize=2, linewidth=1, label='Center profile')
        ax7.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax7.set_xlabel('Row (pixels)')
        ax7.set_ylabel('Vertical displacement (m)')
        ax7.set_title(f'Vertical Displacement Profile (Column {center_col})')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # Adjust layout manually to avoid tight_layout warning
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.35, wspace=0.35)
        plt.savefig(output_dir / 'vertical_displacement_distributions.png', dpi=200, bbox_inches='tight')
        print(f"    Saved: vertical_displacement_distributions.png")
        plt.close()



# ======================
# EXAMPLE USAGE
# ======================
if __name__ == "__main__":
    
    # --- User Inputs ---
    dem1_file = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_LIDAR/2018_0p5m_upper_rg_dem_larger_roi.tif")
    dem2_file = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2023_lidar/2023_0p5m_4_imcorr_upper_rg_larger_roi.tif")
    stable_file = Path(r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\shapefiles\stable_2.shp")
    
    # Analysis parameters
    TIME_DELTA_YEARS = 2022.0 - 2018.0  # Adjust to your actual dates
    COREG_METHOD = 'nuth_kaab'  # Options: 'icp', 'nuth_kaab', 'none'
    OUTLIER_METHOD = 'iqr'  # Options: 'iqr', 'mad'
    OUTLIER_THRESHOLD = 3.0
    
    try:
        # Create output directory
        output_dir = Path("results_vertical")
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}\n")
        
        # Initialize analyzer
        print("="*60)
        print("VERTICAL DISPLACEMENT ANALYSIS")
        print("="*60)
        analyzer = VerticalDisplacementAnalyzer(stable_area_shapefile=stable_file)
        
        # Load DEMs
        print("\n--- LOADING DEMs ---")
        dem1, transform, crs, _ = analyzer.load_dem(dem1_file, nodata_threshold=-1e10)
        dem2, _, _, _ = analyzer.load_dem(dem2_file, nodata_threshold=-1e10)
        
        # Store transform and CRS for later use
        analyzer.transform = transform
        analyzer.crs = crs
        
        # Create stable mask
        analyzer.create_stable_mask(dem1.shape, transform, crs)
        
        # Check DEM alignment
        analyzer.check_dem_alignment(dem1, dem2, output_dir)
        
        # Calculate vertical displacement
        dh = analyzer.calculate_vertical_displacement(dem1, dem2, method=COREG_METHOD)
        
        # Filter outliers
        dh_filtered = analyzer.filter_outliers(dh, method=OUTLIER_METHOD, 
                                               threshold=OUTLIER_THRESHOLD)
        
        # Visualize results
        analyzer.visualize_results(dh_filtered, output_dir, 
                                   time_delta_years=TIME_DELTA_YEARS)
        
        # Save results
        output_file = output_dir / "vertical_displacement.npz"
        np.savez_compressed(
            output_file,
            dh_m=dh_filtered,
            velocity_m_per_year=dh_filtered / TIME_DELTA_YEARS,
            stable_mask=analyzer.stable_mask,
            transform=transform,
            metadata={
                'time_delta_years': TIME_DELTA_YEARS,
                'coreg_method': COREG_METHOD,
                'outlier_method': OUTLIER_METHOD,
                'outlier_threshold': OUTLIER_THRESHOLD
            }
        )
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please verify all file paths are correct and files exist.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()