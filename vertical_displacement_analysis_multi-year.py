"""
Multi-Temporal Vertical Displacement Analysis for Rock Glaciers
================================================================

Enhanced version supporting multiple DEMs for time series analysis.
Includes snow detection, improved stable area visualization, and
comprehensive temporal comparisons.

Key features:
- Multiple DEM epochs (e.g., 2018, 2022, 2023, 2024)
- Pairwise and cumulative displacement tracking
- Snow/seasonal change detection and masking
- Enhanced stable area visualization
- Time series plots and trend analysis
"""

import numpy as np
import rasterio
import geopandas as gpd
import xdem
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
    Analyzes vertical displacement across multiple time periods with
    enhanced snow detection and visualization.
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
        
        self.epochs = OrderedDict()  # Store DEMs by epoch name
        self.dates = OrderedDict()   # Store acquisition dates
        self.stable_mask = None
        self.transform = None
        self.crs = None
        self.snow_masks = {}  # Store snow masks by epoch pair
        
    def add_epoch(self, epoch_name, dem_path, date=None, nodata_threshold=-1e10):
        """
        Add a DEM epoch to the analysis.
        
        Parameters:
        -----------
        epoch_name : str
            Name for this epoch (e.g., '2018', '2022', '2023')
        dem_path : str or Path
            Path to DEM raster file
        date : str or datetime, optional
            Acquisition date (e.g., '2018-09-15')
        nodata_threshold : float
            Values below this are considered nodata
        """
        print(f"\nAdding epoch: {epoch_name}")
        print(f"  Loading: {Path(dem_path).name}")
        
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float64)
            transform = src.transform
            crs = src.crs
            nodata_val = src.nodata
            
            print(f"  Shape: {dem.shape}")
            print(f"  Raw range: [{np.min(dem):.2e}, {np.max(dem):.2e}]")
        
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
        
        # Store the epoch
        self.epochs[epoch_name] = {
            'dem': dem,
            'nodata_mask': nodata_mask
        }
        
        # Store date if provided
        if date is not None:
            if isinstance(date, str):
                self.dates[epoch_name] = datetime.strptime(date, '%Y-%m-%d')
            else:
                self.dates[epoch_name] = date
        
        # Store transform and CRS from first epoch
        if self.transform is None:
            self.transform = transform
            self.crs = crs
        else:
            # Check consistency
            if transform != self.transform:
                print(f"  ⚠️ WARNING: Transform differs from first epoch")
            if crs != self.crs:
                print(f"  ⚠️ WARNING: CRS differs from first epoch")
    
    def create_stable_mask(self):
        """
        Create boolean mask from stable ground shapefile using first epoch's grid.
        """
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
        """
        Detect potential snow cover changes based on elevation difference patterns.
        
        Snow accumulation/melt creates characteristic patterns:
        - Large positive changes (accumulation)
        - Large negative changes (melt)
        - Spatially coherent patches
        
        Parameters:
        -----------
        dh : np.ndarray
            Elevation difference array
        threshold : float
            Threshold for flagging as potential snow (default: 0.3m)
            
        Returns:
        --------
        np.ndarray : Boolean mask where True = potential snow change
        """
        from scipy import ndimage
        
        # Identify large elevation changes
        large_changes = np.abs(dh) > threshold
        
        # Remove isolated pixels (keep spatially coherent patches)
        # Snow patches are typically >10 pixels
        structure = ndimage.generate_binary_structure(2, 2)
        large_changes_filtered = ndimage.binary_opening(
            large_changes, 
            structure=structure, 
            iterations=2
        )
        
        # Dilate slightly to capture edges
        snow_mask = ndimage.binary_dilation(
            large_changes_filtered,
            structure=structure,
            iterations=1
        )
        
        return snow_mask
    
    def calculate_pairwise_displacement(self, epoch1_name, epoch2_name, 
                                       coreg_method='nuth_kaab',
                                       filter_outliers=True,
                                       detect_snow=True):
        """
        Calculate displacement between two epochs.
        
        Parameters:
        -----------
        epoch1_name, epoch2_name : str
            Names of epochs to compare
        coreg_method : str
            Co-registration method ('icp', 'nuth_kaab', or 'none')
        filter_outliers : bool
            Whether to filter statistical outliers
        detect_snow : bool
            Whether to detect and mask potential snow changes
            
        Returns:
        --------
        dict : Results including dh, velocity, masks, and quality metrics
        """
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
            # Try to extract year from epoch name
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
        elif coreg_method == 'nuth_kaab':
            try:
                coreg = xdem.coreg.NuthKaab()
                coreg.fit(dem1_xdem, dem2_xdem, inlier_mask=self.stable_mask, verbose=False)
                dem2_aligned = coreg.apply(dem2_xdem)
                print(f"  ✓ Nuth-Kaab successful")
            except Exception as e:
                print(f"  ✗ Failed: {e}, skipping co-registration")
                dem2_aligned = dem2_xdem
        elif coreg_method == 'icp':
            try:
                coreg = xdem.coreg.ICP()
                coreg.fit(dem1_xdem, dem2_xdem, inlier_mask=self.stable_mask, verbose=False)
                dem2_aligned = coreg.apply(dem2_xdem)
                print(f"  ✓ ICP successful")
            except Exception as e:
                print(f"  ✗ Failed: {e}, trying Nuth-Kaab instead")
                coreg_method = 'nuth_kaab'
                coreg = xdem.coreg.NuthKaab()
                coreg.fit(dem1_xdem, dem2_xdem, inlier_mask=self.stable_mask, verbose=False)
                dem2_aligned = coreg.apply(dem2_xdem)
        
        # Calculate difference
        dh = dem2_aligned - dem1_xdem
        dh_array = dh.data.copy()
        
        # Quality metrics on stable areas
        valid = np.isfinite(dh_array)
        stable_valid = valid & self.stable_mask
        
        if np.any(stable_valid):
            stable_dh = dh_array[stable_valid]
            mean_bias = np.mean(stable_dh)
            std_bias = np.std(stable_dh)
            nmad = 1.4826 * np.median(np.abs(stable_dh - np.median(stable_dh)))
            
            print(f"  Stable area quality:")
            print(f"    Mean: {mean_bias:.3f} m")
            print(f"    Std:  {std_bias:.3f} m")
            print(f"    NMAD: {nmad:.3f} m")
        else:
            mean_bias = std_bias = nmad = np.nan
        
        # Detect potential snow changes
        snow_mask = None
        if detect_snow:
            snow_mask = self.detect_snow_change(dh_array, threshold=0.3)
            n_snow = np.sum(snow_mask & valid)
            pct_snow = 100 * n_snow / np.sum(valid) if np.any(valid) else 0
            print(f"  Potential snow change: {n_snow} pixels ({pct_snow:.1f}%)")
            
            # Store for later use
            pair_key = f"{epoch1_name}_{epoch2_name}"
            self.snow_masks[pair_key] = snow_mask
        
        # Filter outliers
        dh_filtered = dh_array.copy()
        outlier_mask = np.zeros_like(dh_array, dtype=bool)
        
        if filter_outliers:
            q1, q3 = np.nanpercentile(dh_array[valid], [25, 75])
            iqr = q3 - q1
            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr
            
            outlier_mask = (dh_array < lower) | (dh_array > upper)
            n_outliers = np.sum(outlier_mask & valid)
            pct_outliers = 100 * n_outliers / np.sum(valid) if np.any(valid) else 0
            
            dh_filtered[outlier_mask] = np.nan
            print(f"  Outliers removed: {n_outliers} ({pct_outliers:.1f}%)")
        
        # Calculate velocity
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
                'coreg_method': coreg_method
            }
        }
    
    def calculate_all_pairwise(self, coreg_method='nuth_kaab', 
                               filter_outliers=True, detect_snow=True):
        """
        Calculate displacement for all consecutive epoch pairs.
        
        Returns:
        --------
        dict : Results for each epoch pair
        """
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
                detect_snow=detect_snow
            )
        
        return results
    
    def visualize_pairwise_comparison(self, results, output_dir, 
                                     show_snow_mask=True):
        """
        Create comprehensive visualization of pairwise displacements.
        
        Parameters:
        -----------
        results : dict
            Results from calculate_all_pairwise()
        output_dir : Path
            Directory to save plots
        show_snow_mask : bool
            Whether to overlay detected snow changes
        """
        print("\n--- CREATING PAIRWISE COMPARISON PLOTS ---")
        
        n_pairs = len(results)
        
        # Create figure with subplots for each pair
        fig, axes = plt.subplots(2, n_pairs, figsize=(6*n_pairs, 12))
        if n_pairs == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (pair_key, result) in enumerate(results.items()):
            dh = result['dh']
            velocity = result['velocity']
            dt = result['time_delta']
            
            # Use writable copy for percentile calculation
            dh_copy = np.array(dh, dtype=np.float64)
            valid_dh = dh_copy[np.isfinite(dh_copy)]
            
            if len(valid_dh) > 0:
                vmax = np.percentile(np.abs(valid_dh), 95)
            else:
                vmax = 0.5
            
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            
            # Top row: Displacement
            im1 = axes[0, idx].imshow(dh, cmap='RdBu_r', norm=norm)
            axes[0, idx].set_title(f'{pair_key.replace("_", " → ")}\n'
                                  f'Displacement (m), Δt={dt:.2f}yr',
                                  fontsize=11, fontweight='bold')
            
            # Overlay stable areas with THICK, VISIBLE lines
            if self.stable_mask is not None:
                # Draw white outline first for contrast
                axes[0, idx].contour(
                    self.stable_mask, 
                    levels=[0.5], 
                    colors='white', 
                    linewidths=5,
                    linestyles='-',
                    alpha=0.8
                )
                # Draw lime green line on top
                axes[0, idx].contour(
                    self.stable_mask, 
                    levels=[0.5], 
                    colors='lime', 
                    linewidths=3,
                    linestyles='-',
                    alpha=1.0
                )
            
            # Overlay snow mask if requested
            if show_snow_mask and result['snow_mask'] is not None:
                snow_overlay = np.ma.masked_where(
                    ~result['snow_mask'], 
                    np.ones_like(dh)
                )
                axes[0, idx].contour(
                    snow_overlay, 
                    levels=[0.5], 
                    colors='cyan', 
                    linewidths=2,
                    linestyles='--',
                    alpha=0.8
                )
            
            plt.colorbar(im1, ax=axes[0, idx], label='Subsidence ← | → Uplift (m)')
            axes[0, idx].set_xlabel('Column (pixels)')
            axes[0, idx].set_ylabel('Row (pixels)')
            
            # Bottom row: Velocity
            vel_copy = np.array(velocity, dtype=np.float64)
            valid_vel = vel_copy[np.isfinite(vel_copy)]
            
            if len(valid_vel) > 0:
                vmax_vel = np.percentile(np.abs(valid_vel), 95)
            else:
                vmax_vel = 0.1
            
            norm_vel = TwoSlopeNorm(vmin=-vmax_vel, vcenter=0, vmax=vmax_vel)
            im2 = axes[1, idx].imshow(velocity, cmap='RdBu_r', norm=norm_vel)
            axes[1, idx].set_title(f'Velocity (m/year)', fontsize=11, fontweight='bold')
            plt.colorbar(im2, ax=axes[1, idx], label='Subsidence ← | → Uplift (m/yr)')
            axes[1, idx].set_xlabel('Column (pixels)')
            axes[1, idx].set_ylabel('Row (pixels)')
        
        # Add legend
        legend_elements = [
            Patch(facecolor='lime', edgecolor='white', linewidth=2, label='Stable areas')
        ]
        if show_snow_mask:
            legend_elements.append(
                Patch(facecolor='cyan', edgecolor='cyan', linewidth=2, 
                      label='Potential snow', linestyle='--')
            )
        
        fig.legend(handles=legend_elements, loc='upper center', 
                  ncol=len(legend_elements), fontsize=11, 
                  bbox_to_anchor=(0.5, 0.98))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / 'pairwise_displacements.png', dpi=200, bbox_inches='tight')
        print(f"  Saved: pairwise_displacements.png")
        plt.close()
    
    def create_time_series_plot(self, results, output_dir):
        """
        Create time series plot showing displacement evolution.
        
        Parameters:
        -----------
        results : dict
            Results from calculate_all_pairwise()
        output_dir : Path
            Directory to save plot
        """
        print("  Creating time series plot...")
        
        # Extract statistics for each pair
        pairs = []
        mean_dh = []
        median_dh = []
        std_dh = []
        stable_mean = []
        stable_std = []
        time_intervals = []
        
        for pair_key, result in results.items():
            pairs.append(pair_key.replace('_', ' → '))
            
            valid = np.isfinite(result['dh'])
            dh_valid = result['dh'][valid]
            
            mean_dh.append(np.mean(dh_valid))
            median_dh.append(np.median(dh_valid))
            std_dh.append(np.std(dh_valid))
            
            stable_mean.append(result['quality']['mean_bias'])
            stable_std.append(result['quality']['std_bias'])
            time_intervals.append(result['time_delta'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x = np.arange(len(pairs))
        
        # 1. Mean displacement
        axes[0, 0].bar(x, mean_dh, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Epoch Pair')
        axes[0, 0].set_ylabel('Mean Displacement (m)')
        axes[0, 0].set_title('Mean Vertical Displacement by Period', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(pairs, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Median with error bars
        axes[0, 1].errorbar(x, median_dh, yerr=std_dh, fmt='o-', 
                           linewidth=2, markersize=8, capsize=5,
                           color='darkgreen', label='Median ± Std')
        axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Epoch Pair')
        axes[0, 1].set_ylabel('Displacement (m)')
        axes[0, 1].set_title('Median Displacement with Variability', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(pairs, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Stable area quality
        axes[1, 0].bar(x, stable_mean, alpha=0.7, color='lightgreen', 
                      edgecolor='black', label='Stable area mean')
        axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].axhline(0.2, color='orange', linestyle=':', linewidth=1, alpha=0.7)
        axes[1, 0].axhline(-0.2, color='orange', linestyle=':', linewidth=1, alpha=0.7)
        axes[1, 0].set_xlabel('Epoch Pair')
        axes[1, 0].set_ylabel('Mean Displacement (m)')
        axes[1, 0].set_title('Stable Area Co-registration Quality', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(pairs, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].text(0.02, 0.98, 'Target: |mean| < 0.2m', 
                       transform=axes[1, 0].transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 4. Time intervals and velocities
        velocities = [m/dt if dt > 0 else 0 for m, dt in zip(median_dh, time_intervals)]
        
        ax4a = axes[1, 1]
        ax4b = ax4a.twinx()
        
        bars = ax4a.bar(x, time_intervals, alpha=0.5, color='lightcoral', 
                       edgecolor='black', label='Time interval (years)')
        line = ax4b.plot(x, velocities, 'o-', linewidth=2, markersize=8, 
                        color='darkblue', label='Velocity (m/yr)')
        ax4b.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        ax4a.set_xlabel('Epoch Pair')
        ax4a.set_ylabel('Time Interval (years)', color='darkred')
        ax4b.set_ylabel('Velocity (m/year)', color='darkblue')
        ax4a.set_title('Time Intervals and Velocities', fontweight='bold')
        ax4a.set_xticks(x)
        ax4a.set_xticklabels(pairs, rotation=45, ha='right')
        ax4a.tick_params(axis='y', labelcolor='darkred')
        ax4b.tick_params(axis='y', labelcolor='darkblue')
        ax4a.grid(True, alpha=0.3, axis='x')
        
        # Combined legend
        lines1, labels1 = ax4a.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4a.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'time_series_summary.png', dpi=200, bbox_inches='tight')
        print(f"  Saved: time_series_summary.png")
        plt.close()


# ======================
# EXAMPLE USAGE
# ======================
if __name__ == "__main__":
    
    # --- User Inputs ---
    stable_file = Path(r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\shapefiles\stable_2.shp")
    
    # Define your DEM epochs
    # Format: (epoch_name, dem_path, date)
    dem_epochs = [
        ('2018', r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\preprocessed_dems\2018_0p5m_upper_rg_dem_larger_roi_harmonized.tif", '2018-09-01'),
        ('2023', r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\preprocessed_dems\2023_0p5m_4_imcorr_upper_rg_larger_roi_harmonized.tif", '2023-09-01'),
        # Add your 2023 and 2024 DEMs here:
        ('2024', r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\Code\preprocessed_dems\2024_0p5_4_imcorr__upper_rg_larger_roi_harmonized.tif", '2024-09-01'),
        # ('2024', r"path/to/2024_dem.tif", '2024-09-01'),
    ]
    
    # Analysis parameters
    COREG_METHOD = 'nuth_kaab'  # Options: 'icp', 'nuth_kaab', 'none'
    FILTER_OUTLIERS = True
    DETECT_SNOW = True  # Detect and flag potential snow changes
    
    try:
        # Create output directory
        output_dir = Path("results_multitemporal")
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}\n")
        
        # Initialize analyzer
        print("="*60)
        print("MULTI-TEMPORAL VERTICAL DISPLACEMENT ANALYSIS")
        print("="*60)
        analyzer = MultiTemporalDisplacementAnalyzer(stable_area_shapefile=stable_file)
        
        # Add all epochs
        for epoch_name, dem_path, date in dem_epochs:
            analyzer.add_epoch(epoch_name, dem_path, date)
        
        # Create stable mask
        analyzer.create_stable_mask()
        
        # Calculate all pairwise displacements
        results = analyzer.calculate_all_pairwise(
            coreg_method=COREG_METHOD,
            filter_outliers=FILTER_OUTLIERS,
            detect_snow=DETECT_SNOW
        )
        
        # Create visualizations
        analyzer.visualize_pairwise_comparison(results, output_dir, show_snow_mask=DETECT_SNOW)
        analyzer.create_time_series_plot(results, output_dir)
        
        # Save results
        output_file = output_dir / "multitemporal_results.npz"
        
        # Prepare data for saving
        save_data = {
            'stable_mask': analyzer.stable_mask,
            'transform': analyzer.transform,
            'epoch_names': list(analyzer.epochs.keys())
        }
        
        # Add pairwise results
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
        print(f"\nProcessed {len(dem_epochs)} epochs:")
        for epoch_name, _, _ in dem_epochs:
            print(f"  - {epoch_name}")
        print(f"\nGenerated {len(results)} pairwise comparisons")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please verify all file paths are correct and files exist.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()