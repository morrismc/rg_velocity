"""
Rock Glacier Displacement Visualization Script
==============================================

This script loads pre-computed displacement results and creates
publication-quality visualizations for both 2D spatial patterns
and 1D statistical analyses.

Usage:
------
1. Run the main processing script first to generate results
2. Adjust the file paths below to match your data
3. Run this script to generate visualizations
4. Toggle visualization sections on/off as needed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
from pathlib import Path
import rasterio
from scipy.interpolate import griddata
from scipy import stats

# Set publication-quality plot defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

class DisplacementVisualizer:
    """
    Class to handle all visualization tasks for rock glacier displacement analysis.
    """
    
    def __init__(self, results_file, dem1_file, hs1_file, stable_shapefile=None):
        """
        Initialize the visualizer by loading all necessary data.
        
        Parameters:
        -----------
        results_file : str or Path
            Path to the .npz results file from the processing script
        dem1_file : str or Path
            Path to the reference DEM (for context)
        hs1_file : str or Path
            Path to the reference hillshade (for context)
        stable_shapefile : str or Path, optional
            Path to stable areas shapefile (for overlay)
        """
        print("="*70)
        print("LOADING DATA FOR VISUALIZATION")
        print("="*70)
        
        # Load displacement results
        print(f"Loading results from: {results_file}")
        self.results = np.load(results_file)
        
        # Extract key arrays
        self.piv_x = self.results['x_grid_px']
        self.piv_y = self.results['y_grid_px']
        self.dx_m = self.results['dx_m']
        self.dy_m = self.results['dy_m']
        self.dz_m = self.results['dz_m']
        self.vel_2d = self.results['vel_2d_ma']
        self.vel_3d = self.results['vel_3d_ma']
        self.vel_z = self.results['vel_z_ma']
        self.azimuth = self.results['azimuth_deg']
        
        # Load reference raster for context
        print(f"Loading DEM from: {dem1_file}")
        with rasterio.open(dem1_file) as src:
            self.dem = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            
        print(f"Loading hillshade from: {hs1_file}")
        with rasterio.open(hs1_file) as src:
            self.hillshade = src.read(1)
            
        # Load stable mask if provided
        self.stable_mask = None
        if stable_shapefile is not None:
            print(f"Loading stable areas from: {stable_shapefile}")
            import geopandas as gpd
            from rasterio.features import geometry_mask
            
            stable_gdf = gpd.read_file(stable_shapefile)
            if stable_gdf.crs != self.crs:
                stable_gdf = stable_gdf.to_crs(self.crs)
            
            self.stable_mask = geometry_mask(
                stable_gdf.geometry,
                transform=self.transform,
                invert=True,
                out_shape=self.dem.shape
            )
        
        # Compute derived quantities
        self._compute_statistics()
        
        print("\n✓ Data loaded successfully")
        print(f"  DEM shape: {self.dem.shape}")
        print(f"  PIV grid: {self.piv_x.shape}")
        print(f"  Valid velocity vectors: {np.sum(~np.isnan(self.vel_2d))}")
        
    def _compute_statistics(self):
        """Compute summary statistics for later use."""
        # Flatten and remove NaNs
        self.vel_2d_valid = self.vel_2d[~np.isnan(self.vel_2d)]
        self.vel_3d_valid = self.vel_3d[~np.isnan(self.vel_3d)]
        self.vel_z_valid = self.vel_z[~np.isnan(self.vel_z)]
        
        # Get flattened coordinates for valid vectors
        vel_2d_flat = self.vel_2d.flatten()
        valid_mask = ~np.isnan(vel_2d_flat)
        self.piv_x_valid = self.piv_x.flatten()[valid_mask]
        self.piv_y_valid = self.piv_y.flatten()[valid_mask]
        
    def create_interpolated_velocity_map(self):
        """
        Interpolate PIV grid velocities to full DEM resolution for smooth visualization.
        """
        print("\nInterpolating velocity field to DEM grid...")
        
        # Create dense grid matching DEM
        grid_x, grid_y = np.meshgrid(
            np.arange(self.dem.shape[1]),
            np.arange(self.dem.shape[0])
        )
        
        # Interpolate using cubic for smoothness
        vel_2d_interp = griddata(
            (self.piv_x_valid, self.piv_y_valid),
            self.vel_2d_valid,
            (grid_x, grid_y),
            method='cubic',
            fill_value=np.nan
        )
        
        return vel_2d_interp
    
    # =========================================================================
    # 2D SPATIAL VISUALIZATIONS
    # =========================================================================
    
    def plot_velocity_overview(self, save_path=None, show=True):
        """
        Create a comprehensive 6-panel overview of displacement results.
        """
        print("\n" + "="*70)
        print("CREATING VELOCITY OVERVIEW (6 panels)")
        print("="*70)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Panel 1: Hillshade with stable areas
        ax = axes[0, 0]
        ax.imshow(self.hillshade, cmap='gray', alpha=0.8)
        if self.stable_mask is not None:
            stable_overlay = np.ma.masked_where(~self.stable_mask, self.stable_mask)
            ax.imshow(stable_overlay, cmap='Reds', alpha=0.4)
        ax.set_title('Reference Hillshade\n(Red = Stable Areas)', fontsize=11)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Panel 2: PIV grid points colored by velocity
        ax = axes[0, 1]
        ax.imshow(self.hillshade, cmap='gray', alpha=0.7)
        scatter = ax.scatter(
            self.piv_x_valid, self.piv_y_valid, 
            c=self.vel_2d_valid, 
            cmap='plasma', 
            s=30, 
            edgecolors='black', 
            linewidths=0.5,
            vmin=0, 
            vmax=np.percentile(self.vel_2d_valid, 95)
        )
        plt.colorbar(scatter, ax=ax, label='Velocity (m/yr)', shrink=0.8)
        ax.set_title(f'PIV Vectors (n={len(self.piv_x_valid)})\nColored by 2D Velocity', fontsize=11)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Panel 3: 2D Horizontal Velocity (Raw PIV grid)
        ax = axes[0, 2]
        vel_plot = ax.imshow(
            self.vel_2d, 
            cmap='plasma', 
            vmin=0, 
            vmax=np.percentile(self.vel_2d_valid, 95)
        )
        plt.colorbar(vel_plot, ax=ax, label='Velocity (m/yr)', shrink=0.8)
        ax.set_title(f'2D Horizontal Velocity (Raw)\nMax: {np.nanmax(self.vel_2d):.2f} m/yr', fontsize=11)
        ax.set_xlabel('PIV Grid X')
        ax.set_ylabel('PIV Grid Y')
        
        # Panel 4: 2D Velocity (Interpolated to DEM grid)
        ax = axes[1, 0]
        vel_interp = self.create_interpolated_velocity_map()
        vel_plot = ax.imshow(
            vel_interp, 
            cmap='plasma', 
            vmin=0, 
            vmax=np.percentile(self.vel_2d_valid, 95)
        )
        plt.colorbar(vel_plot, ax=ax, label='Velocity (m/yr)', shrink=0.8)
        ax.set_title('2D Horizontal Velocity (Interpolated)\nSmooth Visualization', fontsize=11)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Panel 5: Vertical velocity
        ax = axes[1, 1]
        vel_z_plot = ax.imshow(
            self.vel_z,
            cmap='RdBu_r',
            vmin=np.percentile(self.vel_z_valid, 5),
            vmax=np.percentile(self.vel_z_valid, 95)
        )
        plt.colorbar(vel_z_plot, ax=ax, label='Vertical Vel. (m/yr)', shrink=0.8)
        ax.set_title('Vertical Velocity\n(Red=Subsidence, Blue=Uplift)', fontsize=11)
        ax.set_xlabel('PIV Grid X')
        ax.set_ylabel('PIV Grid Y')
        
        # Panel 6: Flow direction (azimuth)
        ax = axes[1, 2]
        # Create circular colormap for azimuth
        azimuth_plot = ax.imshow(
            self.azimuth,
            cmap='hsv',
            vmin=0,
            vmax=360
        )
        cbar = plt.colorbar(azimuth_plot, ax=ax, label='Azimuth (°)', shrink=0.8)
        cbar.set_ticks([0, 90, 180, 270, 360])
        cbar.set_ticklabels(['N', 'E', 'S', 'W', 'N'])
        ax.set_title('Flow Direction\n(North = 0°, Clockwise)', fontsize=11)
        ax.set_xlabel('PIV Grid X')
        ax.set_ylabel('PIV Grid Y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_velocity_vectors(self, subsample=2, scale=20, save_path=None, show=True):
        """
        Plot velocity vectors as arrows overlaid on hillshade.
        
        Parameters:
        -----------
        subsample : int
            Plot every Nth vector (to avoid clutter)
        scale : float
            Arrow scaling factor
        """
        print("\n" + "="*70)
        print("CREATING VECTOR FIELD PLOT")
        print("="*70)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Background hillshade
        ax.imshow(self.hillshade, cmap='gray', alpha=0.7)
        
        # Subsample the grid
        x_sub = self.piv_x[::subsample, ::subsample]
        y_sub = self.piv_y[::subsample, ::subsample]
        dx_sub = self.dx_m[::subsample, ::subsample]
        dy_sub = self.dy_m[::subsample, ::subsample]
        vel_sub = self.vel_2d[::subsample, ::subsample]
        
        # Plot vectors
        quiver = ax.quiver(
            x_sub, y_sub,
            dx_sub, dy_sub,
            vel_sub,
            cmap='plasma',
            scale=scale,
            scale_units='xy',
            width=0.003,
            headwidth=3,
            headlength=4,
            alpha=0.8
        )
        
        cbar = plt.colorbar(quiver, ax=ax, label='Velocity (m/yr)', shrink=0.7)
        
        ax.set_title(f'Displacement Vectors\n(Subsample: 1/{subsample}, Scale: {scale})', fontsize=13)
        ax.set_xlabel('X (pixels)', fontsize=11)
        ax.set_ylabel('Y (pixels)', fontsize=11)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_velocity_hillshade_overlay(self, alpha=0.6, save_path=None, show=True):
        """
        Create a velocity map with semi-transparent hillshade overlay.
        """
        print("\n" + "="*70)
        print("CREATING HILLSHADE OVERLAY")
        print("="*70)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Interpolated velocity as base
        vel_interp = self.create_interpolated_velocity_map()
        
        # Plot velocity
        vel_plot = ax.imshow(
            vel_interp,
            cmap='plasma',
            vmin=0,
            vmax=np.percentile(self.vel_2d_valid, 95),
            alpha=1.0
        )
        
        # Overlay hillshade
        ax.imshow(self.hillshade, cmap='gray', alpha=alpha)
        
        plt.colorbar(vel_plot, ax=ax, label='2D Velocity (m/yr)', shrink=0.7)
        ax.set_title('2D Velocity with Hillshade Overlay', fontsize=13)
        ax.set_xlabel('X (pixels)', fontsize=11)
        ax.set_ylabel('Y (pixels)', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_3d_velocity_comparison(self, save_path=None, show=True):
        """
        Compare 2D vs 3D velocity side-by-side.
        """
        print("\n" + "="*70)
        print("CREATING 2D vs 3D COMPARISON")
        print("="*70)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 2D velocity
        ax = axes[0]
        vel_plot = ax.imshow(
            self.vel_2d,
            cmap='plasma',
            vmin=0,
            vmax=np.percentile(self.vel_2d_valid, 95)
        )
        plt.colorbar(vel_plot, ax=ax, label='Velocity (m/yr)')
        ax.set_title(f'2D Horizontal Velocity\nMean: {np.mean(self.vel_2d_valid):.2f} m/yr')
        ax.set_xlabel('PIV Grid X')
        ax.set_ylabel('PIV Grid Y')
        
        # 3D velocity
        ax = axes[1]
        vel_plot = ax.imshow(
            self.vel_3d,
            cmap='plasma',
            vmin=0,
            vmax=np.percentile(self.vel_3d_valid, 95)
        )
        plt.colorbar(vel_plot, ax=ax, label='Velocity (m/yr)')
        ax.set_title(f'3D Total Velocity\nMean: {np.mean(self.vel_3d_valid):.2f} m/yr')
        ax.set_xlabel('PIV Grid X')
        ax.set_ylabel('PIV Grid Y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # =========================================================================
    # 1D STATISTICAL VISUALIZATIONS
    # =========================================================================
    
    def plot_velocity_histograms(self, save_path=None, show=True):
        """
        Create histograms for 2D, 3D, and vertical velocities.
        """
        print("\n" + "="*70)
        print("CREATING VELOCITY HISTOGRAMS")
        print("="*70)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 2D velocity histogram
        ax = axes[0]
        vel_clip = self.vel_2d_valid[self.vel_2d_valid < np.percentile(self.vel_2d_valid, 99)]
        ax.hist(vel_clip, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.median(self.vel_2d_valid), color='red', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(self.vel_2d_valid):.2f}')
        ax.axvline(np.mean(self.vel_2d_valid), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(self.vel_2d_valid):.2f}')
        ax.set_xlabel('2D Velocity (m/yr)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'2D Horizontal Velocity\nn = {len(self.vel_2d_valid)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3D velocity histogram
        ax = axes[1]
        vel_clip = self.vel_3d_valid[self.vel_3d_valid < np.percentile(self.vel_3d_valid, 99)]
        ax.hist(vel_clip, bins=40, edgecolor='black', alpha=0.7, color='darkorange')
        ax.axvline(np.median(self.vel_3d_valid), color='red', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(self.vel_3d_valid):.2f}')
        ax.axvline(np.mean(self.vel_3d_valid), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(self.vel_3d_valid):.2f}')
        ax.set_xlabel('3D Velocity (m/yr)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'3D Total Velocity\nn = {len(self.vel_3d_valid)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Vertical velocity histogram
        ax = axes[2]
        ax.hist(self.vel_z_valid, bins=40, edgecolor='black', alpha=0.7, color='crimson')
        ax.axvline(0, color='black', linestyle='-', linewidth=2, label='No change')
        ax.axvline(np.median(self.vel_z_valid), color='red', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(self.vel_z_valid):.3f}')
        ax.set_xlabel('Vertical Velocity (m/yr)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Vertical Velocity\nn = {len(self.vel_z_valid)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_velocity_statistics(self, save_path=None, show=True):
        """
        Create comprehensive statistical plots.
        """
        print("\n" + "="*70)
        print("CREATING STATISTICAL PLOTS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Velocity vs Distance from center
        ax = axes[0, 0]
        center_x, center_y = np.mean(self.piv_x_valid), np.mean(self.piv_y_valid)
        distances = np.sqrt((self.piv_x_valid - center_x)**2 + (self.piv_y_valid - center_y)**2)
        
        scatter = ax.scatter(distances, self.vel_2d_valid, alpha=0.6, 
                            c=self.vel_2d_valid, cmap='plasma', s=30)
        
        # Add trend line
        z = np.polyfit(distances, self.vel_2d_valid, 1)
        p = np.poly1d(z)
        ax.plot(distances, p(distances), "r--", linewidth=2, 
                label=f'Trend: {z[0]:.4f}x + {z[1]:.2f}')
        
        ax.set_xlabel('Distance from Center (pixels)', fontsize=11)
        ax.set_ylabel('2D Velocity (m/yr)', fontsize=11)
        ax.set_title('Velocity vs. Distance from Center')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Velocity (m/yr)')
        
        # 2. Q-Q plot (normality test)
        ax = axes[0, 1]
        stats.probplot(self.vel_2d_valid, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot: 2D Velocity\n(Test for Normality)')
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        ax = axes[1, 0]
        sorted_vel = np.sort(self.vel_2d_valid)
        cumulative = np.arange(1, len(sorted_vel) + 1) / len(sorted_vel) * 100
        ax.plot(sorted_vel, cumulative, linewidth=2, color='steelblue')
        ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50th percentile')
        ax.axhline(90, color='orange', linestyle='--', alpha=0.7, label='90th percentile')
        ax.set_xlabel('2D Velocity (m/yr)', fontsize=11)
        ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
        ax.set_title('Cumulative Distribution Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Box plot comparison
        ax = axes[1, 1]
        data_to_plot = [self.vel_2d_valid, self.vel_3d_valid, self.vel_z_valid]
        bp = ax.boxplot(data_to_plot, labels=['2D Horiz.', '3D Total', 'Vertical'],
                        patch_artist=True, showfliers=False)
        
        colors = ['steelblue', 'darkorange', 'crimson']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Velocity (m/yr)', fontsize=11)
        ax.set_title('Velocity Component Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_transect_profile(self, start_point, end_point, width=5, save_path=None, show=True):
        """
        Extract and plot a velocity profile along a transect.
        
        Parameters:
        -----------
        start_point : tuple
            (x, y) starting pixel coordinates
        end_point : tuple
            (x, y) ending pixel coordinates
        width : int
            Width of transect in pixels (for averaging)
        """
        print("\n" + "="*70)
        print("CREATING TRANSECT PROFILE")
        print("="*70)
        print(f"  Start: {start_point}, End: {end_point}, Width: {width} pixels")
        
        # Create interpolated velocity map
        vel_interp = self.create_interpolated_velocity_map()
        
        # Generate points along transect
        x0, y0 = start_point
        x1, y1 = end_point
        num_points = int(np.sqrt((x1-x0)**2 + (y1-y0)**2))
        x_points = np.linspace(x0, x1, num_points)
        y_points = np.linspace(y0, y1, num_points)
        
        # Extract velocity values with width
        profile_values = []
        for x, y in zip(x_points, y_points):
            # Sample in perpendicular direction for width
            dx = y1 - y0
            dy = -(x1 - x0)
            length = np.sqrt(dx**2 + dy**2)
            dx /= length
            dy /= length
            
            samples = []
            for w in np.linspace(-width/2, width/2, width):
                xi = int(x + w * dx)
                yi = int(y + w * dy)
                if 0 <= xi < vel_interp.shape[1] and 0 <= yi < vel_interp.shape[0]:
                    val = vel_interp[yi, xi]
                    if not np.isnan(val):
                        samples.append(val)
            
            profile_values.append(np.mean(samples) if samples else np.nan)
        
        # Calculate distance along profile
        distances = np.sqrt((x_points - x0)**2 + (y_points - y0)**2)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top: Map with transect line
        ax = axes[0]
        ax.imshow(vel_interp, cmap='plasma', vmin=0, 
                 vmax=np.nanpercentile(vel_interp, 95))
        ax.plot([x0, x1], [y0, y1], 'r-', linewidth=3, label='Transect')
        ax.plot([x0, x1], [y0, y1], 'w--', linewidth=1)
        ax.scatter([x0, x1], [y0, y1], c='red', s=100, marker='o', 
                  edgecolors='white', linewidths=2, zorder=5)
        ax.set_title('Velocity Map with Transect Location', fontsize=12)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.legend()
        
        # Bottom: Profile plot
        ax = axes[1]
        ax.plot(distances, profile_values, linewidth=2, color='steelblue')
        ax.fill_between(distances, 0, profile_values, alpha=0.3, color='steelblue')
        ax.set_xlabel('Distance along transect (pixels)', fontsize=11)
        ax.set_ylabel('2D Velocity (m/yr)', fontsize=11)
        ax.set_title(f'Velocity Profile (Width: {width} pixels)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add summary stats
        valid_profile = np.array(profile_values)[~np.isnan(profile_values)]
        if len(valid_profile) > 0:
            ax.axhline(np.mean(valid_profile), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(valid_profile):.2f} m/yr')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def print_summary_statistics(self):
        """
        Print comprehensive summary statistics to console.
        """
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        print("\n2D HORIZONTAL VELOCITY:")
        print(f"  n (valid): {len(self.vel_2d_valid)}")
        print(f"  Min:       {np.min(self.vel_2d_valid):.3f} m/yr")
        print(f"  Max:       {np.max(self.vel_2d_valid):.3f} m/yr")
        print(f"  Mean:      {np.mean(self.vel_2d_valid):.3f} m/yr")
        print(f"  Median:    {np.median(self.vel_2d_valid):.3f} m/yr")
        print(f"  Std Dev:   {np.std(self.vel_2d_valid):.3f} m/yr")
        print(f"  25th %ile: {np.percentile(self.vel_2d_valid, 25):.3f} m/yr")
        print(f"  75th %ile: {np.percentile(self.vel_2d_valid, 75):.3f} m/yr")
        print(f"  95th %ile: {np.percentile(self.vel_2d_valid, 95):.3f} m/yr")
        
        print("\n3D TOTAL VELOCITY:")
        print(f"  n (valid): {len(self.vel_3d_valid)}")
        print(f"  Min:       {np.min(self.vel_3d_valid):.3f} m/yr")
        print(f"  Max:       {np.max(self.vel_3d_valid):.3f} m/yr")
        print(f"  Mean:      {np.mean(self.vel_3d_valid):.3f} m/yr")
        print(f"  Median:    {np.median(self.vel_3d_valid):.3f} m/yr")
        print(f"  Std Dev:   {np.std(self.vel_3d_valid):.3f} m/yr")
        
        print("\nVERTICAL VELOCITY:")
        print(f"  n (valid): {len(self.vel_z_valid)}")
        print(f"  Min:       {np.min(self.vel_z_valid):.3f} m/yr")
        print(f"  Max:       {np.max(self.vel_z_valid):.3f} m/yr")
        print(f"  Mean:      {np.mean(self.vel_z_valid):.3f} m/yr")
        print(f"  Median:    {np.median(self.vel_z_valid):.3f} m/yr")
        print(f"  Std Dev:   {np.std(self.vel_z_valid):.3f} m/yr")
        
        # Test if mean is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(self.vel_z_valid, 0)
        print(f"\n  One-sample t-test (H0: mean = 0):")
        print(f"    t-statistic: {t_stat:.3f}")
        print(f"    p-value:     {p_value:.6f}")
        if p_value < 0.05:
            print(f"    Result: Significant vertical motion detected (p < 0.05)")
        else:
            print(f"    Result: No significant vertical motion (p >= 0.05)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # USER CONFIGURATION
    # -------------------------------------------------------------------------
    
    # Input files
    RESULTS_FILE = Path("results/displacement_results_3d.npz")
    DEM1_FILE = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_LIDAR/2018_0p5m_upper_rg_dem_larger_roi.tif")
    HS1_FILE = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_2023_hs_large/2018_0p5m_hs_large.tif")
    STABLE_FILE = Path(r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\shapefiles\stable_2.shp")
    
    # Output directory
    OUTPUT_DIR = Path("visualizations")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Visualization toggles (set to False to skip)
    PLOT_OVERVIEW = True
    PLOT_VECTORS = True
    PLOT_HILLSHADE_OVERLAY = True
    PLOT_2D_VS_3D = True
    PLOT_HISTOGRAMS = True
    PLOT_STATISTICS = True
    PLOT_TRANSECT = True  # Requires manual transect definition
    PRINT_STATS = True
    
    # Display plots interactively?
    SHOW_PLOTS = True
    
    # -------------------------------------------------------------------------
    # RUN VISUALIZATION
    # -------------------------------------------------------------------------
    
    try:
        # Initialize visualizer
        viz = DisplacementVisualizer(
            results_file=RESULTS_FILE,
            dem1_file=DEM1_FILE,
            hs1_file=HS1_FILE,
            stable_shapefile=STABLE_FILE
        )
        
        # Print statistics
        if PRINT_STATS:
            viz.print_summary_statistics()
        
        # Generate plots
        if PLOT_OVERVIEW:
            viz.plot_velocity_overview(
                save_path=OUTPUT_DIR / "01_velocity_overview.png",
                show=SHOW_PLOTS
            )
        
        if PLOT_VECTORS:
            viz.plot_velocity_vectors(
                subsample=2,  # Plot every 2nd vector
                scale=15,     # Adjust arrow size
                save_path=OUTPUT_DIR / "02_vector_field.png",
                show=SHOW_PLOTS
            )
        
        if PLOT_HILLSHADE_OVERLAY:
            viz.plot_velocity_hillshade_overlay(
                alpha=0.5,  # Hillshade transparency
                save_path=OUTPUT_DIR / "03_hillshade_overlay.png",
                show=SHOW_PLOTS
            )
        
        if PLOT_2D_VS_3D:
            viz.plot_3d_velocity_comparison(
                save_path=OUTPUT_DIR / "04_2d_vs_3d_comparison.png",
                show=SHOW_PLOTS
            )
        
        if PLOT_HISTOGRAMS:
            viz.plot_velocity_histograms(
                save_path=OUTPUT_DIR / "05_velocity_histograms.png",
                show=SHOW_PLOTS
            )
        
        if PLOT_STATISTICS:
            viz.plot_velocity_statistics(
                save_path=OUTPUT_DIR / "06_statistical_plots.png",
                show=SHOW_PLOTS
            )
        
        if PLOT_TRANSECT:
            # Define transect endpoints (adjust these based on your data!)
            # These should be (x, y) pixel coordinates
            # You can determine these by looking at your velocity maps
            
            # Example: diagonal transect across the glacier
            shape = viz.dem.shape
            start = (shape[1] * 0.2, shape[0] * 0.2)  # Lower-left quadrant
            end = (shape[1] * 0.8, shape[0] * 0.8)    # Upper-right quadrant
            
            viz.plot_transect_profile(
                start_point=start,
                end_point=end,
                width=5,  # Average across 5 pixels
                save_path=OUTPUT_DIR / "07_transect_profile.png",
                show=SHOW_PLOTS
            )
        
        print("\n" + "="*70)
        print("VISUALIZATION COMPLETE!")
        print("="*70)
        print(f"All figures saved to: {OUTPUT_DIR.absolute()}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found")
        print(f"   {e}")
        print("\n   Make sure you've run the processing script first!")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()