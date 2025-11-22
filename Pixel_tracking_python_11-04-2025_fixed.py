"""
Rock Glacier 3D Displacement Analysis - FIXED VERSION
======================================

This script combines two primary methods to measure 3D rock glacier movement:

1.  **X,Y Displacement (Horizontal):** Uses OpenPIV (Particle Image Velocimetry)
    on repeat LiDAR hillshades to track horizontal feature movement.
2.  **Z Displacement (Vertical):** Uses xdem (DEM differencing) to find
    vertical changes (subsidence/uplift).

Both methods are calibrated using a user-provided shapefile of
stable ground (e.g., bedrock) to remove systematic errors and false positives.
"""

import numpy as np
import rasterio
import geopandas as gpd
import xdem
import openpiv.tools
import openpiv.pyprocess
import openpiv.validation
import openpiv.filters
from rasterio.features import geometry_mask
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

# Add scikit-image import at module level
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. SSIM analysis will be skipped.")
    SSIM_AVAILABLE = False


class RockGlacierTracker:
    """
    A class to manage the full 3D displacement tracking workflow.
    """

    def __init__(self, stable_area_shapefile):
        """
        Initializes the tracker.

        Parameters:
        -----------
        stable_area_shapefile : str or Path
            Filepath to a polygon shapefile (.shp) defining stable bedrock
            areas. This is CRITICAL for accurate calibration.
        """
        self.stable_area_shapefile = stable_area_shapefile
        if not Path(self.stable_area_shapefile).exists():
            raise FileNotFoundError(f"Stable area file not found: {self.stable_area_shapefile}")
            
        # These will be populated by the class methods
        self.stable_mask = None
        self.piv_stable_mask = None
        self.piv_x = None
        self.piv_y = None
        self.piv_grid_shape = None

    def load_and_align_rasters(self, hillshade_path, dem_path):
        """
        Loads the hillshade and DEM for a single time period.
        
        This example assumes they are already co-registered and on the
        same grid. For a more robust workflow, you would first align
        the hillshade to the DEM if they differ.

        Parameters:
        -----------
        hillshade_path : str or Path
            Filepath to the hillshade raster (e.g., "hs_2018.tif").
        dem_path : str or Path
            Filepath to the DEM raster (e.g., "dem_2018.tif").

        Returns:
        --------
        tuple
            (hillshade_array, dem_array, rasterio_transform, rasterio_crs)
        """
        print(f"Loading data from {hillshade_path} and {dem_path}...")
        with rasterio.open(hillshade_path) as src:
            hillshade = src.read(1)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            
        # Open the DEM and ensure it matches the hillshade's grid
        # A more complex workflow might reproject/resample the DEM
        # to the hillshade's grid here.
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            if src.transform != transform or src.shape != hillshade.shape:
                print("Warning: DEM and hillshade grids do not match.")
                # Add reprojection logic here if needed
                
        return hillshade, dem, transform, crs

    def create_stable_mask(self, reference_shape, reference_transform, reference_crs):
        """
        Creates a boolean mask from the stable ground shapefile.
        The mask will be True for stable areas and False for moving areas.

        Parameters:
        -----------
        reference_shape : tuple
            The (row, col) shape of the reference raster (e.g., dem1.shape).
        reference_transform : affine.Affine
            The transform of the reference raster.
        reference_crs : rasterio.crs.CRS
            The CRS of the reference raster.
        """
        print("Creating stable ground mask...")
        # Load the shapefile
        stable_gdf = gpd.read_file(self.stable_area_shapefile)

        # Reproject shapefile to match raster CRS if they don't match
        if stable_gdf.crs != reference_crs:
            print(f"Reprojecting stable mask from {stable_gdf.crs} to {reference_crs}...")
            stable_gdf = stable_gdf.to_crs(reference_crs)

        # Create the mask.
        # 'invert=True' makes pixels *inside* the polygons True.
        self.stable_mask = geometry_mask(
            stable_gdf.geometry,
            transform=reference_transform,
            invert=True,
            out_shape=reference_shape
        )
        
        if not np.any(self.stable_mask):
            print("Warning: The stable mask is empty. Check shapefile and CRS.")
            
    def check_hillshade_quality(self, hs1, hs2, output_dir, downsample_factor=4):
        """
        Analyzes hillshade similarity to determine if PIV will work.
        Uses downsampling for faster computation on large datasets.
        
        Parameters:
        -----------
        hs1, hs2 : np.ndarray
            The two hillshade arrays to compare
        output_dir : Path
            Directory to save diagnostic plots
        downsample_factor : int
            Factor to downsample images for faster SSIM calculation (default: 4)
        """
        print("\n--- HILLSHADE SIMILARITY ANALYSIS ---")
        
        # 1. Calculate hillshade difference
        hs_diff = hs2.astype(float) - hs1.astype(float)
        
        # 2. Downsample for faster correlation calculation
        hs1_small = hs1[::downsample_factor, ::downsample_factor]
        hs2_small = hs2[::downsample_factor, ::downsample_factor]
        
        # 3. Calculate normalized cross-correlation at zero lag
        print("  Calculating correlation (downsampled)...")
        correlation = signal.correlate2d(
            (hs1_small - hs1_small.mean()) / (hs1_small.std() + 1e-10),
            (hs2_small - hs2_small.mean()) / (hs2_small.std() + 1e-10),
            mode='same'
        ) / (hs1_small.size)
        
        # 4. Calculate SSIM if available (on downsampled data)
        if SSIM_AVAILABLE:
            print("  Calculating structural similarity (downsampled)...")
            similarity_score = ssim(
                hs1_small, 
                hs2_small, 
                data_range=hs1_small.max() - hs1_small.min()
            )
            print(f"  Structural similarity (SSIM): {similarity_score:.3f}")
        else:
            similarity_score = None
        
        print(f"  Hillshade correlation: {correlation[correlation.shape[0]//2, correlation.shape[1]//2]:.3f}")
        print(f"  Hillshade difference range: [{np.min(hs_diff):.1f}, {np.max(hs_diff):.1f}]")
        print(f"  Hillshade difference std: {np.std(hs_diff):.1f}")
        
        # 5. Visual check
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original hillshades
        axes[0, 0].imshow(hs1, cmap='gray')
        axes[0, 0].set_title('2018 Hillshade')
        
        axes[0, 1].imshow(hs2, cmap='gray')
        axes[0, 1].set_title('2023 Hillshade')
        
        # Difference
        axes[0, 2].imshow(hs_diff, cmap='RdBu_r', vmin=-50, vmax=50)
        axes[0, 2].set_title('Hillshade Difference')
        
        # Zoom into stable area to check texture (if mask exists)
        if self.stable_mask is not None:
            stable_coords = np.where(self.stable_mask)
            if len(stable_coords[0]) > 0:
                center_y, center_x = int(np.median(stable_coords[0])), int(np.median(stable_coords[1]))
                window = 100
                
                y1, y2 = max(0, center_y - window), min(hs1.shape[0], center_y + window)
                x1, x2 = max(0, center_x - window), min(hs1.shape[1], center_x + window)
                
                axes[1, 0].imshow(hs1[y1:y2, x1:x2], cmap='gray')
                axes[1, 0].set_title('2018 Stable Area (Zoomed)')
                
                axes[1, 1].imshow(hs2[y1:y2, x1:x2], cmap='gray')
                axes[1, 1].set_title('2023 Stable Area (Zoomed)')
                
                axes[1, 2].imshow(hs_diff[y1:y2, x1:x2], cmap='RdBu_r', vmin=-50, vmax=50)
                axes[1, 2].set_title('Difference (Stable Area)')
            else:
                print("  Warning: No stable areas found for zoomed view")
        else:
            print("  Skipping stable area zoom (mask not yet created)")
        
        plt.tight_layout()
        plt.savefig(output_dir / 'hillshade_comparison.png', dpi=150)
        print(f"  Saved: hillshade_comparison.png")
        plt.close()
        
        # 6. Check if hillshades are suitable for PIV
        if similarity_score is not None and similarity_score < 0.7:
            print("\n⚠️  WARNING: Low hillshade similarity (<0.7)")
            print("     Different sensors/processing may prevent reliable feature tracking")
            print("     Consider alternative methods (manual feature tracking, GPS data)")
        
        return similarity_score

    def calculate_2d_displacement(self, hillshade1, hillshade2,
                              window_size=24, overlap=18,
                              search_size=36):
        """
        Calculates the X,Y displacement field using OpenPIV.
    
        Parameters:
        -----------
        hillshade1, hillshade2 : np.ndarray
            The "before" and "after" hillshade arrays.
        window_size : int
            The size (in pixels) of the matching window.
        overlap : int
            Pixel overlap between windows. (window_size - overlap) is the step.
        search_size : int
            The size (in pixels) of the area to search for a match.
            Must be larger than window_size.
    
        Returns:
        --------
        tuple
            (u, v)
            u: X-displacement in pixels (calibrated)
            v: Y-displacement in pixels (calibrated)
        """
        print("Calculating 2D displacement with OpenPIV...")
        
        # Pre-process hillshades to enhance features
        h1 = ndimage.median_filter(hillshade1, size=3)
        h2 = ndimage.median_filter(hillshade2, size=3)
        
        # Normalize to improve correlation
        h1 = (h1 - h1.mean()) / (h1.std() + 1e-10)
        h2 = (h2 - h2.mean()) / (h2.std() + 1e-10)
    
        # --- Run OpenPIV ---
        print("  Running PIV correlation...")
        u, v, sig2noise = openpiv.pyprocess.extended_search_area_piv(
            h1.astype(np.float32),
            h2.astype(np.float32),
            window_size=window_size,
            overlap=overlap,
            search_area_size=search_size,
            sig2noise_method='peak2peak'
        )
    
        # Get the grid coordinates (in pixels)
        x, y = openpiv.pyprocess.get_coordinates(
            image_size=h1.shape,
            search_area_size=search_size,
            overlap=overlap
        )
        
        # Store for later use
        self.piv_x = x
        self.piv_y = y
        self.piv_grid_shape = u.shape
    
        print(f"  PIV grid size: {u.shape}")
        
        # --- Filter bad vectors ---
        print("  Filtering bad vectors...")
        # Create mask for bad vectors based on signal-to-noise ratio
        invalid_mask = openpiv.validation.sig2noise_val(
            sig2noise,
            threshold=0.3
        )
        
        print(f"  Filtered {np.sum(invalid_mask)} bad vectors")
        
        # Replace outliers with local mean
        u, v = openpiv.filters.replace_outliers(
            u, v, 
            invalid_mask,
            method='localmean', 
            max_iter=3, 
            kernel_size=2
        )
        
        print("  2D displacement calculation complete")
        return u, v

    def calibrate_with_stable_areas(self, u, v, reference_transform):
        """
        Removes systematic bias from PIV results using the stable mask.
        This is the "false positive" fix.

        Parameters:
        -----------
        u, v : np.ndarray
            The X and Y displacement fields from OpenPIV.
        reference_transform : affine.Affine
            The raster transform, used to map pixel coords.

        Returns:
        --------
        tuple
            (u_calibrated, v_calibrated)
        """
        print("Calibrating 2D displacement field...")
        if self.stable_mask is None:
            print("Warning: Stable mask not created. Skipping calibration.")
            return u, v
            
        # We need to find which PIV vectors fall on stable ground.
        # The PIV grid (piv_x, piv_y) is different from the DEM grid.
        
        # 1. Get the real-world (map) coordinates of the PIV vector origins
        map_x, map_y = rasterio.transform.xy(
            reference_transform, 
            self.piv_y.flatten(),  # Note: y is rows
            self.piv_x.flatten()   # Note: x is cols
        )

        # 2. Get the row, col indices of these map coords on the original mask
        # Note: ~transform means "inverse transform"
        rows, cols = rasterio.transform.rowcol(
            ~reference_transform, 
            map_x, 
            map_y
        )
        
        # 3. Sample the stable mask at these locations
        # We must clip indices to be within the mask bounds
        rows = np.clip(rows, 0, self.stable_mask.shape[0] - 1)
        cols = np.clip(cols, 0, self.stable_mask.shape[1] - 1)
        
        stable_at_grid_flat = self.stable_mask[rows, cols]
        
        # 4. Reshape this flat array back into the PIV grid shape
        self.piv_stable_mask = stable_at_grid_flat.reshape(self.piv_grid_shape)

        if np.any(self.piv_stable_mask):
            # Calculate mean displacement (bias) ONLY on stable ground
            u_bias = np.mean(u[self.piv_stable_mask])
            v_bias = np.mean(v[self.piv_stable_mask])

            print(f"  Stable area bias found: dx={u_bias:.3f}, dy={v_bias:.3f} pixels")
            print("  Removing systematic bias from all measurements...")

            # Subtract this bias from the *entire* displacement field
            u_calibrated = u - u_bias
            v_calibrated = v - v_bias

            # Optional: Force stable areas to be exactly zero
            u_calibrated[self.piv_stable_mask] = 0
            v_calibrated[self.piv_stable_mask] = 0
            
            return u_calibrated, v_calibrated
        else:
            print("Warning: No stable areas found in PIV grid. Cannot calibrate.")
            return u, v

    def calculate_vertical_change(self, dem1, dem2, transform, crs):
        """
        Calculates vertical (Z) change using xdem, calibrated with the stable mask.
    
        Parameters:
        -----------
        dem1, dem2 : np.ndarray
            The "before" and "after" DEM arrays.
        transform : affine.Affine
            The raster transform.
        crs : rasterio.crs.CRS
            The raster CRS.
    
        Returns:
        --------
        np.ndarray
            The 2D array of vertical change (dh) in the DEM's units.
        """
        print("Calculating vertical (Z) change with xdem...")
        if self.stable_mask is None:
            raise ValueError("Stable mask must be created before calculating vertical change.")
        
        # --- Handle nodata values properly ---
        print("  Cleaning DEM data...")
        
        # Identify nodata pixels
        nodata_mask1 = np.isnan(dem1) | (dem1 < -1000) | (dem1 > 10000)
        nodata_mask2 = np.isnan(dem2) | (dem2 < -1000) | (dem2 > 10000)
        
        # Create copies and mask nodata
        dem1_clean = np.ma.masked_array(dem1, mask=nodata_mask1)
        dem2_clean = np.ma.masked_array(dem2, mask=nodata_mask2)
        
        print(f"  DEM1: {np.sum(nodata_mask1)} nodata pixels ({100*np.sum(nodata_mask1)/nodata_mask1.size:.1f}%)")
        print(f"  DEM2: {np.sum(nodata_mask2)} nodata pixels ({100*np.sum(nodata_mask2)/nodata_mask2.size:.1f}%)")
        print(f"  DEM1 range: {np.min(dem1_clean):.1f} to {np.max(dem1_clean):.1f} m")
        print(f"  DEM2 range: {np.min(dem2_clean):.1f} to {np.max(dem2_clean):.1f} m")
    
        # Create xdem.DEM objects using cleaned data
        dem1_xdem = xdem.DEM.from_array(
            data=dem1_clean.filled(-9999),
            transform=transform,
            crs=crs,
            nodata=-9999
        )
        dem2_xdem = xdem.DEM.from_array(
            data=dem2_clean.filled(-9999),
            transform=transform,
            crs=crs,
            nodata=-9999
        )
    
        # --- Attempt co-registration with robust settings ---
        print("  Attempting DEM co-registration...")
        try:
            coregistration = xdem.coreg.ICP()
            coregistration.fit(
                dem1_xdem, 
                dem2_xdem, 
                inlier_mask=self.stable_mask,
                verbose=False
            )
            dem2_aligned = coregistration.apply(dem2_xdem)
            print("  ICP co-registration successful")
            
        except Exception as e:
            print(f"  ICP failed ({e}), using Nuth-Kaab...")
            try:
                coregistration = xdem.coreg.NuthKaab()
                coregistration.fit(dem1_xdem, dem2_xdem, inlier_mask=self.stable_mask)
                dem2_aligned = coregistration.apply(dem2_xdem)
            except Exception as e2:
                print(f"  Nuth-Kaab also failed ({e2}), skipping co-registration!")
                dem2_aligned = dem2_xdem
    
        # Calculate the difference
        dh = dem2_aligned - dem1_xdem
        
        # Get the data and re-mask nodata
        dh_array = dh.data.copy()
        dh_array[nodata_mask1 | nodata_mask2] = np.nan
        
        # Check the result on stable areas
        stable_dh = dh_array[self.stable_mask & ~nodata_mask1 & ~nodata_mask2]
        if len(stable_dh) > 0:
            print(f"  Stable area dh: mean={np.nanmean(stable_dh):.2f}m, std={np.nanstd(stable_dh):.2f}m")
            print(f"  This should be close to zero if co-registration worked!")
        
        return dh_array

    def combine_to_3d_vectors(self, u_cal, v_cal, dh,
                              pixel_size_x, pixel_size_y, 
                              time_delta_years=1.0):
        """
        Combines horizontal (X,Y) and vertical (Z) displacements
        into final 3D vector components.

        Parameters:
        -----------
        u_cal, v_cal : np.ndarray
            The *calibrated* X and Y displacements in *pixels*.
        dh : np.ndarray
            The vertical change (Z) array, on the *original DEM grid*.
        pixel_size_x, pixel_size_y : float
            The ground size of a pixel (e.g., 0.5 for 0.5m).
        time_delta_years : float
            The time in years between the two datasets.

        Returns:
        --------
        dict
            A dictionary containing all final displacement and
            velocity products, all on the PIV grid.
        """
        print("Combining X,Y, and Z components...")
        
        # --- Convert X,Y from pixels to meters ---
        dx_m = u_cal * pixel_size_x
        dy_m = v_cal * pixel_size_y

        # --- Resample Z to the PIV grid ---
        rows = np.arange(dh.shape[0])
        cols = np.arange(dh.shape[1])
        
        interp = RegularGridInterpolator(
            (rows, cols), 
            dh, 
            bounds_error=False,
            fill_value=np.nan
        )

        points_to_sample = np.column_stack(
            [self.piv_y.flatten(), self.piv_x.flatten()]
        )

        dz_at_grid_flat = interp(points_to_sample)
        dz_m = dz_at_grid_flat.reshape(self.piv_grid_shape)
        
        # --- Calculate final 3D metrics ---
        mag_2d = np.sqrt(dx_m**2 + dy_m**2)
        mag_3d = np.sqrt(dx_m**2 + dy_m**2 + dz_m**2)

        vel_2d_ma = mag_2d / time_delta_years
        vel_3d_ma = mag_3d / time_delta_years
        vel_z_ma = dz_m / time_delta_years

        azimuth = np.degrees(np.arctan2(dx_m, dy_m))
        azimuth = (azimuth + 360) % 360

        return {
            'x_grid_px': self.piv_x,
            'y_grid_px': self.piv_y,
            'dx_m': dx_m,
            'dy_m': dy_m,
            'dz_m': dz_m,
            'vel_2d_ma': vel_2d_ma,
            'vel_3d_ma': vel_3d_ma,
            'vel_z_ma': vel_z_ma,
            'azimuth_deg': azimuth,
            'stable_mask': self.piv_stable_mask
        }

    def filter_vertical_outliers(self, dh, method='iqr', multiplier=3.0):
        """
        Removes extreme outliers from vertical change using IQR method.
        
        Parameters:
        -----------
        dh : np.ndarray
            The vertical change array
        method : str
            'iqr' for interquartile range or 'percentile' for percentile clipping
        multiplier : float
            Number of IQRs beyond which values are considered outliers (default 3.0)
            
        Returns:
        --------
        np.ndarray
            Filtered vertical change array with outliers set to NaN
        """
        dh_filtered = dh.copy()
        valid = np.isfinite(dh)
        
        if method == 'iqr':
            q1, q3 = np.percentile(dh[valid], [25, 75])
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            print(f"  IQR filtering: removing values < {lower:.2f}m or > {upper:.2f}m")
        elif method == 'percentile':
            lower, upper = np.percentile(dh[valid], [1, 99])
            print(f"  Percentile filtering: removing values < {lower:.2f}m or > {upper:.2f}m")
        
        outliers = (dh < lower) | (dh > upper)
        dh_filtered[outliers] = np.nan
        print(f"  Removed {np.sum(outliers)} outliers ({100*np.sum(outliers)/valid.sum():.1f}% of valid pixels)")
        
        return dh_filtered


# =============
#  USAGE EXAMPLE
# =============
if __name__ == "__main__":
    
    # --- 1. Define User Inputs ---
    dem1_file = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_LIDAR/2018_0p5m_upper_rg_dem_larger_roi.tif")
    hs1_file = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_2023_hs_large/2018_0p5m_hs_large.tif")
    dem2_file = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2023_lidar/2023_0p5m_4_imcorr_upper_rg_larger_roi.tif")
    hs2_file = Path(r"M:/My Drive/Rock Glaciers/Field_Sites/Snowbird/Gad_valley/Rasters/2018_2023_hs_large/2023_0p5m_hs_large.tif")
    stable_file = Path(r"M:\My Drive\Rock Glaciers\Field_Sites\Snowbird\Gad_valley\shapefiles\stable_2.shp")

    # Analysis parameters
    PIXEL_SIZE = 0.5  # 0.5 meters
    TIME_DELTA = 2022.0 - 2018.0  # 4.0 years

    try:
        # --- 2. Create output directory FIRST ---
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}\n")
        
        # --- 3. Initialize Tracker ---
        tracker = RockGlacierTracker(stable_area_shapefile=stable_file)

        # --- 4. Load Data (ONCE!) ---
        print("=" * 60)
        print("LOADING RASTER DATA")
        print("=" * 60)
        hs1, dem1, transform, crs = tracker.load_and_align_rasters(hs1_file, dem1_file)
        hs2, dem2, _, _ = tracker.load_and_align_rasters(hs2_file, dem2_file)
        
        # --- 5. DEM Quality Check ---
        print("\n" + "=" * 60)
        print("DEM QUALITY CHECK")
        print("=" * 60)
        print(f"DEM1 shape: {dem1.shape}, dtype: {dem1.dtype}")
        print(f"DEM2 shape: {dem2.shape}, dtype: {dem2.dtype}")
        print(f"DEM1 stats: min={np.nanmin(dem1):.1f}, max={np.nanmax(dem1):.1f}, mean={np.nanmean(dem1):.1f}")
        print(f"DEM2 stats: min={np.nanmin(dem2):.1f}, max={np.nanmax(dem2):.1f}, mean={np.nanmean(dem2):.1f}")
        print(f"NaN pixels: DEM1={np.sum(np.isnan(dem1))}, DEM2={np.sum(np.isnan(dem2))}")
        
        # --- 6. Create Stable Mask (BEFORE hillshade check) ---
        print("\n" + "=" * 60)
        print("CREATING STABLE GROUND MASK")
        print("=" * 60)
        tracker.create_stable_mask(dem1.shape, transform, crs)
        
        # --- 7. Hillshade Quality Check ---
        print("\n" + "=" * 60)
        print("HILLSHADE QUALITY CHECK")
        print("=" * 60)
        similarity = tracker.check_hillshade_quality(hs1, hs2, output_dir, downsample_factor=4)
        
        # --- 8. Run Z-Displacement (Vertical) ---
        print("\n" + "=" * 60)
        print("CALCULATING VERTICAL DISPLACEMENT")
        print("=" * 60)
        dh_map = tracker.calculate_vertical_change(dem1, dem2, transform, crs)
        dh_map = tracker.filter_vertical_outliers(dh_map, method='iqr', multiplier=3.0)
        
        # --- 9. Run X,Y-Displacement (Horizontal) ---
        print("\n" + "=" * 60)
        print("CALCULATING HORIZONTAL DISPLACEMENT")
        print("=" * 60)
        u, v = tracker.calculate_2d_displacement(
            hs1, hs2,
            window_size=32,
            overlap=24,
            search_size=48
        )
        
        # --- 10. Calibrate X,Y ---
        u_cal, v_cal = tracker.calibrate_with_stable_areas(u, v, transform)
        
        # --- 11. Combine All Components ---
        print("\n" + "=" * 60)
        print("COMBINING 3D COMPONENTS")
        print("=" * 60)
        results = tracker.combine_to_3d_vectors(
            u_cal, v_cal,
            dh_map,
            pixel_size_x=PIXEL_SIZE,
            pixel_size_y=PIXEL_SIZE,
            time_delta_years=TIME_DELTA
        )
        
        # --- 12. Save Results ---
        output_file = output_dir / "displacement_results_3d.npz"
        np.savez_compressed(output_file, **results)
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Results saved to {output_file}")
        
        # Print summary statistics
        valid_vel = results['vel_3d_ma'][~np.isnan(results['vel_3d_ma'])]
        print(f"\nMax 3D velocity: {np.max(valid_vel):.2f} m/year")
        print(f"Mean 3D velocity: {np.mean(valid_vel):.2f} m/year")
        print(f"Median 3D velocity: {np.median(valid_vel):.2f} m/year")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: A data file was not found.")
        print(f"   {e}")
        print(f"\n   Please check that:")
        print(f"   1. The M: drive is mounted")
        print(f"   2. All file paths are correct")
        print(f"   3. You have read permissions for the files")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()