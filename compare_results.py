"""
Quick Comparison Tool for Rock Glacier Results
===============================================

This script allows you to compare results from different processing runs
side-by-side to evaluate the effect of parameter changes.

Usage:
------
1. Run the main processing script with different parameters
2. Save each result with a descriptive name
3. Use this script to compare them visually
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

def load_results(filepath):
    """Load a results file and extract key metrics."""
    results = np.load(filepath)
    
    vel_2d = results['vel_2d_ma']
    vel_3d = results['vel_3d_ma']
    vel_z = results['vel_z_ma']
    
    # Get valid (non-NaN) values
    vel_2d_valid = vel_2d[~np.isnan(vel_2d)]
    vel_3d_valid = vel_3d[~np.isnan(vel_3d)]
    vel_z_valid = vel_z[~np.isnan(vel_z)]
    
    return {
        'vel_2d': vel_2d,
        'vel_3d': vel_3d,
        'vel_z': vel_z,
        'vel_2d_valid': vel_2d_valid,
        'vel_3d_valid': vel_3d_valid,
        'vel_z_valid': vel_z_valid,
        'n_vectors': len(vel_2d_valid),
        'mean_2d': np.mean(vel_2d_valid),
        'median_2d': np.median(vel_2d_valid),
        'std_2d': np.std(vel_2d_valid),
        'max_2d': np.max(vel_2d_valid),
    }

def compare_two_runs(file1, file2, label1="Run 1", label2="Run 2", 
                     save_path=None, show=True):
    """
    Create a comprehensive comparison between two processing runs.
    """
    print("="*70)
    print(f"COMPARING TWO RUNS")
    print("="*70)
    print(f"Run 1: {file1}")
    print(f"Run 2: {file2}")
    
    # Load results
    r1 = load_results(file1)
    r2 = load_results(file2)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Velocity maps
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(r1['vel_2d'], cmap='plasma', vmin=0, 
                     vmax=max(np.nanmax(r1['vel_2d']), np.nanmax(r2['vel_2d'])))
    ax1.set_title(f'{label1}\n2D Velocity', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='m/yr', shrink=0.7)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(r2['vel_2d'], cmap='plasma', vmin=0,
                     vmax=max(np.nanmax(r1['vel_2d']), np.nanmax(r2['vel_2d'])))
    ax2.set_title(f'{label2}\n2D Velocity', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='m/yr', shrink=0.7)
    
    # Difference map
    ax3 = fig.add_subplot(gs[0, 2])
    diff = r2['vel_2d'] - r1['vel_2d']
    vmax_diff = np.nanpercentile(np.abs(diff), 95)
    im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    ax3.set_title(f'Difference\n({label2} - {label1})', fontsize=11)
    plt.colorbar(im3, ax=ax3, label='m/yr', shrink=0.7)
    
    # Row 2: Histograms
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(r1['vel_2d_valid'], bins=30, alpha=0.6, label=label1, 
             color='steelblue', edgecolor='black')
    ax4.axvline(r1['mean_2d'], color='blue', linestyle='--', linewidth=2)
    ax4.set_xlabel('2D Velocity (m/yr)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'{label1} Distribution\nMean: {r1["mean_2d"]:.2f}', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(r2['vel_2d_valid'], bins=30, alpha=0.6, label=label2,
             color='darkorange', edgecolor='black')
    ax5.axvline(r2['mean_2d'], color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('2D Velocity (m/yr)')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'{label2} Distribution\nMean: {r2["mean_2d"]:.2f}', fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # Overlay histograms
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(r1['vel_2d_valid'], bins=30, alpha=0.5, label=label1,
             color='steelblue', edgecolor='black')
    ax6.hist(r2['vel_2d_valid'], bins=30, alpha=0.5, label=label2,
             color='darkorange', edgecolor='black')
    ax6.set_xlabel('2D Velocity (m/yr)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Overlaid Distributions', fontsize=11)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Row 3: Statistics
    ax7 = fig.add_subplot(gs[2, 0])
    metrics = ['Mean', 'Median', 'Std Dev', 'Max', 'N Vectors']
    r1_vals = [r1['mean_2d'], r1['median_2d'], r1['std_2d'], 
               r1['max_2d'], r1['n_vectors']]
    r2_vals = [r2['mean_2d'], r2['median_2d'], r2['std_2d'],
               r2['max_2d'], r2['n_vectors']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize for plotting (except N Vectors)
    r1_plot = r1_vals.copy()
    r2_plot = r2_vals.copy()
    r1_plot[4] = r1_plot[4] / 100  # Scale down vector count
    r2_plot[4] = r2_plot[4] / 100
    
    ax7.bar(x - width/2, r1_plot, width, label=label1, color='steelblue', alpha=0.7)
    ax7.bar(x + width/2, r2_plot, width, label=label2, color='darkorange', alpha=0.7)
    ax7.set_ylabel('Value')
    ax7.set_title('Metric Comparison', fontsize=11)
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics, rotation=45, ha='right')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.text(4, max(r1_plot[4], r2_plot[4]) * 1.1, '(×100)', ha='center', fontsize=8)
    
    # Cumulative distribution
    ax8 = fig.add_subplot(gs[2, 1])
    sorted_r1 = np.sort(r1['vel_2d_valid'])
    sorted_r2 = np.sort(r2['vel_2d_valid'])
    cum_r1 = np.arange(1, len(sorted_r1) + 1) / len(sorted_r1) * 100
    cum_r2 = np.arange(1, len(sorted_r2) + 1) / len(sorted_r2) * 100
    
    ax8.plot(sorted_r1, cum_r1, linewidth=2, label=label1, color='steelblue')
    ax8.plot(sorted_r2, cum_r2, linewidth=2, label=label2, color='darkorange')
    ax8.set_xlabel('2D Velocity (m/yr)')
    ax8.set_ylabel('Cumulative %')
    ax8.set_title('Cumulative Distributions', fontsize=11)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Summary table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
    SUMMARY COMPARISON
    
    {label1:15s} {label2:15s} Δ
    
    Mean:    {r1['mean_2d']:5.2f} m/yr  {r2['mean_2d']:5.2f} m/yr  {r2['mean_2d']-r1['mean_2d']:+5.2f}
    Median:  {r1['median_2d']:5.2f} m/yr  {r2['median_2d']:5.2f} m/yr  {r2['median_2d']-r1['median_2d']:+5.2f}
    Std Dev: {r1['std_2d']:5.2f} m/yr  {r2['std_2d']:5.2f} m/yr  {r2['std_2d']-r1['std_2d']:+5.2f}
    Max:     {r1['max_2d']:5.2f} m/yr  {r2['max_2d']:5.2f} m/yr  {r2['max_2d']-r1['max_2d']:+5.2f}
    
    N Vect:  {r1['n_vectors']:5d}       {r2['n_vectors']:5d}       {r2['n_vectors']-r1['n_vectors']:+5d}
    
    Correlation: {np.corrcoef(r1['vel_2d'].flatten()[~np.isnan(r1['vel_2d'].flatten()) & ~np.isnan(r2['vel_2d'].flatten())], r2['vel_2d'].flatten()[~np.isnan(r1['vel_2d'].flatten()) & ~np.isnan(r2['vel_2d'].flatten())])[0,1]:.3f}
    """
    
    ax9.text(0.1, 0.5, summary_text, fontfamily='monospace', 
             fontsize=9, verticalalignment='center')
    
    plt.suptitle('Processing Run Comparison', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def compare_parameter_sweep(result_files, labels, save_dir=None):
    """
    Compare multiple runs (e.g., testing different window sizes).
    
    Parameters:
    -----------
    result_files : list of Path
        List of .npz result files
    labels : list of str
        Labels for each run
    save_dir : Path
        Directory to save comparison plots
    """
    print("="*70)
    print(f"PARAMETER SWEEP COMPARISON ({len(result_files)} runs)")
    print("="*70)
    
    # Load all results
    results = [load_results(f) for f in result_files]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Mean velocity vs. parameter
    ax = axes[0, 0]
    means = [r['mean_2d'] for r in results]
    ax.plot(range(len(labels)), means, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean 2D Velocity (m/yr)', fontsize=11)
    ax.set_title('Mean Velocity Across Runs', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Number of vectors
    ax = axes[0, 1]
    n_vects = [r['n_vectors'] for r in results]
    ax.bar(range(len(labels)), n_vects, color='darkorange', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Number of Valid Vectors', fontsize=11)
    ax.set_title('Data Coverage', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Overlaid histograms
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, (r, label, color) in enumerate(zip(results, labels, colors)):
        ax.hist(r['vel_2d_valid'], bins=30, alpha=0.5, label=label,
                color=color, edgecolor='black')
    ax.set_xlabel('2D Velocity (m/yr)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Velocity Distributions', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Box plot comparison
    ax = axes[1, 1]
    data = [r['vel_2d_valid'] for r in results]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('2D Velocity (m/yr)', fontsize=11)
    ax.set_title('Statistical Distribution Comparison', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / "parameter_sweep_comparison.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved: {save_path}")
    
    plt.show()
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Run':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'Max':>8} {'N Vect':>8}")
    print("-"*70)
    for label, r in zip(labels, results):
        print(f"{label:<20} {r['mean_2d']:8.2f} {r['median_2d']:8.2f} "
              f"{r['std_2d']:8.2f} {r['max_2d']:8.2f} {r['n_vectors']:8d}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # Example 1: Compare two different processing runs
    # -------------------------------------------------------------------------
    
    # Uncomment and adjust paths:
    """
    compare_two_runs(
        file1=Path("results/run1_window32.npz"),
        file2=Path("results/run2_window64.npz"),
        label1="Window=32",
        label2="Window=64",
        save_path=Path("comparisons/window_comparison.png"),
        show=True
    )
    """
    
    # -------------------------------------------------------------------------
    # Example 2: Parameter sweep (multiple window sizes)
    # -------------------------------------------------------------------------
    
    # Uncomment and adjust:
    """
    result_files = [
        Path("results/window16.npz"),
        Path("results/window32.npz"),
        Path("results/window64.npz"),
        Path("results/window128.npz"),
    ]
    
    labels = ["Win=16", "Win=32", "Win=64", "Win=128"]
    
    compare_parameter_sweep(
        result_files=result_files,
        labels=labels,
        save_dir=Path("comparisons")
    )
    """
    
    # -------------------------------------------------------------------------
    # Example 3: Compare current results with default
    # -------------------------------------------------------------------------
    
    print("\nQuick Comparison Template")
    print("="*70)
    print("""
To compare two runs:

1. Run processing script with parameters A, save as: results_A.npz
2. Run processing script with parameters B, save as: results_B.npz
3. Run this script with:

compare_two_runs(
    file1="results_A.npz",
    file2="results_B.npz",
    label1="Setup A",
    label2="Setup B",
    save_path="comparison.png"
)

Common comparisons:
- Different window sizes (16, 32, 64, 128)
- Different overlap values (8, 16, 24, 32)
- Different search sizes (32, 48, 64, 96)
- With/without stable area calibration
- Different time periods
    """)
    
    print("\nTo use this script, uncomment one of the examples above")
