# pode_splitter/prediction_analysis/plotting.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress, norm

def plot_scatter(df: pd.DataFrame, output_dir: str, true_col: str, pred_col: str, plot_name_prefix: str = ""):
    """Plot true age vs. predicted age scatter plot, fully matching the reference style."""
    print(f"Generating scatter plot (exact style) - {pred_col} vs {true_col}...")
    
    true_age = df[true_col].to_numpy(dtype=float)
    pred_age = df[pred_col].to_numpy(dtype=float)

    # --- Style parameters (from scatter_plot_performance.py) ---
    CUSTOM_HEX_COLOR = '#94070A'
    POINT_ALPHA = 0.4
    POINT_SIZE = 10
    
    plt.figure(figsize=(8, 8))

    plt.scatter(true_age, pred_age, c=CUSTOM_HEX_COLOR, alpha=POINT_ALPHA, s=POINT_SIZE, edgecolors='none', label='Predictions')

    max_val = max(true_age.max(), pred_age.max()) + 5
    min_val = 10 # fixed value used in the reference file
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Fit (y=x)')

    plt.xlabel(f'{true_col}', fontsize=14)
    plt.ylabel(f'{pred_col}', fontsize=14)
    
    # --- Coordinate system settings (from the reference file) ---
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    # Note: the reference file inverts the Y axis, do the same here
    plt.gca().invert_yaxis()

    plt.grid(False)
    plt.legend(loc='upper right', frameon=True, fontsize=12)

    # Add a prefix to the filename to distinguish between different prediction columns
    prefix = f"{plot_name_prefix}_" if plot_name_prefix else ""
    save_path = os.path.join(output_dir, f"{prefix}scatter_plot_exact_style.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Scatter plot (exact style) saved to: {save_path}")

def plot_bland_altman(df: pd.DataFrame, output_dir: str, true_col: str, pred_col: str, plot_name_prefix: str = ""):
    """Plot Bland-Altman plot, fully matching the reference style."""
    print(f"Generating Bland-Altman plot (exact style) - {pred_col} vs {true_col}...")
    
    val1 = df[pred_col].to_numpy(dtype=float)
    val2 = df[true_col].to_numpy(dtype=float)

    means = (val1 + val2) / 2
    diffs = val1 - val2

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    # --- Style parameters (from bland-altman_performance.py) ---
    CUSTOM_HEX_COLOR = '#4169E1'
    POINT_ALPHA = 0.3
    POINT_SIZE = 16

    plt.figure(figsize=(10, 7))
    plt.scatter(means, diffs, c=CUSTOM_HEX_COLOR, alpha=POINT_ALPHA, s=POINT_SIZE, edgecolors='none', label='Samples')

    # --- Reference lines and annotations (from the reference file) ---
    plt.axhline(float(mean_diff), color='black', linestyle='-', linewidth=2, alpha=0.8)
    plt.text(x=max(means), y=float(mean_diff), s=f' Mean: {mean_diff:.2f}', va='bottom', ha='left', fontsize=10, weight='bold', color='black')
    plt.axhline(float(upper_loa), color='tab:red', linestyle='--')
    plt.text(x=max(means), y=float(upper_loa), s=f' +1.96 SD: {upper_loa:.2f}', va='bottom', ha='left', fontsize=10, color='tab:red')
    plt.axhline(float(lower_loa), color='tab:red', linestyle='--')
    plt.text(x=max(means), y=float(lower_loa), s=f' -1.96 SD: {lower_loa:.2f}', va='top', ha='left', fontsize=10, color='tab:red')
    plt.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    plt.xlabel('Average of True and Predicted Age (years)', fontsize=14)
    plt.ylabel('Difference (Predicted - True)', fontsize=14)
    plt.xlim(min(means), max(means) * 1.15) # match the X-axis extension scheme of the reference file

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(False)

    prefix = f"{plot_name_prefix}_" if plot_name_prefix else ""
    save_path = os.path.join(output_dir, f"{prefix}bland_altman_plot_exact_style.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Bland-Altman plot (exact style) saved to: {save_path}")

def plot_error_distribution(df: pd.DataFrame, output_dir: str, true_col: str, pred_col: str, plot_name_prefix: str = ""):
    """Plot the error distribution histogram, fully matching the reference style."""
    print(f"Generating error distribution histogram (exact style) - {pred_col} vs {true_col}...")
    
    errors = df[pred_col].to_numpy(dtype=float) - df[true_col].to_numpy(dtype=float)
    mu, sigma = np.mean(errors), np.std(errors)
    mae = np.mean(np.abs(errors))

    # --- Style parameters (from error_distribution_performance.py) ---
    HIST_COLOR = '#4169E1'
    KDE_COLOR = '#000080'
    HIST_ALPHA = 0.6
    BINS_NUMBER = 50

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=BINS_NUMBER, density=True, facecolor=HIST_COLOR, alpha=HIST_ALPHA, edgecolor='white', linewidth=0.5, label='Error Distribution')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, color=KDE_COLOR, linewidth=2, linestyle='-', label='Normal Fit')

    # --- Reference lines and annotations (from the reference file) ---
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero Error')
    plt.axvline(float(mu), color='tab:red', linestyle='-', linewidth=1.5, alpha=0.8, label=f'Mean Bias: {mu:.2f}')
    
    stats_text = f'Mean Bias: {mu:.2f} years\nStd Dev: {sigma:.2f} years\nMAE: {mae:.2f} years'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11, va='top', ha='right', bbox=props)

    plt.xlabel('Age Gap (Predicted - True)', fontsize=14)
    plt.ylabel('Density / Frequency', fontsize=14)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(False)
    plt.legend(loc='upper left', frameon=False, fontsize=11)

    prefix = f"{plot_name_prefix}_" if plot_name_prefix else ""
    save_path = os.path.join(output_dir, f"{prefix}error_distribution_exact_style.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Error distribution plot (exact style) saved to: {save_path}")

def plot_bias_check(df: pd.DataFrame, output_dir: str, true_col: str, pred_col: str, plot_name_prefix: str = ""):
    """Check systematic bias, fully matching the reference style."""
    print(f"Generating bias check plot (exact style) - {pred_col} vs {true_col}...")
    
    true_age = df[true_col].to_numpy(dtype=float)
    delta_age = df[pred_col].to_numpy(dtype=float) - true_age

    corr, _ = pearsonr(true_age, delta_age)
    slope, intercept, _, _, _ = linregress(true_age, delta_age)

    # --- Style parameters (from bias_check_performance.py) ---
    SCATTER_COLOR = '#94070A'
    FIT_LINE_COLOR = '#4169E1'
    POINT_ALPHA = 0.3
    POINT_SIZE = 18

    plt.figure(figsize=(9, 7))
    plt.scatter(true_age, delta_age, c=SCATTER_COLOR, alpha=POINT_ALPHA, s=POINT_SIZE, edgecolors='none', label='Samples')

    # --- Reference lines and annotations (from the reference file) ---
    x_range = np.linspace(true_age.min(), true_age.max(), 100)
    y_fit = slope * x_range + intercept
    plt.plot(x_range, y_fit, color=FIT_LINE_COLOR, linewidth=2, linestyle='-', label=f'Trend (Slope={slope:.3f})')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal Zero Bias')

    stats_text = f'$r$ = {corr:.3f}\nSlope = {slope:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12, va='top', bbox=props)

    plt.xlabel('True Age (Chronological)', fontsize=14)
    plt.ylabel('Delta Age (Predicted - True)', fontsize=14)
    plt.xlim(true_age.min() - 2, true_age.max() + 2)
    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(False)
    plt.legend(loc='upper right', frameon=True, fontsize=11)

    prefix = f"{plot_name_prefix}_" if plot_name_prefix else ""
    save_path = os.path.join(output_dir, f"{prefix}bias_check_plot_exact_style.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Bias check plot (exact style) saved to: {save_path}")
