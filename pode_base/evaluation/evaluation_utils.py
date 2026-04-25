# evaluation_utils.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate and print regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Calculate Pearson and Spearman correlation coefficients
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    print("\n" + "=" * 60)
    print("Regression Metrics Summary")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE):       {mae:.4f}")
    print(f"Mean Squared Error (MSE):        {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.4f}")
    print(f"R-squared (R²):                  {r2:.4f}")
    print("-" * 60)
    print(f"Pearson Correlation:             {pearson_corr:.4f}")
    print(f"Spearman Correlation:           {spearman_corr:.4f}")
    print("=" * 60)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Pearson': pearson_corr,
        'Spearman': spearman_corr
    }


def plot_scatter(y_true, y_pred, save_path):
    """
    Draw scatter plot and linear regression fit line.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)

    # Add diagonal line y=x (representing perfect prediction)
    min_val = min(y_true.min(), y_pred.min()) - 5
    max_val = max(y_true.max(), y_pred.max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')

    # Add linear regression fit line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot([min_val, max_val], [p(min_val), p(max_val)], 'g-', linewidth=2, label=f'Linear Fit (y={z[0]:.2f}x+{z[1]:.2f})')

    plt.xlabel('True Age (years)', fontsize=14)
    plt.ylabel('Predicted Age (years)', fontsize=14)
    plt.title('Predicted vs. True Age', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Set axis limits with some margin
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Scatter plot saved to: {save_path}")
    plt.close()


def plot_residual(y_true, y_pred, save_path):
    """
    Draw residual plot (Residual = Predicted - True).
    Positive value indicates predicted age > true age.
    """
    residuals = y_pred - y_true

    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

    plt.xlabel('True Age (years)', fontsize=14)
    plt.ylabel('Residual (Predicted - True)', fontsize=14)
    plt.title('Residual Plot', fontsize=16)
    plt.grid(True, alpha=0.3)

    # Add mean and std lines
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    plt.axhline(y=mean_res, color='green', linestyle='-', linewidth=1, label=f'Mean: {mean_res:.2f}')
    plt.axhline(y=mean_res + 2 * std_res, color='orange', linestyle=':', linewidth=1, label=f'+2 Std: {mean_res + 2 * std_res:.2f}')
    plt.axhline(y=mean_res - 2 * std_res, color='orange', linestyle=':', linewidth=1, label=f'-2 Std: {mean_res - 2 * std_res:.2f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Residual plot saved to: {save_path}")
    plt.close()


def plot_age_gap_histogram(y_true, y_pred, save_path):
    """
    Draw histogram of absolute age prediction error (|Predicted - True|).
    """
    abs_errors = np.abs(y_pred - y_true)

    plt.figure(figsize=(10, 6))
    plt.hist(abs_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)

    mean_error = np.mean(abs_errors)
    median_error = np.median(abs_errors)
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f} years')
    plt.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f} years')

    plt.xlabel('Absolute Error (years)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Absolute Prediction Errors', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Age gap histogram saved to: {save_path}")
    plt.close()


def plot_bland_altman(y_true, y_pred, save_path):
    """
    Draw Bland-Altman plot to assess agreement between two methods.
    """
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)

    plt.figure(figsize=(10, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.5, s=20)

    # Horizontal lines: mean difference and 95% limits of agreement (LoA)
    plt.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean difference: {mean_diff:.2f}')
    plt.axhline(mean_diff + 1.96 * std_diff, color='blue', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}')
    plt.axhline(mean_diff - 1.96 * std_diff, color='blue', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}')

    plt.xlabel('Average Age (years)', fontsize=14)
    plt.ylabel('Difference in Age (Predicted - True)', fontsize=14)
    plt.title('Bland-Altman Plot', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Bland-Altman plot saved to: {save_path}")
    plt.close()