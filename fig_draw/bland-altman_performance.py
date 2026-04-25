# bland-altman_performance.py - Bland-Altman Performance Visualization

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Bland-Altman plot for method comparison")
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV file')
    parser.add_argument('--output_dir', type=str, default='./bland_altman', help='Output directory')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from {args.predictions}")
    df = pd.read_csv(args.predictions)
    
    y_true = df['True_Age'].values
    y_pred = df['Predicted_Age'].values
    
    # Calculate Bland-Altman metrics
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(mean_vals, diff_vals, alpha=0.5, s=20, c='steelblue')
    
    # Mean line
    ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean difference: {mean_diff:.2f}')
    
    # Limits of agreement (95% CI)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    ax.axhline(upper_loa, color='blue', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {upper_loa:.2f}')
    ax.axhline(lower_loa, color='blue', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {lower_loa:.2f}')
    
    # Zero line
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Average Age (years)', fontsize=14)
    ax.set_ylabel('Difference in Age (Predicted - True)', fontsize=14)
    ax.set_title('Bland-Altman Plot', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = f"{args.output_dir}/bland_altman_performance.png"
    plt.savefig(save_path, dpi=300)
    print(f"Bland-Altman plot saved to: {save_path}")
    
    # Print statistics
    print("\n--- Bland-Altman Statistics ---")
    print(f"Mean difference: {mean_diff:.4f} years")
    print(f"Standard deviation: {std_diff:.4f} years")
    print(f"Upper limit of agreement (+1.96 SD): {upper_loa:.4f} years")
    print(f"Lower limit of agreement (-1.96 SD): {lower_loa:.4f} years")
    print(f"95% of predictions fall within ±{1.96*std_diff:.2f} years")

if __name__ == '__main__':
    main()