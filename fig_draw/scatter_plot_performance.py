# scatter_plot_performance.py - Scatter Plot Performance Visualization

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Scatter plot for prediction performance")
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV file')
    parser.add_argument('--output_dir', type=str, default='./scatter', help='Output directory')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from {args.predictions}")
    df = pd.read_csv(args.predictions)
    
    y_true = df['True_Age'].values
    y_pred = df['Predicted_Age'].values
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Basic scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20, c='steelblue')
    min_val = min(y_true.min(), y_pred.min()) - 5
    max_val = max(y_true.max(), y_pred.max()) + 5
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    
    # Linear fit
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    axes[0].plot([min_val, max_val], [p(min_val), p(max_val)], 'g-', linewidth=2, label=f'Linear Fit (y={z[0]:.2f}x+{z[1]:.2f})')
    
    axes[0].set_xlabel('True Age (years)', fontsize=14)
    axes[0].set_ylabel('Predicted Age (years)', fontsize=14)
    axes[0].set_title('Predicted vs True Age', fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(min_val, max_val)
    axes[0].set_ylim(min_val, max_val)
    
    # 2. Hexbin plot for density
    hb = axes[1].hexbin(y_true, y_pred, gridsize=30, cmap='YlOrRd', mincnt=1)
    axes[1].plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('True Age (years)', fontsize=14)
    axes[1].set_ylabel('Predicted Age (years)', fontsize=14)
    axes[1].set_title('Prediction Density (Hexbin)', fontsize=16)
    axes[1].legend(fontsize=12)
    plt.colorbar(hb, ax=axes[1], label='Count')
    
    plt.tight_layout()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = f"{args.output_dir}/scatter_plot_performance.png"
    plt.savefig(save_path, dpi=300)
    print(f"Scatter plot saved to: {save_path}")
    
    # Calculate correlation metrics
    from scipy.stats import pearsonr, spearmanr
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    print("\n--- Scatter Plot Statistics ---")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")
    print(f"Linear Fit: y = {z[0]:.4f}x + {z[1]:.4f}")

if __name__ == '__main__':
    main()