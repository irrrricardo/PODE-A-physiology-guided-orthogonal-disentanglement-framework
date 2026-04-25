# bias_check_performance.py - Bias Check Performance Visualization

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Check and visualize prediction bias")
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV file')
    parser.add_argument('--output_dir', type=str, default='./bias_check', help='Output directory')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from {args.predictions}")
    df = pd.read_csv(args.predictions)
    
    y_true = df['True_Age'].values
    y_pred = df['Predicted_Age'].values
    errors = y_pred - y_true
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Error distribution by age
    axes[0].scatter(y_true, errors, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    mean_error = np.mean(errors)
    axes[0].axhline(y=mean_error, color='green', linestyle='-', linewidth=1.5, label=f'Mean Error: {mean_error:.2f}')
    axes[0].set_xlabel('True Age (years)')
    axes[0].set_ylabel('Prediction Error (years)')
    axes[0].set_title('Prediction Error vs True Age')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Error histogram
    axes[1].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].axvline(x=mean_error, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_error:.2f}')
    axes[1].set_xlabel('Prediction Error (years)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].legend()
    
    # 3. Error by age group
    age_bins = pd.cut(y_true, bins=range(0, 101, 10), right=False)
    bin_errors = pd.DataFrame({'age_group': age_bins, 'error': errors}).groupby('age_group', observed=False)['error'].agg(['mean', 'std'])
    bin_errors.plot(kind='bar', y='mean', yerr='std', ax=axes[2], color='mediumseagreen', edgecolor='black', capsize=3)
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    axes[2].set_xlabel('Age Group')
    axes[2].set_ylabel('Mean Error (years)')
    axes[2].set_title('Mean Error by Age Group (with Std)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = f"{args.output_dir}/bias_check_performance.png"
    plt.savefig(save_path, dpi=300)
    print(f"Bias check plot saved to: {save_path}")
    
    # Print statistics
    print("\n--- Bias Check Statistics ---")
    print(f"Mean Error: {mean_error:.4f} years")
    print(f"Std of Error: {np.std(errors):.4f} years")
    print(f"Max Over-prediction: {np.max(errors):.4f} years")
    print(f"Max Under-prediction: {np.min(errors):.4f} years")

if __name__ == '__main__':
    main()