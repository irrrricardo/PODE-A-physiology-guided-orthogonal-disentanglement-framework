# pode_splitter/prediction_analysis/advanced_plotting.py
"""
Advanced visualization scripts: radar plots and Manhattan plots
Used to analyze the relationship between Age_Delta and physiological indicators.
Updated: adapted to the new physiological subgroup classification
(hemodynamic, metabolic, renal, hematologic, immune)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. New physiological-indicator grouping definition (consistent with the V2 model)
# ============================================

PHYSIO_GROUPS = {
    'hemodynamic': {
        'name': 'Hemodynamic',
        'full_name': 'Hemodynamics',
        'cols': ['SBP', 'DBP'],
        'color': '#4169E1'  # blue
    },
    'metabolic': {
        'name': 'Metabolic',
        'full_name': 'Glucose & Lipid Metabolism',
        'cols': ['BMI', 'FBG', 'HbA1c', 'TG', 'TC', 'LDL-C', 'HDL-C'],
        'color': '#228B22'  # green
    },
    'renal': {
        'name': 'Renal',
        'full_name': 'Kidney Function',
        'cols': ['Creatinine', 'BUN', 'UA', 'Urine_pH', 'USG'],
        'color': '#94070A'  # red
    },
    'hematologic': {
        'name': 'Hematologic',
        'full_name': 'Blood Components',
        'cols': ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW-CV', 'PLT', 'MPV', 'PDW', 'PCT'],
        'color': '#FF8C00'  # orange
    },
    'immune': {
        'name': 'Immune',
        'full_name': 'Immune Inflammation',
        'cols': ['WBC', 'Neutrophil_Count', 'Lymphocyte_Count', 'Monocyte_Count', 'Eosinophil_Count', 'Basophil_Count'],
        'color': '#8B008B'  # purple
    }
}

# Collect all physiological indicator columns
ALL_PHYSIO_COLS = []
for group_name, group_info in PHYSIO_GROUPS.items():
    ALL_PHYSIO_COLS.extend(group_info['cols'])

# Color list (in order)
GROUP_COLORS = [info['color'] for info in PHYSIO_GROUPS.values()]

# Age Delta columns (using new grouping names)
AGE_DELTA_COLS = [
    'Age_Delta_hemodynamic',
    'Age_Delta_metabolic', 
    'Age_Delta_renal',
    'Age_Delta_hematologic',
    'Age_Delta_immune'
]


# ============================================
# 2. Radar chart (Radar Chart)
# ============================================

def plot_radar_correlation(df: pd.DataFrame, output_dir: str, age_delta_col: str, physio_cols: List[str], group_name: str, color: str = '#4169E1'):
    """
    Plot the correlation-coefficient radar chart between Age_Delta and a group of physiological indicators.
    """
    print(f"Generating radar chart - {age_delta_col} vs {group_name}...")
    
    valid_cols = [col for col in physio_cols if col in df.columns]
    if len(valid_cols) < 3:
        print(f"⚠️ {group_name} group has too few valid columns, skipping")
        return
    
    # Compute correlation coefficients
    correlations = []
    for col in valid_cols:
        valid_mask = ~(df[age_delta_col].isna() | df[col].isna())
        if valid_mask.sum() > 10:
            corr, _ = stats.pearsonr(df.loc[valid_mask, age_delta_col], df.loc[valid_mask, col])
            correlations.append(abs(corr))  # use absolute values
        else:
            correlations.append(0)
    
    if not correlations:
        return
    
    # Normalize to 0-1
    max_corr = max(correlations) if max(correlations) > 0 else 1
    normalized = [c / max_corr for c in correlations]
    
    # Radar chart settings
    n_vars = len(valid_cols)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop
    
    normalized += normalized[:1]  # close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot the radar chart
    ax.fill(angles, normalized, color=color, alpha=0.25)
    ax.plot(angles, normalized, color=color, linewidth=2, marker='o', markersize=8)
    
    # Add value labels
    for i, (angle, val, col) in enumerate(zip(angles[:-1], normalized[:-1], valid_cols)):
        ax.annotate(f'{val:.2f}', xy=(angle, val), xytext=(angle, val + 0.08),
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(valid_cols, fontsize=10)
    
    # Set Y axis
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
    
    ax.set_title(f'{age_delta_col}\nCorrelation with {group_name}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"radar_{age_delta_col}_{group_name.lower()}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Radar chart saved to: {save_path}")


def plot_radar_comparison(df: pd.DataFrame, output_dir: str, age_delta_cols: List[str], physio_groups: Dict):
    """
    Plot the integrated radar comparison chart for multiple Age_Delta values.
    """
    print("Generating integrated radar comparison chart...")
    
    n_groups = len(physio_groups)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    colors = ['#4169E1', '#94070A', '#228B22', '#FF8C00', '#8B008B']
    
    for idx, (delta_col, color) in enumerate(zip(age_delta_cols, colors)):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        group_data = []
        for group_name, group_info in physio_groups.items():
            valid_cols = [col for col in group_info['cols'] if col in df.columns]
            if len(valid_cols) < 3:
                continue
            
            correlations = []
            for col in valid_cols:
                valid_mask = ~(df[delta_col].isna() | df[col].isna())
                if valid_mask.sum() > 10:
                    corr, _ = stats.pearsonr(df.loc[valid_mask, delta_col], df.loc[valid_mask, col])
                    correlations.append(abs(corr))
            
            if correlations:
                group_data.append((group_info['name'], np.mean(correlations)))
        
        if not group_data:
            ax.set_visible(False)
            continue
        
        names = [g[0] for g in group_data]
        values = [g[1] for g in group_data]
        
        # Normalize
        max_val = max(values) if max(values) > 0 else 1
        normalized = [v / max_val for v in values]
        
        # Radar chart
        n_vars = len(names)
        angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
        angles += angles[:1]
        normalized += normalized[:1]
        
        ax.fill(angles, normalized, color=color, alpha=0.2)
        ax.plot(angles, normalized, color=color, linewidth=2, marker='o', markersize=8)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylim(0, 1.2)
        ax.set_title(delta_col, fontsize=13, fontweight='bold', pad=15)
    
    # Hide redundant subplots
    for idx in range(len(age_delta_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Age Delta Correlation with Physiological Groups (New Classification)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "radar_comparison_all_new.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Integrated radar comparison chart saved to: {save_path}")


# ============================================
# 3. Manhattan plot (Manhattan Plot)
# ============================================

def plot_manhattan_correlation(df: pd.DataFrame, output_dir: str, age_delta_col: str, all_physio_cols: List[str], physio_groups: Dict):
    """
    Plot the correlation-coefficient Manhattan plot of Age_Delta vs all physiological indicators
    (colored by the new grouping).
    """
    print(f"Generating Manhattan plot - {age_delta_col}...")
    
    # Compute the correlation coefficient and P-value for each indicator
    results = []
    for col in all_physio_cols:
        if col not in df.columns:
            continue
        
        valid_mask = ~(df[age_delta_col].isna() | df[col].isna())
        n_valid = valid_mask.sum()
        
        if n_valid > 10:
            corr, pval = stats.pearsonr(df.loc[valid_mask, age_delta_col], df.loc[valid_mask, col])
            results.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'pvalue': pval,
                '-log10(pvalue)': -np.log10(pval + 1e-300) if pval > 0 else 50
            })
    
    if not results:
        print(f"⚠️ No valid correlation data")
        return
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('abs_correlation', ascending=False)
    
    # Create position index
    results_df['position'] = range(len(results_df))
    
    # Assign colors by group
    def get_group_color(feature):
        for group_name, group_info in physio_groups.items():
            if feature in group_info['cols']:
                return group_info['color']
        return '#808080'  # gray as default
    
    results_df['color'] = results_df['feature'].apply(get_group_color)
    
    # Significance threshold line (-log10(0.05) ≈ 1.3)
    significance_threshold = -np.log10(0.05)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1], sharex=True)
    
    # Top: correlation-coefficient bar chart
    colors_bar = results_df['color'].tolist()
    bars = ax1.bar(results_df['position'], results_df['correlation'], color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axhline(y=-0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_title(f'Manhattan Plot: {age_delta_col} vs Physiological Indicators\n(New Classification: hemodynamic, metabolic, renal, hematologic, immune)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.5, 0.5)
    
    # Add value labels
    for i, (pos, corr) in enumerate(zip(results_df['position'], results_df['correlation'])):
        if abs(corr) > 0.15:
            ax1.annotate(f'{corr:.2f}', xy=(pos, corr), xytext=(0, 5 if corr > 0 else -10),
                        textcoords='offset points', ha='center', fontsize=7, rotation=45)
    
    # Bottom: -log10(pvalue)
    ax2.bar(results_df['position'], results_df['-log10(pvalue)'], color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=significance_threshold, color='red', linestyle='--', linewidth=1.5, label=f'p=0.05')
    ax2.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax2.set_xlabel('Physiological Indicators', fontsize=12)
    ax2.set_xticks(results_df['position'])
    ax2.set_xticklabels(results_df['feature'], rotation=45, ha='right', fontsize=8)
    
    # Add group separators and labels
    current_idx = 0
    for group_name, group_info in physio_groups.items():
        group_cols = [col for col in group_info['cols'] if col in results_df['feature'].values]
        if group_cols:
            ax1.axvline(x=current_idx - 0.5, color='gray', linestyle='--', alpha=0.3)
            ax2.axvline(x=current_idx - 0.5, color='gray', linestyle='--', alpha=0.3)
            ax1.text(current_idx, ax1.get_ylim()[1] * 0.9, group_name, ha='center', va='top', 
                    fontsize=10, fontweight='bold', color=group_info['color'])
        current_idx += len(group_cols)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"manhattan_{age_delta_col}_new.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Manhattan plot saved to: {save_path}")
    
    # Save the statistical results to a CSV
    csv_path = os.path.join(output_dir, f"correlation_{age_delta_col}_new.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"📊 Correlation coefficient data saved to: {csv_path}")


def plot_manhattan_heatmap(df: pd.DataFrame, output_dir: str, age_delta_cols: List[str], all_physio_cols: List[str], physio_groups: Dict):
    """
    Plot the correlation-coefficient heatmap (Manhattan style) between multiple Age_Delta values
    and all physiological indicators.
    """
    print("Generating Manhattan heatmap...")
    
    # Compute the correlation matrix
    corr_matrix = np.zeros((len(age_delta_cols), len(all_physio_cols)))
    pval_matrix = np.zeros((len(age_delta_cols), len(all_physio_cols)))
    
    for i, delta_col in enumerate(age_delta_cols):
        for j, physio_col in enumerate(all_physio_cols):
            if physio_col not in df.columns:
                corr_matrix[i, j] = np.nan
                continue
            
            valid_mask = ~(df[delta_col].isna() | df[physio_col].isna())
            if valid_mask.sum() > 10:
                corr, pval = stats.pearsonr(df.loc[valid_mask, delta_col], df.loc[valid_mask, physio_col])
                corr_matrix[i, j] = corr
                pval_matrix[i, j] = pval
            else:
                corr_matrix[i, j] = np.nan
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('custom_diverging', ['#94070A', '#FFFFFF', '#4169E1'])
    
    # Plot
    im = ax.imshow(corr_matrix, cmap=cmap, aspect='auto', vmin=-0.5, vmax=0.5)
    
    # Set ticks
    ax.set_xticks(range(len(all_physio_cols)))
    ax.set_xticklabels(all_physio_cols, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(age_delta_cols)))
    ax.set_yticklabels(age_delta_cols, fontsize=10)
    
    # Add values
    for i in range(len(age_delta_cols)):
        for j in range(len(all_physio_cols)):
            if not np.isnan(corr_matrix[i, j]):
                text_color = 'white' if abs(corr_matrix[i, j]) > 0.25 else 'black'
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', 
                       fontsize=7, color=text_color)
    
    # Add significance markers
    for i in range(len(age_delta_cols)):
        for j in range(len(all_physio_cols)):
            if not np.isnan(pval_matrix[i, j]) and pval_matrix[i, j] < 0.05:
                ax.text(j, i, '*', ha='center', va='center', fontsize=12, color='black', fontweight='bold')
    
    # Add group separators
    current_idx = 0
    for group_name, group_info in physio_groups.items():
        group_cols = [col for col in group_info['cols'] if col in all_physio_cols]
        if group_cols:
            ax.axvline(x=current_idx + len(group_cols) - 0.5, color='gray', linestyle='-', linewidth=2, alpha=0.5)
        current_idx += len(group_cols)
    
    # Color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson Correlation', fontsize=11)
    
    plt.title('Age_Delta vs Physiological Indicators Correlation Matrix (New Classification)\n(* p < 0.05)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Physiological Indicators', fontsize=12)
    plt.ylabel('Age_Delta', fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "manhattan_heatmap_all_new.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Manhattan heatmap saved to: {save_path}")


# ============================================
# 4. Statistical summary function
# ============================================

def generate_summary_stats(df: pd.DataFrame, output_dir: str, age_delta_cols: List[str], all_physio_cols: List[str]):
    """
    Generate a statistical summary report.
    """
    print("Generating statistical summary...")
    
    summary_data = []
    
    for delta_col in age_delta_cols:
        for physio_col in all_physio_cols:
            if physio_col not in df.columns:
                continue
            
            valid_mask = ~(df[delta_col].isna() | df[physio_col].isna())
            n = valid_mask.sum()
            
            if n > 10:
                corr, pval = stats.pearsonr(df.loc[valid_mask, delta_col], df.loc[valid_mask, physio_col])
                summary_data.append({
                    'Age_Delta': delta_col,
                    'Physio_Indicator': physio_col,
                    'N': n,
                    'Pearson_r': corr,
                    'P_value': pval,
                    'Significant': 'Yes' if pval < 0.05 else 'No',
                    'R_squared': corr ** 2
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['Age_Delta', 'Pearson_r'], ascending=[True, False])
    
    csv_path = os.path.join(output_dir, "correlation_summary_new.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"📊 Statistical summary saved to: {csv_path}")
    
    return summary_df


# ============================================
# 5. Main function
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Radar and Manhattan plot analysis of Age_Delta vs physiological indicators (new grouping)")
    
    # --- Data arguments ---
    parser.add_argument('--data_path', type=str, required=True, help="Path to the prediction-result table (.xlsx)")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory")
    
    # --- Analysis options ---
    parser.add_argument('--plot-radar', action='store_true', help="Generate radar charts")
    parser.add_argument('--plot-manhattan', action='store_true', help="Generate Manhattan plots")
    parser.add_argument('--plot-heatmap', action='store_true', help="Generate the correlation heatmap")
    parser.add_argument('--plot-all', action='store_true', help="Generate all charts")
    
    # --- Specify the Age_Delta columns to analyze ---
    parser.add_argument('--age-delta-cols', type=str, default=",".join(AGE_DELTA_COLS),
                       help="Age_Delta columns to analyze, comma separated")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data: {args.data_path}")
    df = pd.read_excel(args.data_path)
    print(f"Data shape: {df.shape}")
    
    # Parse the Age_Delta columns
    age_delta_cols = [col.strip() for col in args.age_delta_cols.split(',') if col.strip() in df.columns]
    if not age_delta_cols:
        age_delta_cols = [col for col in AGE_DELTA_COLS if col in df.columns]
    print(f"Age_Delta columns to analyze: {age_delta_cols}")
    
    # Collect all physiological-indicator columns
    all_physio_cols = [col for col in ALL_PHYSIO_COLS if col in df.columns]
    print(f"Number of physiological indicators to analyze: {len(all_physio_cols)}")
    print(f"Physiological-indicator groups: {list(PHYSIO_GROUPS.keys())}")
    
    # Generate the statistical summary
    summary_df = generate_summary_stats(df, args.output_dir, age_delta_cols, all_physio_cols)
    
    # Generate the charts
    if args.plot_all or args.plot_radar:
        # Individual radar charts
        for delta_col in age_delta_cols:
            for group_name, group_info in PHYSIO_GROUPS.items():
                plot_radar_correlation(df, args.output_dir, delta_col, group_info['cols'], group_info['name'], group_info['color'])
        
        # Integrated radar comparison chart
        plot_radar_comparison(df, args.output_dir, age_delta_cols, PHYSIO_GROUPS)
    
    if args.plot_all or args.plot_manhattan:
        # Individual Manhattan plots
        for delta_col in age_delta_cols:
            plot_manhattan_correlation(df, args.output_dir, delta_col, all_physio_cols, PHYSIO_GROUPS)
    
    if args.plot_all or args.plot_heatmap:
        # Manhattan heatmap
        plot_manhattan_heatmap(df, args.output_dir, age_delta_cols, all_physio_cols, PHYSIO_GROUPS)
    
    print("\n✅ Analysis complete!")
    print("\n📊 Output files:")
    print(f"  - Radar charts: {os.path.join(args.output_dir, 'radar_*.png')}")
    print(f"  - Manhattan plots: {os.path.join(args.output_dir, 'manhattan_*.png')}")
    print(f"  - Heatmap: {os.path.join(args.output_dir, 'manhattan_heatmap_all_new.png')}")
    print(f"  - Statistical summary: {os.path.join(args.output_dir, 'correlation_summary_new.csv')}")


if __name__ == '__main__':
    main()
