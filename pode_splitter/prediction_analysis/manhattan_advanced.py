# pode_splitter/prediction_analysis/manhattan_advanced.py
"""
Advanced Manhattan plot - Mutually Exclusive Control version
Updated: adapted to the new physiological subgroup classification
(hemodynamic, metabolic, renal, hematologic, immune)

- X axis: all physiological indicators (grouped by system)
- Y axis: partial correlation coefficient, controlling for age and the other components
- Mutually exclusive control: when computing hemodynamic, control for [Age + the other components];
                              when computing metabolic, control for [Age + the other components], etc.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Optional
from statsmodels.stats.multitest import multipletests
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ============================================
# 1. New physiological-indicator grouping (by system) - consistent with the V2 model
# ============================================

PHYSIO_GROUPS_ORDERED = [
    ('Hemodynamic', ['SBP', 'DBP']),
    ('Metabolic', ['BMI', 'FBG', 'TG', 'TC', 'LDL-C', 'HDL-C']),
    ('Renal', ['Creatinine', 'BUN', 'UA', 'Urine_pH', 'USG']),
    ('Hematologic', ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW-CV', 'PLT', 'MPV', 'PDW', 'PCT']),
    ('Immune', ['WBC', 'Neutrophil_Count', 'Lymphocyte_Count', 'Monocyte_Count', 'Eosinophil_Count', 'Basophil_Count']),
]

ALL_PHYSIO_COLS = []
for group_name, cols in PHYSIO_GROUPS_ORDERED:
    ALL_PHYSIO_COLS.extend(cols)

# NPG (Nature Publishing Group) style palette - low saturation, "expensive" feeling
NATURE_PALETTE = {
    'Hemodynamic': '#4C72B0',  # steady blue
    'Metabolic':   '#55A868',  # forest green
    'Renal':       '#C44E52',  # brick red
    'Hematologic': '#DD8452',  # terracotta orange
    'Immune':      '#8172B3'   # elegant purple
}

# Old-version palette (kept for backwards compatibility)
GROUP_COLORS = {
    'Hemodynamic': '#4169E1',      # blue
    'Metabolic': '#228B22',        # green
    'Renal': '#94070A',            # red
    'Hematologic': '#FF8C00',      # orange
    'Immune': '#8B008B'            # purple
}

# Group mapping (indicator -> group name)
GROUP_INDICATORS = {}
for group_name, cols in PHYSIO_GROUPS_ORDERED:
    for col in cols:
        GROUP_INDICATORS[col] = group_name


# ============================================
# 0. Nature-style global configuration (Global Style Config)
# ============================================

def set_nature_style():
    """Configure matplotlib to comply with Nature publication standards."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],  # prefer Arial
        'font.size': 7,                # body default 7pt (Nature standard)
        'axes.labelsize': 8,           # axis label 8pt
        'axes.titlesize': 9,           # title 9pt (bold)
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.5,         # lines should be thin and refined
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'lines.linewidth': 1,
        'figure.dpi': 300,             # clear on screen
        'savefig.dpi': 600,            # print-grade resolution
        'svg.fonttype': 'none',        # vector-graphic fonts editable
        'pdf.fonttype': 42             # embed fonts
    })


# ============================================
# 2. Multivariate partial correlation function
# ============================================

def multi_partial_correlation(df, x_col, y_col, control_cols):
    """
    Compute the multivariate partial correlation coefficient: the association between
    x_col and y_col after controlling for multiple covariates.
    
    Args:
        df: data frame
        x_col: target variable X
        y_col: target variable Y
        control_cols: list of covariates to control for
    
    Returns: (partial_corr, p_value, n_samples)
    """
    # Get all valid data (non-NaN)
    cols_needed = [x_col, y_col] + control_cols
    mask = pd.Series(True, index=df.index)
    for col in cols_needed:
        if col in df.columns:
            mask = mask & ~df[col].isna()
        else:
            print(f"  ⚠️ Warning: column '{col}' does not exist, skipping")
            return 0.0, 1.0, 0
    
    n = mask.sum()
    if n < 10:
        return 0.0, 1.0, n
    
    # Extract data
    x = df.loc[mask, x_col].values.astype(float)
    y = df.loc[mask, y_col].values.astype(float)
    z = df.loc[mask, control_cols].values.astype(float)
    
    # Step 1: predict x using all covariates and obtain the residuals rx
    try:
        from numpy.linalg import lstsq
        Z = np.column_stack([np.ones(len(z)), z])
        coeffs_x, _, _, _ = lstsq(Z, x, rcond=None)
        rx = x - Z @ coeffs_x
    except Exception:
        rx = x - np.mean(x)
    
    # Step 2: predict y using all covariates and obtain the residuals ry
    try:
        coeffs_y, _, _, _ = lstsq(Z, y, rcond=None)
        ry = y - Z @ coeffs_y
    except Exception:
        ry = y - np.mean(y)
    
    # Step 3: compute the correlation coefficient between rx and ry
    if np.std(rx) < 1e-8 or np.std(ry) < 1e-8:
        return 0.0, 1.0, n
    
    corr, pval = stats.pearsonr(rx, ry)
    return float(corr), float(pval), n


def simple_correlation(df, x_col, y_col):
    """Simple correlation analysis (no control variables)."""
    mask = ~(df[x_col].isna() | df[y_col].isna())
    n = mask.sum()
    
    if n < 10:
        return 0.0, 1.0, n
    
    x = df.loc[mask, x_col].values.astype(float)
    y = df.loc[mask, y_col].values.astype(float)
    
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0, 1.0, n
    
    corr, pval = stats.pearsonr(x, y)
    return float(corr), float(pval), n


# ============================================
# 3. Advanced Manhattan plot - Mutually Exclusive Control version
# ============================================

def plot_mutually_exclusive_manhattan(
    df: pd.DataFrame,
    output_dir: str,
    top_delta_col: str,
    bottom_delta_col: str,
    age_col: str,
    other_delta_cols: List[str],  # all other components
    metric: str = 'neg_log_pvalue',
    control_hgb: bool = False,
    top_color: str = '#4169E1',
    bottom_color: str = '#94070A',
):
    """
    Plot the mutually-exclusive-control Manhattan plot.
    
    When computing hemodynamic: control for [Age, metabolic, renal, hematologic, immune]
    When computing immune: control for [Age, hemodynamic, metabolic, renal, hematologic]
    """
    control_note = f"Mutually Exclusive Control (Age + {len(other_delta_cols)} other components)"
    if control_hgb:
        control_note += " + HGB"
    
    print(f"Generating mutually-exclusive-control Manhattan plot...")
    print(f"  {top_delta_col}: control [Age, {', '.join(other_delta_cols)}]" + (" + HGB" if control_hgb else ""))
    print(f"  {bottom_delta_col}: control [Age, {', '.join(other_delta_cols)}]" + (" + HGB" if control_hgb else ""))
    
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[1, 1])
    
    # Build the list of covariates
    def get_control_covars(exclude_col):
        covars = [age_col] + other_delta_cols
        if exclude_col in covars:
            covars.remove(exclude_col)
        if control_hgb and 'HGB' in df.columns:
            covars.append('HGB')
        return covars
    
    # Compute the data for the top plot (control for Age + the other components)
    top_covars = get_control_covars(top_delta_col)
    top_data = []
    top_pvals = []
    top_ns = []
    
    for col in ALL_PHYSIO_COLS:
        if col not in df.columns:
            top_data.append(0)
            top_pvals.append(1.0)
            top_ns.append(0)
            continue
        
        corr, pval, n = multi_partial_correlation(df, top_delta_col, col, top_covars)
        
        if metric == 'neg_log_pvalue':
            value = -np.log10(max(pval, 1e-300))
        else:
            value = abs(corr)
        
        top_data.append(value)
        top_pvals.append(pval)
        top_ns.append(n)
    
    # Compute the data for the bottom plot (control for Age + the other components)
    bottom_covars = get_control_covars(bottom_delta_col)
    bottom_data = []
    bottom_pvals = []
    bottom_ns = []
    
    for col in ALL_PHYSIO_COLS:
        if col not in df.columns:
            bottom_data.append(0)
            bottom_pvals.append(1.0)
            bottom_ns.append(0)
            continue
        
        corr, pval, n = multi_partial_correlation(df, bottom_delta_col, col, bottom_covars)
        
        if metric == 'neg_log_pvalue':
            value = -np.log10(max(pval, 1e-300))
        else:
            value = abs(corr)
        
        bottom_data.append(value)
        bottom_pvals.append(pval)
        bottom_ns.append(n)
    
    x_range = list(range(len(ALL_PHYSIO_COLS)))
    
    # Group color helper
    def get_color(feature):
        for group_name, cols in PHYSIO_GROUPS_ORDERED:
            if feature in cols:
                return GROUP_COLORS.get(group_name, '#808080')
        return '#808080'
    
    top_colors = [get_color(col) for col in ALL_PHYSIO_COLS]
    bottom_colors = [get_color(col) for col in ALL_PHYSIO_COLS]
    
    # Plot top
    ax_top.bar(x_range, top_data, color=top_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    for i, pval in enumerate(top_pvals):
        if pval < 0.001:
            ax_top.annotate('***', xy=(i, top_data[i]), xytext=(0, 3),
                           textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color='red')
        elif pval < 0.01:
            ax_top.annotate('**', xy=(i, top_data[i]), xytext=(0, 3),
                           textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color='orange')
        elif pval < 0.05:
            ax_top.annotate('*', xy=(i, top_data[i]), xytext=(0, 3),
                           textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color='green')
    
    if metric == 'neg_log_pvalue':
        ax_top.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='p=0.05')
        ax_top.axhline(y=-np.log10(0.01), color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='p=0.01')
        ax_top.axhline(y=-np.log10(0.001), color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='p=0.001')
    
    ax_top.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax_top.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax_top.set_title(f'{top_delta_col}\n(Controlled: Age + {len(other_delta_cols)} other components' + 
                     (' + HGB' if control_hgb else '') + ')', fontsize=12, fontweight='bold', 
                     color=top_color, pad=10)
    ax_top.set_xlim(-0.5, len(ALL_PHYSIO_COLS) - 0.5)
    ax_top.set_xticks([])
    ax_top.spines['bottom'].set_visible(False)
    ax_top.legend(loc='upper right', fontsize=8)
    
    # Plot bottom
    bottom_data_neg = [-v for v in bottom_data]
    ax_bottom.bar(x_range, bottom_data_neg, color=bottom_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    for i, pval in enumerate(bottom_pvals):
        if pval < 0.001:
            ax_bottom.annotate('***', xy=(i, bottom_data_neg[i]), xytext=(0, -3),
                               textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color='red')
        elif pval < 0.01:
            ax_bottom.annotate('**', xy=(i, bottom_data_neg[i]), xytext=(0, -3),
                               textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color='orange')
        elif pval < 0.05:
            ax_bottom.annotate('*', xy=(i, bottom_data_neg[i]), xytext=(0, -3),
                               textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color='green')
    
    if metric == 'neg_log_pvalue':
        ax_bottom.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_bottom.axhline(y=-np.log10(0.01), color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_bottom.axhline(y=-np.log10(0.001), color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax_bottom.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax_bottom.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax_bottom.set_title(f'{bottom_delta_col}\n(Controlled: Age + {len(other_delta_cols)} other components' + 
                        (' + HGB' if control_hgb else '') + ')', fontsize=12, fontweight='bold', 
                        color=bottom_color, pad=10)
    ax_bottom.set_xlim(-0.5, len(ALL_PHYSIO_COLS) - 0.5)
    ax_bottom.set_xticks(x_range)
    ax_bottom.set_xticklabels(ALL_PHYSIO_COLS, rotation=45, ha='right', fontsize=9)
    ax_bottom.spines['top'].set_visible(False)
    
    # Add group separators and labels
    current_idx = 0
    for group_name, cols in PHYSIO_GROUPS_ORDERED:
        group_cols = [col for col in cols if col in df.columns]
        if group_cols:
            ax_top.axvline(x=current_idx - 0.5, color='gray', linestyle='--', alpha=0.3)
            ax_bottom.axvline(x=current_idx - 0.5, color='gray', linestyle='--', alpha=0.3)
            ax_top.text(current_idx, ax_top.get_ylim()[1] * 0.9, group_name, ha='center', va='top', 
                       fontsize=10, fontweight='bold', color=GROUP_COLORS.get(group_name, 'gray'))
        current_idx += len(group_cols)
    
    metric_label = "-log₁₀(P-value)" if metric == 'neg_log_pvalue' else "|r|"
    hgb_note = " (+ HGB)" if control_hgb else ""
    fig.suptitle(f'Mutually Exclusive Control Analysis (New Classification){hgb_note}\n{metric_label} for Specific Component Associations', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    hgb_suffix = "_with_HGB" if control_hgb else ""
    filename = f"manhattan_mutually_exclusive_{top_delta_col}_vs_{bottom_delta_col}{hgb_suffix}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Manhattan plot saved to: {save_path}")


def generate_mutually_exclusive_table(df: pd.DataFrame, output_dir: str, 
                                       age_delta_cols: List[str], age_col: str,
                                       control_hgb: bool = False):
    """Generate the mutually-exclusive-control correlation coefficient table."""
    print(f"Generating mutually-exclusive-control correlation table...")
    
    results = []
    
    for delta_col in age_delta_cols:
        # Determine the "other" components (all components except the current one)
        other_cols = [c for c in age_delta_cols if c != delta_col]
        
        # Build the list of covariates
        covars = [age_col] + other_cols
        if control_hgb and 'HGB' in df.columns:
            covars.append('HGB')
        
        for physio_col in ALL_PHYSIO_COLS:
            if physio_col not in df.columns:
                continue
            
            corr, pval, n = multi_partial_correlation(df, delta_col, physio_col, covars)
            
            results.append({
                'Target': delta_col,
                'Controlled_Variables': '+'.join(covars),
                'Physio_Indicator': physio_col,
                'N': n,
                'Partial_r': corr,
                'P_value': pval,
                'Neg_log10_pvalue': -np.log10(max(pval, 1e-300)),
                'Significant_0.05': pval < 0.05,
                'Significant_0.01': pval < 0.01,
                'Significant_0.001': pval < 0.001,
            })
    
    df_result = pd.DataFrame(results)
    hgb_suffix = "_with_HGB" if control_hgb else ""
    csv_path = os.path.join(output_dir, f"correlation_table_mutually_exclusive{hgb_suffix}_new.csv")
    df_result.to_csv(csv_path, index=False)
    print(f"📊 Correlation coefficient table saved to: {csv_path}")
    
    return df_result


# Color configuration for Age Delta components
DELTA_COLORS = {
    'Age_Delta_hemodynamic': '#4169E1',   # blue
    'Age_Delta_metabolic': '#228B22',      # green
    'Age_Delta_renal': '#94070A',          # red
    'Age_Delta_hematologic': '#FF8C00',    # orange
    'Age_Delta_immune': '#8B008B',         # purple
}

DELTA_SHORT_NAMES = {
    'Age_Delta_hemodynamic': 'Hemodynamic',
    'Age_Delta_metabolic': 'Metabolic',
    'Age_Delta_renal': 'Renal',
    'Age_Delta_hematologic': 'Hematologic',
    'Age_Delta_immune': 'Immune',
}


def plot_all_components_manhattan(
    df: pd.DataFrame,
    output_dir: str,
    age_delta_cols: List[str],
    age_col: str,
    metric: str = 'neg_log_pvalue',
    control_hgb: bool = False,
    dpi: int = 600,
):
    """
    Nature-style integrated Manhattan plot
    
    Features:
    - Arial/Helvetica font
    - NPG low-saturation palette
    - FDR correction + shaded region
    - Non-significant bars turn gray
    - constrained_layout for automatic layout
    - PDF + PNG dual-format output
    """
    # Apply Nature style
    set_nature_style()
    
    print(f"Generating Nature-style integrated Manhattan plot (with FDR correction)...")
    
    n_components = len(age_delta_cols)
    # Nature standard: double-column width 18cm, 3cm height per row
    fig, axes = plt.subplots(n_components, 1, figsize=(7.2, 1.8 * n_components), constrained_layout=True)
    if n_components == 1:
        axes = [axes]
    
    # Compute the data for each component
    all_data = {}
    all_pvals = {}
    all_corrs = {}
    
    for delta_col in age_delta_cols:
        other_cols = [c for c in age_delta_cols if c != delta_col]
        covars = [age_col] + other_cols
        if control_hgb and 'HGB' in df.columns:
            covars.append('HGB')
        
        data = []
        pvals = []
        corrs = []
        
        for col in ALL_PHYSIO_COLS:
            if col not in df.columns:
                data.append(0)
                pvals.append(1.0)
                corrs.append(0)
                continue
            
            corr, pval, n = multi_partial_correlation(df, delta_col, col, covars)
            value = -np.log10(max(pval, 1e-300))
            data.append(value)
            pvals.append(pval)
            corrs.append(corr)
        
        all_data[delta_col] = data
        all_pvals[delta_col] = pvals
        all_corrs[delta_col] = corrs
    
    x_range = np.arange(len(ALL_PHYSIO_COLS))
    
    # Plot each component
    for idx, delta_col in enumerate(age_delta_cols):
        ax = axes[idx]
        data = all_data[delta_col]
        pvals = np.array(all_pvals[delta_col])
        corrs = all_corrs[delta_col]
        
        # ========== FDR correction ==========
        reject, _, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        
        # Compute the FDR threshold
        fdr_significant_pvals = pvals[reject]
        if len(fdr_significant_pvals) > 0:
            fdr_threshold_p = max(fdr_significant_pvals)
            fdr_threshold_log = -np.log10(fdr_threshold_p)
        else:
            fdr_threshold_log = None
        
        # Signed -log10(P)
        y_values = -np.log10(pvals) * np.sign(corrs)
        
        # Color: significant ones use the NPG palette, non-significant ones use light gray
        bar_colors = []
        for i, (is_sig, indicator) in enumerate(zip(reject, ALL_PHYSIO_COLS)):
            if is_sig:
                group = GROUP_INDICATORS.get(indicator, 'Hemodynamic')
                bar_colors.append(NATURE_PALETTE.get(group, '#808080'))
            else:
                bar_colors.append('#E0E0E0')  # light gray background
        
        # ========== Plotting ==========
        # FDR threshold region (light shaded) - only annotate FDR on the topmost subplot
        if fdr_threshold_log is not None:
            ax.axhspan(-fdr_threshold_log, fdr_threshold_log, 
                      color='#F5F5F5', alpha=0.5, zorder=0, linewidth=0)
            ax.axhline(fdr_threshold_log, color='#555555', linestyle=':', linewidth=0.8, zorder=1)
            ax.axhline(-fdr_threshold_log, color='#555555', linestyle=':', linewidth=0.8, zorder=1)
            # Only annotate FDR on the topmost subplot
            if idx == 0:
                ax.text(len(ALL_PHYSIO_COLS) - 0.5, fdr_threshold_log + 0.3, 
                       'FDR < 0.05', fontsize=6, color='#555555', ha='right', va='bottom', style='italic')
        
        # Plot the bars
        bars = ax.bar(x_range, y_values, color=bar_colors, width=0.8, zorder=10,
                     edgecolor='none', linewidth=0)
        
        # Title (left aligned, bold)
        short_name = DELTA_SHORT_NAMES.get(delta_col, delta_col.replace('Age_Delta_', ''))
        ax.set_title(short_name, loc='left', fontsize=9, fontweight='bold', pad=10)
        
        # Y-axis label
        ax.set_ylabel(r"Signed $-\log_{10}(P)$", fontsize=7)
        
        # X-axis handling
        ax.set_xlim(-0.6, len(ALL_PHYSIO_COLS) - 0.4)
        if idx == n_components - 1:
            ax.set_xticks(x_range)
            ax.set_xticklabels(ALL_PHYSIO_COLS, rotation=45, ha='right', 
                              rotation_mode="anchor", fontsize=6)
        else:
            ax.set_xticks([])
        
        # Despine
        sns.despine(ax=ax, top=True, right=True, bottom=(idx != n_components - 1))
        
        # Horizontal middle line
        ax.axhline(0, color='black', linewidth=0.8, zorder=11)
        
        # Add group separators (dashed lines, present on every subplot)
        current_pos = 0
        for group_name, cols in PHYSIO_GROUPS_ORDERED:
            group_cols_in_data = [c for c in cols if c in df.columns]
            group_len = len(group_cols_in_data)
            current_pos += group_len
            # Add dashed lines between groups (but not after the last group)
            if current_pos < len(ALL_PHYSIO_COLS):
                ax.axvline(x=current_pos - 0.5, color='#BBBBBB', linestyle='--', linewidth=0.6, zorder=5)
    
    # Save (without overall title)
    hgb_suffix = "_with_HGB" if control_hgb else ""
    
    # PDF (vector graphic, suitable for submission)
    pdf_path = os.path.join(output_dir, f"manhattan_all_components{hgb_suffix}.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    # PNG (high-resolution raster)
    png_path = os.path.join(output_dir, f"manhattan_all_components{hgb_suffix}.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print(f"✅ Nature-style Manhattan plot saved:")
    print(f"   PDF: {pdf_path}")
    print(f"   PNG: {png_path}")


def compare_correlations(df, output_dir, age_delta_cols, age_col):
    """Compare the results of simple correlation, age-adjusted partial correlation and mutually-exclusive control."""
    print("\n=== Association comparison analysis (New Classification) ===")
    
    compare_results = []
    
    for delta_col in age_delta_cols:
        for physio_col in ['SBP', 'DBP', 'RBC', 'HGB', 'WBC']:
            if physio_col not in df.columns:
                continue
            
            # 1. Simple correlation
            simple_r, simple_p, n1 = simple_correlation(df, delta_col, physio_col)
            
            # 2. Control for age only
            age_r, age_p, n2 = multi_partial_correlation(df, delta_col, physio_col, [age_col])
            
            # 3. Mutually exclusive control (control for age + the other components)
            other_cols = [c for c in age_delta_cols if c != delta_col]
            mutual_r, mutual_p, n3 = multi_partial_correlation(
                df, delta_col, physio_col, [age_col] + other_cols
            )
            
            compare_results.append({
                'Age_Delta': delta_col,
                'Physio': physio_col,
                'Simple_r': simple_r,
                'Simple_p': simple_p,
                'Age_Adjusted_r': age_r,
                'Age_Adjusted_p': age_p,
                'Mutual_Control_r': mutual_r,
                'Mutual_Control_p': mutual_p,
            })
    
    df_compare = pd.DataFrame(compare_results)
    csv_path = os.path.join(output_dir, "correlation_comparison_new.csv")
    df_compare.to_csv(csv_path, index=False)
    print(f"📊 Comparison table saved to: {csv_path}")
    
    # Print key comparisons
    print("\nKey comparison (SBP, HGB, WBC):")
    for _, row in df_compare[df_compare['Physio'].isin(['SBP', 'HGB', 'WBC'])].iterrows():
        print(f"\n{row['Age_Delta']} vs {row['Physio']}:")
        print(f"  Simple correlation:    r = {row['Simple_r']:.4f}, p = {row['Simple_p']:.2e}")
        print(f"  Age-adjusted:          r = {row['Age_Adjusted_r']:.4f}, p = {row['Age_Adjusted_p']:.2e}")
        print(f"  Mutually exclusive:    r = {row['Mutual_Control_r']:.4f}, p = {row['Mutual_Control_p']:.2e}")
    
    return df_compare


# ============================================
# 4. Main function
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Mutually-exclusive-control Manhattan plot analysis (new grouping)")
    
    parser.add_argument('--data_path', type=str, required=True, help="Path to the prediction-result table (.xlsx)")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory")
    parser.add_argument('--age_col', type=str, default='Age', help="Column name of the true age")
    parser.add_argument('--metric', type=str, default='neg_log_pvalue', 
                       choices=['neg_log_pvalue', 'correlation'],
                       help="The metric to use")
    parser.add_argument('--top-delta', type=str, default='Age_Delta_hemodynamic',
                       help="Age_Delta column shown above the X axis")
    parser.add_argument('--bottom-delta', type=str, default='Age_Delta_immune',
                       help="Age_Delta column shown below the X axis")
    parser.add_argument('--control-hgb', action='store_true',
                       help="Additionally control for HGB (the strongest confounder)")
    parser.add_argument('--compare-only', action='store_true',
                       help="Only run the comparison analysis, do not generate plots")
    parser.add_argument('--age-delta-cols', type=str, 
                       default='Age_Delta_hemodynamic,Age_Delta_metabolic,Age_Delta_renal,Age_Delta_hematologic,Age_Delta_immune',
                       help="All Age_Delta columns, comma separated")
    parser.add_argument('--plot-all-components', action='store_true',
                       help="Generate the integrated Manhattan plot showing all 5 components")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.data_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data: {args.data_path}")
    df = pd.read_excel(args.data_path)
    print(f"Data shape: {df.shape}")
    print(f"Using new physiological grouping: hemodynamic, metabolic, renal, hematologic, immune")
    
    # Check if the age column exists
    if args.age_col not in df.columns:
        print(f"⚠️ Age column '{args.age_col}' does not exist, falling back to simple correlation analysis")
        return
    
    age_delta_cols = [col.strip() for col in args.age_delta_cols.split(',') if col.strip() in df.columns]
    print(f"Age_Delta columns to analyze: {age_delta_cols}")
    print(f"Control HGB: {args.control_hgb}")
    
    # If only running the comparison analysis
    if args.compare_only:
        compare_correlations(df, args.output_dir, age_delta_cols, args.age_col)
        return
    
    # Generate the mutually-exclusive-control correlation table
    generate_mutually_exclusive_table(df, args.output_dir, age_delta_cols, args.age_col, args.control_hgb)
    
    # Get the list of other components
    other_cols = [c for c in age_delta_cols if c not in [args.top_delta, args.bottom_delta]]
    
    # Plot the mutually-exclusive-control Manhattan plot
    plot_mutually_exclusive_manhattan(
        df, args.output_dir,
        top_delta_col=args.top_delta,
        bottom_delta_col=args.bottom_delta,
        age_col=args.age_col,
        other_delta_cols=other_cols,
        metric=args.metric,
        control_hgb=args.control_hgb
    )
    
    # If requested, generate the integrated Manhattan plot for all components
    if args.plot_all_components:
        plot_all_components_manhattan(
            df, args.output_dir,
            age_delta_cols=age_delta_cols,
            age_col=args.age_col,
            metric=args.metric,
            control_hgb=args.control_hgb
        )
    
    # Comparison analysis
    compare_correlations(df, args.output_dir, age_delta_cols, args.age_col)
    
    print("\n✅ Analysis complete!")
    print("\n📌 Interpretation guide (new grouping):")
    print("  - Mutually exclusive control = control for age + the other 4 components")
    print("  - Hemodynamic significant on SBP/DBP → true blood-pressure specificity")
    print("  - Metabolic significant on glucose & lipid indicators → true metabolic specificity")
    print("  - Renal significant on kidney-function indicators → true kidney-function specificity")
    print("  - Hematologic significant on blood-cell indicators → true blood specificity")
    print("  - Immune significant on inflammation indicators → true immune specificity")
    print("  - If a component's prediction is greatly weakened by mutually-exclusive control → likely a shared signal")


if __name__ == '__main__':
    main()
