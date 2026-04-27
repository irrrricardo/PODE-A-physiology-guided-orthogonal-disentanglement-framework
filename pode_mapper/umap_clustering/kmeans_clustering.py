# kmeans_clustering.py - K-Means Forced Clustering with Detailed Statistics
# Fig. 4C + 4E: K-means clustering on UMAP coordinates
#
# Usage (from the repository root):
#   python PODE/pode_mapper/umap_clustering/kmeans_clustering.py \
#       --data      outputs/umap/umap_coordinates.csv \
#       --output_dir outputs/kmeans \
#       --n_clusters 18 \
#       --target_col delta_age

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Default feature columns (matching PODE-Splitter physiological grouping)
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_COLUMNS = [
    'BMI', 'SBP', 'DBP', 'Creatinine', 'WBC', 'PDW',
    'HDL-C', 'FBG', 'Lymphocyte_Count', 'Neutrophil_Count', 'LDL-C',
    'MCH', 'Eosinophil_Count', 'PCT', 'MPV', 'PLT',
    'HCT', 'Monocyte_Count', 'BUN', 'Basophil_Count', 'MCV', 'TG', 'TC',
    'Urine_pH'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fig. 4C/E: K-Means forced clustering on UMAP coordinates"
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to the UMAP coordinates file (.xlsx or .csv). '
             'Must contain UMAP_1 and UMAP_2 columns (output of generate_umap.py).'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./outputs/kmeans',
        help='Directory to save clustering results and plots.'
    )
    parser.add_argument(
        '--n_clusters', type=int, default=18,
        help='Number of K-Means clusters (default: 18).'
    )
    parser.add_argument(
        '--target_col', type=str, default='delta_age',
        help='Column name for Δage (default: delta_age).'
    )
    parser.add_argument(
        '--age_col', type=str, default='Age',
        help='Column name for chronological age (default: Age).'
    )
    parser.add_argument(
        '--fundus_age_col', type=str, default='Predicted_Age',
        help='Column name for fundus age prediction (default: Predicted_Age).'
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help='Random seed for K-Means (default: 42).'
    )
    parser.add_argument(
        '--n_init', type=int, default=10,
        help='Number of K-Means initializations (default: 10).'
    )
    parser.add_argument(
        '--dpi', type=int, default=300,
        help='Figure DPI (default: 300).'
    )
    return parser.parse_args()


def force_cluster_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading file: {args.data}")
    try:
        df = pd.read_excel(args.data) if args.data.endswith('.xlsx') else pd.read_csv(args.data)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if 'UMAP_1' not in df.columns or 'UMAP_2' not in df.columns:
        print("Error: UMAP_1 and/or UMAP_2 columns not found! "
              "Please run generate_umap.py first.")
        return

    real_age_col    = args.age_col        if args.age_col        in df.columns else None
    real_fundus_col = args.fundus_age_col if args.fundus_age_col in df.columns else None
    real_target_col = args.target_col     if args.target_col     in df.columns else None

    if real_target_col is None:
        print(f"Warning: target column '{args.target_col}' not found. "
              f"Available: {list(df.columns)}")

    # ── K-Means on UMAP coordinates ───────────────────────────────────────────
    print(f"Fitting K-Means with k={args.n_clusters}...")
    umap_scaler = StandardScaler()
    X_umap = umap_scaler.fit_transform(df[['UMAP_1', 'UMAP_2']])

    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        n_init=args.n_init,
    )
    df['Cluster'] = kmeans.fit_predict(X_umap)

    # ── Cluster map plot ──────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x='UMAP_1', y='UMAP_2',
        hue='Cluster', palette='tab20',
        style='Cluster', s=30, alpha=0.8,
    )
    plt.title(f'K-Means Clustering (k={args.n_clusters})', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    map_path = os.path.join(args.output_dir, f'kmeans_map_k{args.n_clusters}.png')
    plt.savefig(map_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Cluster map saved → {map_path}")

    # ── Per-cluster statistics ────────────────────────────────────────────────
    print(f"\nGenerating per-cluster statistics (k={args.n_clusters})...")

    # Only keep feature columns that actually exist in the dataframe
    feature_cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    missing_feats = set(DEFAULT_FEATURE_COLUMNS) - set(feature_cols)
    if missing_feats:
        print(f"⚠️  Feature columns not found (skipped): {sorted(missing_feats)}")

    summary_report = []
    for cid in range(args.n_clusters):
        island_df   = df[df['Cluster'] == cid]
        mainland_df = df[df['Cluster'] != cid]

        stats_dict = {
            'Cluster_ID': cid,
            'Size': len(island_df),
        }

        if real_target_col:
            stats_dict['Delta_Age_Mean'] = round(island_df[real_target_col].mean(), 2)
            stats_dict['Delta_Age_Std']  = round(island_df[real_target_col].std(),  2)
        if real_age_col:
            stats_dict['Age_Mean'] = round(island_df[real_age_col].mean(), 2)
            stats_dict['Age_Std']  = round(island_df[real_age_col].std(),  2)
        if real_fundus_col:
            stats_dict['FundusAge_Mean'] = round(island_df[real_fundus_col].mean(), 2)
            stats_dict['FundusAge_Std']  = round(island_df[real_fundus_col].std(),  2)

        # Top-5 differentiating features vs. rest of cohort
        diff_list = []
        for col in feature_cols:
            mean_island = island_df[col].mean()
            mean_main   = mainland_df[col].mean()
            denominator = abs(mean_main) if mean_main != 0 else 1e-6
            diff_pct = (mean_island - mean_main) / denominator * 100
            diff_list.append({
                'Feature': col,
                'Diff_Percent': diff_pct,
                'Abs_Diff': abs(diff_pct),
                'Island_Mean': mean_island,
            })

        diff_df  = pd.DataFrame(diff_list).sort_values('Abs_Diff', ascending=False).head(5)
        feat_str = ", ".join(
            f"{r['Feature']}(Val:{r['Island_Mean']:.2f}, Diff:{r['Diff_Percent']:.0f}%)"
            for _, r in diff_df.iterrows()
        )
        stats_dict['Key_Features_Values'] = feat_str
        summary_report.append(stats_dict)

    report_df = pd.DataFrame(summary_report)

    # Re-order columns sensibly
    first_cols = ['Cluster_ID', 'Size']
    if real_target_col:
        first_cols += ['Delta_Age_Mean', 'Delta_Age_Std']
    if real_age_col:
        first_cols += ['Age_Mean', 'Age_Std']
    if real_fundus_col:
        first_cols += ['FundusAge_Mean', 'FundusAge_Std']
    first_cols.append('Key_Features_Values')
    report_df = report_df[[c for c in first_cols if c in report_df.columns]]

    report_path = os.path.join(args.output_dir, 'kmeans_detailed_summary.csv')
    report_df.to_csv(report_path, index=False)

    print("\n" + "=" * 80)
    print(f"K-Means Clustering Summary (k={args.n_clusters})")
    print("=" * 80)
    print(report_df.to_string(index=False))
    print(f"\nResults saved → {report_path}")
    print("✅ K-Means clustering analysis complete!")


def main():
    args = parse_args()
    force_cluster_analysis(args)


if __name__ == "__main__":
    main()
