# forced_cluster_new.py - K-Means Forced Clustering with Detailed Statistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==================== [Configuration] ====================
FILE_PATH = "/vepfs-bj-climate/climate/ricardo/PycharmProjectStorage/data4train/fig_draw/umap_analysis/subgroup1/umap_results_with_coords.xlsx"
OUTPUT_DIR = "/vepfs-bj-climate/climate/ricardo/PycharmProjectStorage/data4train/fig_draw/umap_analysis/kmeans_results_new/"
N_CLUSTERS = 18
TARGET_COLUMN = 'delta age'
AGE_COLUMN = 'Age'
FUNDUS_AGE_COLUMN = 'FundusAge'
FEATURE_COLUMNS = [
    'BMI', 'SBP', 'DBP', 'Creatinine', 'WBC', 'PDW',
    'HDL-C', 'FBG', 'Lymphocyte_Count', 'Neutrophil_Count', 'LDL-C',
    'MCH', 'Eosinophil_Count', 'PCT', 'MPV', 'PLT',
    'HCT', 'Monocyte_Count', 'BUN', 'Basophil_Count', 'MCV', 'TG', 'TC',
    'Urine_pH'
]


# ========================================================

def force_cluster_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading file: {FILE_PATH}")
    try:
        df = pd.read_excel(FILE_PATH)
    except:
        df = pd.read_csv(FILE_PATH)

    if 'UMAP_1' not in df.columns:
        print("Error: UMAP coordinates not found!")
        return

    real_age_col = AGE_COLUMN if AGE_COLUMN in df.columns else None
    real_fundus_col = FUNDUS_AGE_COLUMN if FUNDUS_AGE_COLUMN in df.columns else None

    print(f"Forcing {N_CLUSTERS} clusters...")
    umap_scaler = StandardScaler()
    X_umap = umap_scaler.fit_transform(df[['UMAP_1', 'UMAP_2']])

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_umap)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='UMAP_1', y='UMAP_2', hue='Cluster', palette='tab10', style='Cluster', s=30, alpha=0.8)
    plt.title(f'Forced Clustering (K-Means, k={N_CLUSTERS})', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(OUTPUT_DIR, f'kmeans_map_k{N_CLUSTERS}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\nGenerating detailed statistics report...")
    summary_report = []

    for cid in range(N_CLUSTERS):
        island_df = df[df['Cluster'] == cid]
        mainland_df = df[df['Cluster'] != cid]

        stats_dict = {
            'Cluster_ID': cid,
            'Size': len(island_df),
            'Delta_Age_Mean': round(island_df[TARGET_COLUMN].mean(), 2),
            'Delta_Age_Std': round(island_df[TARGET_COLUMN].std(), 2)
        }

        if real_age_col:
            stats_dict['Age_Mean'] = round(island_df[real_age_col].mean(), 2)
            stats_dict['Age_Std'] = round(island_df[real_age_col].std(), 2)

        if real_fundus_col:
            stats_dict['FundusAge_Mean'] = round(island_df[real_fundus_col].mean(), 2)
            stats_dict['FundusAge_Std'] = round(island_df[real_fundus_col].std(), 2)

        diff_list = []
        for col in FEATURE_COLUMNS:
            if col not in df.columns: continue

            mean_island = island_df[col].mean()
            mean_main = mainland_df[col].mean()
            if mean_main == 0: mean_main = 1e-6

            diff_pct = (mean_island - mean_main) / abs(mean_main) * 100
            diff_list.append({
                'Feature': col,
                'Diff_Percent': diff_pct,
                'Abs_Diff': abs(diff_pct),
                'Island_Mean': mean_island
            })

        diff_df = pd.DataFrame(diff_list).sort_values('Abs_Diff', ascending=False).head(5)

        feat_str = ", ".join([f"{r['Feature']}(Val:{r['Island_Mean']:.2f}, Diff:{r['Diff_Percent']:.0f}%)" for _, r in diff_df.iterrows()])
        stats_dict['Key_Features_Values'] = feat_str

        summary_report.append(stats_dict)

    report_df = pd.DataFrame(summary_report)

    first_cols = ['Cluster_ID', 'Size', 'Delta_Age_Mean', 'Delta_Age_Std']
    if real_age_col: first_cols.extend(['Age_Mean', 'Age_Std'])
    if real_fundus_col: first_cols.extend(['FundusAge_Mean', 'FundusAge_Std'])
    first_cols.append('Key_Features_Values')

    final_cols = [c for c in first_cols if c in report_df.columns]
    report_df = report_df[final_cols]

    report_path = os.path.join(OUTPUT_DIR, 'kmeans_detailed_summary.csv')
    report_df.to_csv(report_path, index=False)

    print("\n" + "=" * 80)
    print(f"Forced clustering report (k={N_CLUSTERS})")
    print("=" * 80)
    print(report_df.to_string(index=False))
    print(f"\nResults saved to: {report_path}")


if __name__ == "__main__":
    force_cluster_analysis()