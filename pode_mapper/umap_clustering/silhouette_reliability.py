# k-means_cluster_reliability.py - K-Means Clustering Reliability Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import colorsys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from scipy import stats

# ==================== [Configuration] ====================
FILE_PATH = "/vepfs-bj-climate/climate/ricardo/PycharmProjectStorage/data4train/fig_draw/umap_analysis/subgroup1/umap_results_with_coords.xlsx"
OUTPUT_DIR = "/vepfs-bj-climate/climate/ricardo/PycharmProjectStorage/data4train/fig_draw/umap_analysis/kmeans_smart_color_soft/reliability_validation_18/"
N_CLUSTERS = 18
VALIDATION_FEATURES = [
    'BMI', 'SBP', 'DBP', 'Creatinine', 'WBC', 'PDW',
    'HDL-C', 'FBG', 'Lymphocyte_Count', 'Neutrophil_Count', 'LDL-C',
    'MCH', 'Eosinophil_Count', 'PCT', 'MPV', 'PLT',
    'HCT', 'Monocyte_Count', 'BUN', 'Basophil_Count', 'MCV', 'TG', 'TC',
    'Urine_pH'
]


# ========================================================

def reliability_analysis():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading file: {FILE_PATH}")
    try:
        df = pd.read_excel(FILE_PATH)
    except:
        df = pd.read_csv(FILE_PATH)

    print(f"Running K-Means clustering validation (k={N_CLUSTERS})...")

    umap_scaler = StandardScaler()
    X_umap = umap_scaler.fit_transform(df[['UMAP_1', 'UMAP_2']])

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_umap)
    df['Cluster'] = cluster_labels

    print("Calculating reliability metrics (Silhouette, CH, DB)...")
    sil_score = silhouette_score(X_umap, cluster_labels)
    ch_score = calinski_harabasz_score(X_umap, cluster_labels)
    db_score = davies_bouldin_score(X_umap, cluster_labels)

    print(f"   -> Silhouette Score: {sil_score:.4f}")
    print(f"   -> Calinski-Harabasz: {ch_score:.1f}")
    print(f"   -> Davies-Bouldin: {db_score:.4f}")

    print("Generating Silhouette Analysis plot...")

    centroids = df.groupby('Cluster')[['UMAP_1', 'UMAP_2']].mean().values
    sorted_clusters = np.argsort(centroids[:, 0])

    hues = [(i * 0.618033988749895) % 1.0 for i in range(N_CLUSTERS)]
    optimized_colors = [colorsys.hsv_to_rgb(h, 0.50, 0.95) for h in hues]

    color_map = {}
    for rank, cluster_id in enumerate(sorted_clusters):
        color_map[cluster_id] = optimized_colors[rank]

    plt.figure(figsize=(10, 8))
    plt.xlim([-0.1, 0.8])
    plt.ylim([0, len(X_umap) + (N_CLUSTERS + 1) * 10])

    sample_silhouette_values = silhouette_samples(X_umap, cluster_labels)
    y_lower = 10

    for i in range(N_CLUSTERS):
        ith_cluster_sil_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_sil_values.sort()

        size_cluster_i = ith_cluster_sil_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = color_map[i]

        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_sil_values,
                          facecolor=color,
                          edgecolor=color,
                          alpha=0.8)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=10)
        y_lower = y_upper + 10

    plt.title(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {N_CLUSTERS}", fontsize=14)
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.axvline(x=sil_score, color="red", linestyle="--", label="Average Score")
    plt.legend()
    plt.yticks([])

    save_path = os.path.join(OUTPUT_DIR, 'silhouette_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Silhouette analysis plot saved: {save_path}")

    print("\nRunning biological feature ANOVA significance tests...")
    anova_results = []

    for feature in VALIDATION_FEATURES:
        if feature not in df.columns: continue

        groups = [df[df['Cluster'] == i][feature].dropna() for i in range(N_CLUSTERS)]
        f_stat, p_val = stats.f_oneway(*groups)

        anova_results.append({
            'Feature': feature,
            'F_Statistic': f_stat,
            'P_Value': p_val,
            'Significant': 'Yes ***' if p_val < 0.001 else ('Yes *' if p_val < 0.05 else 'No')
        })

    anova_df = pd.DataFrame(anova_results).sort_values('F_Statistic', ascending=False)
    anova_path = os.path.join(OUTPUT_DIR, 'anova_significance.csv')
    anova_df.to_csv(anova_path, index=False)

    print("=" * 60)
    print(anova_df.head(10).to_string(index=False))
    print("=" * 60)
    print(f"Significance test results saved: {anova_path}")


if __name__ == "__main__":
    reliability_analysis()