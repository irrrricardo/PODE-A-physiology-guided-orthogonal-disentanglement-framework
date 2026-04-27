# silhouette_reliability.py - K-Means Clustering Reliability Analysis
# Fig. 4A + 4D: Silhouette analysis, ANOVA feature significance
#
# Usage (from the repository root):
#   python PODE/pode_mapper/umap_clustering/silhouette_reliability.py \
#       --data      outputs/umap/umap_coordinates.csv \
#       --output_dir outputs/kmeans_reliability \
#       --n_clusters 18

import argparse
import os
import colorsys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
)
from scipy import stats


# ---------------------------------------------------------------------------
# Default validation feature columns
# ---------------------------------------------------------------------------
DEFAULT_VALIDATION_FEATURES = [
    'BMI', 'SBP', 'DBP', 'Creatinine', 'WBC', 'PDW',
    'HDL-C', 'FBG', 'Lymphocyte_Count', 'Neutrophil_Count', 'LDL-C',
    'MCH', 'Eosinophil_Count', 'PCT', 'MPV', 'PLT',
    'HCT', 'Monocyte_Count', 'BUN', 'Basophil_Count', 'MCV', 'TG', 'TC',
    'Urine_pH'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fig. 4A/D: K-Means clustering reliability — Silhouette + ANOVA analysis"
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to the UMAP coordinates file (.xlsx or .csv). '
             'Must contain UMAP_1 and UMAP_2 columns (output of generate_umap.py).'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./outputs/kmeans_reliability',
        help='Directory to save reliability plots and ANOVA results.'
    )
    parser.add_argument(
        '--n_clusters', type=int, default=18,
        help='Number of K-Means clusters (default: 18).'
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


def reliability_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
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

    n_clusters = args.n_clusters

    # ── K-Means ───────────────────────────────────────────────────────────────
    print(f"Running K-Means clustering validation (k={n_clusters})...")
    umap_scaler = StandardScaler()
    X_umap = umap_scaler.fit_transform(df[['UMAP_1', 'UMAP_2']])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=args.random_state,
        n_init=args.n_init,
    )
    cluster_labels = kmeans.fit_predict(X_umap)
    df['Cluster'] = cluster_labels

    # ── Reliability metrics ───────────────────────────────────────────────────
    print("Calculating reliability metrics (Silhouette, CH, DB)...")
    sil_score = silhouette_score(X_umap, cluster_labels)
    ch_score  = calinski_harabasz_score(X_umap, cluster_labels)
    db_score  = davies_bouldin_score(X_umap, cluster_labels)

    print(f"   Silhouette Score      : {sil_score:.4f}")
    print(f"   Calinski-Harabasz     : {ch_score:.1f}")
    print(f"   Davies-Bouldin        : {db_score:.4f}")

    # Save scalar metrics to CSV for downstream use
    metrics_df = pd.DataFrame([{
        'n_clusters': n_clusters,
        'silhouette_score': round(sil_score, 4),
        'calinski_harabasz': round(ch_score, 2),
        'davies_bouldin': round(db_score, 4),
    }])
    metrics_path = os.path.join(args.output_dir, 'cluster_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Cluster metrics saved → {metrics_path}")

    # ── Silhouette plot ───────────────────────────────────────────────────────
    print("Generating Silhouette Analysis plot...")

    # Assign colours sorted by UMAP_1 centroid position (left → right)
    centroids     = df.groupby('Cluster')[['UMAP_1', 'UMAP_2']].mean().values
    sorted_clusters = np.argsort(centroids[:, 0])
    hues          = [(i * 0.618033988749895) % 1.0 for i in range(n_clusters)]
    cluster_colors = [colorsys.hsv_to_rgb(h, 0.50, 0.95) for h in hues]
    color_map     = {int(cid): cluster_colors[rank]
                     for rank, cid in enumerate(sorted_clusters)}

    sample_sil_vals = silhouette_samples(X_umap, cluster_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim([-0.1, 0.8])
    ax.set_ylim([0, len(X_umap) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_vals = np.sort(sample_sil_vals[cluster_labels == i])
        y_upper  = y_lower + len(ith_vals)
        color    = color_map.get(i, (0.5, 0.5, 0.5))

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, ith_vals,
            facecolor=color, edgecolor=color, alpha=0.8,
        )
        ax.text(-0.05, y_lower + 0.5 * len(ith_vals), str(i), fontsize=10)
        y_lower = y_upper + 10

    ax.set_title(
        f"Silhouette Analysis for K-Means (k={n_clusters})\n"
        f"Avg Silhouette = {sil_score:.4f}",
        fontsize=14,
    )
    ax.set_xlabel("Silhouette Coefficient", fontsize=12)
    ax.set_ylabel("Cluster Label", fontsize=12)
    ax.axvline(x=sil_score, color='red', linestyle='--', label=f'Avg = {sil_score:.4f}')
    ax.legend(fontsize=11)
    ax.set_yticks([])

    sil_path = os.path.join(args.output_dir, 'silhouette_plot.png')
    plt.savefig(sil_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Silhouette plot saved → {sil_path}")

    # ── ANOVA significance tests ──────────────────────────────────────────────
    validation_features = [f for f in DEFAULT_VALIDATION_FEATURES if f in df.columns]
    missing_feats = set(DEFAULT_VALIDATION_FEATURES) - set(validation_features)
    if missing_feats:
        print(f"⚠️  Validation features not found (skipped): {sorted(missing_feats)}")

    if validation_features:
        print(f"\nRunning one-way ANOVA for {len(validation_features)} features "
              f"across {n_clusters} clusters...")
        anova_results = []
        for feature in validation_features:
            groups   = [df[df['Cluster'] == i][feature].dropna().values
                        for i in range(n_clusters)]
            f_stat, p_val = stats.f_oneway(*groups)
            anova_results.append({
                'Feature': feature,
                'F_Statistic': round(float(f_stat), 4),
                'P_Value': float(p_val),
                'Significant': ('Yes ***' if p_val < 0.001
                                else ('Yes *' if p_val < 0.05 else 'No')),
            })

        anova_df   = pd.DataFrame(anova_results).sort_values('F_Statistic', ascending=False)
        anova_path = os.path.join(args.output_dir, 'anova_significance.csv')
        anova_df.to_csv(anova_path, index=False)

        print("=" * 60)
        print(anova_df.head(10).to_string(index=False))
        print("=" * 60)
        print(f"ANOVA results saved → {anova_path}")
    else:
        print("⚠️  No validation feature columns found; ANOVA skipped.")

    print("✅ Reliability analysis complete!")


def main():
    args = parse_args()
    reliability_analysis(args)


if __name__ == "__main__":
    main()
