# pode_mapper/umap_clustering/generate_umap.py
# Fig. 4B: UMAP projection of the clinical + FundusAge feature space
#
# Produces:
#   - umap_continuous_delta_age.png   — points colored by continuous Δage
#   - umap_quartile_delta_age.png     — points colored by Δage quartile
#   - umap_coordinates.csv            — (umap_1, umap_2, delta_age, ...) for downstream use
#   - umap_hyperparam_search.csv      — trustworthiness scores across hyperparameter grid
#   - umap_model.pkl                  — serialized UMAP reducer for reproducibility
#
# Usage (from the repository root):
#   python PODE/pode_mapper/umap_clustering/generate_umap.py \
#       --data data/full_age_02.xlsx \
#       --output_dir outputs/umap \
#       --delta_col delta_age \
#       --n_neighbors 30 \
#       --min_dist 0.1

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
import umap
import umap.plot

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Default feature list: all 31 clinical indicators + FundusAge (Predicted_Age)
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_COLS = [
    'BMI', 'SBP', 'DBP', 'MCHC', 'Creatinine', 'WBC', 'PDW',
    'HDL-C', 'FBG', 'Lymphocyte_Count', 'Neutrophil_Count', 'LDL-C',
    'MCH', 'RDW-CV', 'RBC', 'Eosinophil_Count', 'PCT', 'MPV', 'PLT',
    'HCT', 'Monocyte_Count', 'BUN', 'Basophil_Count', 'MCV', 'TG', 'TC',
    'HGB', 'UA', 'Urine_pH', 'USG',
    'Predicted_Age',   # FundusAge from PODE-Base (may also appear as 'FundusAge')
]

# Hyperparameter search grid
SEARCH_N_NEIGHBORS = [10, 15, 20, 30, 50]
SEARCH_MIN_DIST    = [0.05, 0.10, 0.20]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fig. 4B: UMAP projection of the clinical + FundusAge feature space"
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to full_age_02.xlsx (must contain delta_age and clinical cols).')
    parser.add_argument('--output_dir', type=str, default='./outputs/umap',
                        help='Directory to save figures, coordinates and model.')
    parser.add_argument('--delta_col', type=str, default='delta_age',
                        help='Column name for Δage used to colour points (default: delta_age).')
    parser.add_argument('--n_neighbors', type=int, default=30,
                        help='UMAP n_neighbors (default: 30). Ignored if --hyperparam_search.')
    parser.add_argument('--min_dist', type=float, default=0.10,
                        help='UMAP min_dist (default: 0.10). Ignored if --hyperparam_search.')
    parser.add_argument('--n_components', type=int, default=2,
                        help='UMAP embedding dimensionality (default: 2).')
    parser.add_argument('--metric', type=str, default='euclidean',
                        help='Distance metric for UMAP (default: euclidean).')
    parser.add_argument('--max_samples', type=int, default=20000,
                        help='Cap sample count for performance (default: 20000; 0 = no cap).')
    parser.add_argument('--hyperparam_search', action='store_true',
                        help='Run a grid search over n_neighbors × min_dist using trustworthiness.')
    parser.add_argument('--tw_n_neighbors', type=int, default=15,
                        help='k for trustworthiness evaluation (default: 15).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Figure DPI (default: 300).')
    parser.add_argument('--point_size', type=float, default=3.0,
                        help='Scatter point size (default: 3.0).')
    parser.add_argument('--point_alpha', type=float, default=0.5,
                        help='Scatter point alpha (default: 0.5).')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: load + preprocess
# ---------------------------------------------------------------------------
def load_and_preprocess(path: str, feature_cols: list, delta_col: str,
                         max_samples: int, seed: int):
    """
    Load data, keep valid feature+target rows, standardize features.

    Returns
    -------
    X_scaled : np.ndarray  (N, D)
    df_clean : pd.DataFrame  (N rows, columns = feature_cols + delta_col)
    scaler   : fitted StandardScaler
    used_cols: list[str]  — subset of feature_cols that were actually present
    """
    print(f"Loading data from '{path}'...")
    df = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")

    # Handle 'FundusAge' alias
    if 'Predicted_Age' not in df.columns and 'FundusAge' in df.columns:
        df = df.rename(columns={'FundusAge': 'Predicted_Age'})

    used_cols = [c for c in feature_cols if c in df.columns]
    missing   = set(feature_cols) - set(used_cols)
    if missing:
        print(f"  ⚠️  Feature columns not found (skipped): {sorted(missing)}")

    required = used_cols + ([delta_col] if delta_col in df.columns else [])
    df_clean = df[required].dropna().reset_index(drop=True)
    print(f"  After dropna: {len(df_clean)} rows, {len(used_cols)} features.")

    if max_samples > 0 and len(df_clean) > max_samples:
        df_clean = df_clean.sample(n=max_samples, random_state=seed)
        df_clean = df_clean.reset_index(drop=True)
        print(f"  Subsampled to {max_samples} rows for efficiency.")

    X = df_clean[used_cols].values.astype(np.float32)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df_clean, scaler, used_cols


# ---------------------------------------------------------------------------
# Helper: trustworthiness-based hyperparameter search
# ---------------------------------------------------------------------------
def hyperparameter_search(X_scaled: np.ndarray, seed: int,
                           tw_k: int, output_dir: str) -> tuple:
    """
    Grid-search over SEARCH_N_NEIGHBORS × SEARCH_MIN_DIST.
    Evaluates each combination using sklearn trustworthiness.
    Returns (best_n_neighbors, best_min_dist, results_df).
    """
    print("\n--- Hyperparameter search (trustworthiness) ---")
    records = []
    best_tw, best_nn, best_md = -1, SEARCH_N_NEIGHBORS[0], SEARCH_MIN_DIST[0]

    for nn in SEARCH_N_NEIGHBORS:
        for md in SEARCH_MIN_DIST:
            reducer = umap.UMAP(
                n_neighbors=nn, min_dist=md, n_components=2,
                metric='euclidean', random_state=seed, n_jobs=1,
                low_memory=True
            )
            emb = reducer.fit_transform(X_scaled)
            # Evaluate on a subsample for speed (max 5000 points)
            n_eval = min(5000, len(X_scaled))
            idx = np.random.default_rng(seed).choice(len(X_scaled), n_eval, replace=False)
            tw = trustworthiness(X_scaled[idx], emb[idx], n_neighbors=tw_k)
            print(f"  n_neighbors={nn:3d}, min_dist={md:.2f}  →  trustworthiness={tw:.4f}")
            records.append({'n_neighbors': nn, 'min_dist': md, 'trustworthiness': tw})
            if tw > best_tw:
                best_tw, best_nn, best_md = tw, nn, md

    results_df = pd.DataFrame(records).sort_values('trustworthiness', ascending=False)
    csv_path = os.path.join(output_dir, 'umap_hyperparam_search.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nBest: n_neighbors={best_nn}, min_dist={best_md}  (trustworthiness={best_tw:.4f})")
    print(f"Search results saved → {csv_path}")

    # Plot heatmap
    pivot = results_df.pivot(index='n_neighbors', columns='min_dist', values='trustworthiness')
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index], fontsize=9)
    ax.set_xlabel('min_dist', fontsize=11)
    ax.set_ylabel('n_neighbors', fontsize=11)
    ax.set_title('UMAP Trustworthiness\nHyperparameter Grid', fontsize=12)
    plt.colorbar(im, ax=ax, label='Trustworthiness')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f'{pivot.values[i, j]:.3f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if pivot.values[i, j] < pivot.values.mean() else 'black')
    plt.tight_layout()
    hm_path = os.path.join(output_dir, 'umap_hyperparam_heatmap.png')
    plt.savefig(hm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Hyperparameter heatmap saved → {hm_path}")

    return best_nn, best_md, results_df


# ---------------------------------------------------------------------------
# Helper: plot UMAP coloured by continuous delta_age
# ---------------------------------------------------------------------------
def plot_continuous(emb: np.ndarray, delta_values: np.ndarray,
                    output_dir: str, dpi: int, size: float, alpha: float,
                    n_neighbors: int, min_dist: float):
    """Fig. 4B-style: continuous Δage colour gradient."""
    vmin = np.percentile(delta_values, 2)
    vmax = np.percentile(delta_values, 98)

    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(
        emb[:, 0], emb[:, 1],
        c=delta_values, cmap='RdBu_r',
        vmin=vmin, vmax=vmax,
        s=size, alpha=alpha, linewidths=0
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Δage (years)', fontsize=12)
    ax.set_xlabel('UMAP 1', fontsize=13)
    ax.set_ylabel('UMAP 2', fontsize=13)
    ax.set_title(
        f'UMAP — Continuous Δage\n'
        f'(n_neighbors={n_neighbors}, min_dist={min_dist})',
        fontsize=13
    )
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(output_dir, 'umap_continuous_delta_age.png')
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Continuous Δage UMAP saved → {out}")


# ---------------------------------------------------------------------------
# Helper: plot UMAP coloured by Δage quartile
# ---------------------------------------------------------------------------
def plot_quartile(emb: np.ndarray, delta_values: np.ndarray,
                  output_dir: str, dpi: int, size: float, alpha: float,
                  n_neighbors: int, min_dist: float):
    """Discrete quartile coloring (Q1–Q4)."""
    quartiles   = pd.qcut(delta_values, q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    palette     = {'Q1 (Low)': '#2166AC', 'Q2': '#74ADD1', 'Q3': '#F46D43', 'Q4 (High)': '#D73027'}
    colors      = [palette[str(q)] for q in quartiles]

    fig, ax = plt.subplots(figsize=(9, 8))
    for label, color in palette.items():
        mask = np.array([str(q) == label for q in quartiles])
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=color, s=size, alpha=alpha, linewidths=0,
            label=f'{label} (n={mask.sum():,})'
        )
    ax.set_xlabel('UMAP 1', fontsize=13)
    ax.set_ylabel('UMAP 2', fontsize=13)
    ax.set_title(
        f'UMAP — Δage Quartile\n'
        f'(n_neighbors={n_neighbors}, min_dist={min_dist})',
        fontsize=13
    )
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(output_dir, 'umap_quartile_delta_age.png')
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Quartile Δage UMAP saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load + preprocess ──────────────────────────────────────────────────
    X_scaled, df_clean, scaler, used_cols = load_and_preprocess(
        path=args.data,
        feature_cols=DEFAULT_FEATURE_COLS,
        delta_col=args.delta_col,
        max_samples=args.max_samples,
        seed=args.seed
    )

    # ── 2. Hyperparameter search (optional) ───────────────────────────────────
    if args.hyperparam_search:
        best_nn, best_md, _ = hyperparameter_search(
            X_scaled, seed=args.seed,
            tw_k=args.tw_n_neighbors,
            output_dir=args.output_dir
        )
        n_neighbors = best_nn
        min_dist    = best_md
        print(f"\n→ Using best hyperparameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
    else:
        n_neighbors = args.n_neighbors
        min_dist    = args.min_dist

    # ── 3. Fit UMAP ───────────────────────────────────────────────────────────
    print(f"\nFitting UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, "
          f"metric={args.metric}, n_components={args.n_components})...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=args.n_components,
        metric=args.metric,
        random_state=args.seed,
        n_jobs=1,
        low_memory=True,
        verbose=False
    )
    embedding = reducer.fit_transform(X_scaled)
    print(f"Embedding shape: {embedding.shape}")

    # Trustworthiness on the fitted embedding
    n_eval = min(5000, len(X_scaled))
    idx_eval = np.random.default_rng(args.seed).choice(len(X_scaled), n_eval, replace=False)
    tw = trustworthiness(X_scaled[idx_eval], embedding[idx_eval],
                         n_neighbors=args.tw_n_neighbors)
    print(f"Trustworthiness (k={args.tw_n_neighbors}): {tw:.4f}")

    # ── 4. Save model ─────────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, 'umap_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'reducer': reducer, 'scaler': scaler,
                     'feature_cols': used_cols,
                     'n_neighbors': n_neighbors, 'min_dist': min_dist,
                     'trustworthiness': tw}, f)
    print(f"UMAP model saved → {model_path}")

    # ── 5. Save coordinates ───────────────────────────────────────────────────
    coord_df = pd.DataFrame({
        'umap_1': embedding[:, 0],
        'umap_2': embedding[:, 1],
    })
    if args.n_components >= 3:
        coord_df['umap_3'] = embedding[:, 2]

    # Append delta_age and other key metadata if available
    for col in [args.delta_col, 'Age', 'Predicted_Age']:
        if col in df_clean.columns:
            coord_df[col] = df_clean[col].values

    csv_path = os.path.join(args.output_dir, 'umap_coordinates.csv')
    coord_df.to_csv(csv_path, index=False)
    print(f"Coordinates saved → {csv_path}")

    # ── 6. Visualize ──────────────────────────────────────────────────────────
    if args.n_components >= 2 and args.delta_col in df_clean.columns:
        delta_vals = df_clean[args.delta_col].values.astype(float)

        # Fig. 4B: continuous colour
        plot_continuous(
            embedding, delta_vals,
            output_dir=args.output_dir,
            dpi=args.dpi,
            size=args.point_size,
            alpha=args.point_alpha,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )

        # Quartile colour version
        plot_quartile(
            embedding, delta_vals,
            output_dir=args.output_dir,
            dpi=args.dpi,
            size=args.point_size,
            alpha=args.point_alpha,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
    else:
        print(f"⚠️  '{args.delta_col}' not found; visualization skipped.")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  UMAP Summary")
    print(f"{'─'*55}")
    print(f"  Samples        : {len(X_scaled):,}")
    print(f"  Features used  : {len(used_cols)}")
    print(f"  n_neighbors    : {n_neighbors}")
    print(f"  min_dist       : {min_dist}")
    print(f"  Trustworthiness: {tw:.4f}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"{'─'*55}")
    print("✅ UMAP analysis complete!")


if __name__ == '__main__':
    main()
