# pode_base/downstream_analysis/scatter_lowess.py
# Fig. 3 A–D: Δage trajectories across age — healthy vs. disease groups with LOWESS fit
#
# Usage (from the repository root):
#   python PODE/pode_base/downstream_analysis/scatter_lowess.py \
#       --data_healthy  data/healthy_analysis.xlsx \
#       --data_hyper    data/full_age_Hyper_pure.xlsx \
#       --output_dir    outputs/scatter_lowess

import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Dataset configurations
# Each entry: (arg_name, default_label, color, marker, sample_size, lowess_frac,
#              density_threshold, bins)
# These are visual defaults; paths are provided via argparse.
# ---------------------------------------------------------------------------
DATASET_DEFAULTS = {
    'data_healthy':    ('Healthy Population',       'cyan',    'o', 40000, 0.35, 50, 60),
    'data_hyper':      ('Hypertension',             'green',   's', 10000, 0.35, 10, 50),
    'data_hyper_cas':  ('Hypertension + CAS',       'peru',    'd',  5000, 0.35, 10, 60),
    'data_hyper_dia':  ('Hypertension + Diabetes',  'hotpink', 'X',  1000, 0.35,  5, 30),
    'data_all_disease':('All Diseases',             'crimson', 'H',  1000, 0.35,  5, 30),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fig. 3: Δage vs Age scatter plot with LOWESS fit across disease groups"
    )
    # --- Input data paths (all optional; missing datasets are silently skipped) ---
    parser.add_argument('--data_healthy',     type=str, default=None,
                        help='Path to healthy reference cohort (.xlsx). Age col: age, Δage col: Δage')
    parser.add_argument('--data_hyper',       type=str, default=None,
                        help='Path to hypertension cohort (.xlsx). Age col: Age, Δage col: Δage')
    parser.add_argument('--data_hyper_cas',   type=str, default=None,
                        help='Path to hypertension + CAS cohort (.xlsx).')
    parser.add_argument('--data_hyper_dia',   type=str, default=None,
                        help='Path to hypertension + diabetes cohort (.xlsx).')
    parser.add_argument('--data_all_disease', type=str, default=None,
                        help='Path to all-disease cohort (.xlsx).')
    # --- Column names ---
    parser.add_argument('--age_col',       type=str, default='Age',
                        help='Age column name (default: Age; use lowercase "age" if needed).')
    parser.add_argument('--delta_col',     type=str, default='delta_age',
                        help='Δage column name (default: delta_age).')
    # --- Output ---
    parser.add_argument('--output_dir',    type=str, default='./outputs/scatter_lowess',
                        help='Directory to save the output figure.')
    parser.add_argument('--output_filename', type=str, default='fig3_lowess_trajectories.png',
                        help='Output filename (default: fig3_lowess_trajectories.png).')
    parser.add_argument('--dpi', type=int, default=400,
                        help='Figure DPI (default: 400).')
    return parser.parse_args()


def load_and_sample(path, age_col, delta_col, sample_size, label):
    """Load a dataset, keep only the two target columns, and take a random sample."""
    if path is None or not os.path.exists(path):
        if path is not None:
            print(f"  ⚠️  File not found for '{label}': {path} — skipped.")
        return None

    try:
        print(f"  Loading '{label}' from {path}...")
        df = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
        # Flexible column lookup: try the provided name and lowercase version
        real_age_col   = age_col   if age_col   in df.columns else age_col.lower()
        real_delta_col = delta_col if delta_col in df.columns else 'Δage'
        df = df[[real_age_col, real_delta_col]].dropna()
        df.columns = ['age', 'delta_age']
        df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"    → {len(df_sampled)} samples ready.")
        return df_sampled
    except Exception as e:
        print(f"  ⚠️  Failed to load '{label}': {e} — skipped.")
        return None


def plot_dataset(ax, data, base_color, label, lowess_frac,
                 density_threshold, bins, marker, z_order):
    """Add scatter points and LOWESS fit line for one dataset to an existing Axes."""
    try:
        rgb = mcolors.to_rgb(base_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        hsv_dark = list(hsv); hsv_dark[2] *= 0.6
        darker = mcolors.hsv_to_rgb(hsv_dark)
    except (ValueError, AttributeError):
        darker = base_color

    sns.scatterplot(
        x='age', y='delta_age', data=data,
        alpha=0.4, s=25, color=base_color,
        label=f'{label} (n={len(data):,})',
        ax=ax, marker=marker, edgecolor='w', linewidth=0.5, zorder=z_order
    )

    try:
        x_data = data['age'].values
        y_data = data['delta_age'].values
        counts, bin_edges = np.histogram(x_data, bins=bins)
        reliable_idx = np.where(counts >= density_threshold)[0]

        if len(reliable_idx) > 0:
            x_lo = bin_edges[reliable_idx[0]]
            x_hi = bin_edges[reliable_idx[-1] + 1]
            smoothed = sm.nonparametric.lowess(y_data, x_data, frac=lowess_frac)
            mask = (smoothed[:, 0] >= x_lo) & (smoothed[:, 0] <= x_hi)
            ax.plot(
                smoothed[mask, 0], smoothed[mask, 1],
                color=darker, linewidth=3,
                label=f'{label} LOWESS (frac={lowess_frac})',
                zorder=z_order + 10
            )
        else:
            print(f"    ⚠️  Insufficient data density for LOWESS fit on '{label}'.")
    except Exception as e:
        print(f"    ⚠️  LOWESS failed for '{label}': {e}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    path_map = {
        'data_healthy':     args.data_healthy,
        'data_hyper':       args.data_hyper,
        'data_hyper_cas':   args.data_hyper_cas,
        'data_hyper_dia':   args.data_hyper_dia,
        'data_all_disease': args.data_all_disease,
    }

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("Loading datasets...")
    datasets = {}
    for key, (label, color, marker, n, frac, threshold, bins_n) in DATASET_DEFAULTS.items():
        path = path_map[key]
        data = load_and_sample(path, args.age_col, args.delta_col, n, label)
        datasets[key] = (data, label, color, marker, frac, threshold, bins_n)

    if all(v[0] is None for v in datasets.values()):
        print("Error: no valid dataset was loaded. Please provide at least one --data_* argument.")
        return

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nGenerating Fig. 3 scatter + LOWESS plot...")
    fig, ax = plt.subplots(figsize=(16, 10))

    for z, (key, (data, label, color, marker, frac, threshold, bins_n)) in \
            enumerate(datasets.items(), start=1):
        if data is not None:
            plot_dataset(ax, data, color, label, frac, threshold, bins_n, marker, z_order=z)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6, zorder=0)
    ax.set_xlabel('Chronological Age (years)', fontsize=14)
    ax.set_ylabel('Δage (Predicted − True, years)', fontsize=14)
    ax.set_title(
        'Relationship between Δage and Age with LOWESS Fit\n'
        '(Healthy vs. Disease Sub-groups)',
        fontsize=15
    )
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=11, loc='upper left', ncol=2)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, args.output_filename)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Fig. 3 saved → {out_path}")


if __name__ == '__main__':
    main()
