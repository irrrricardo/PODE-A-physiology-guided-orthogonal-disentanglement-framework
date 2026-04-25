# pode_base/downstream_analysis/octile_violin_box.py
# Supp. Fig. 1: Δage octile violin/box plots across clinical indicators
#
# Usage (from the repository root):
#   python PODE/pode_base/downstream_analysis/octile_violin_box.py \
#       --data data/full_age_02.xlsx \
#       --output_dir outputs/octile_violin \
#       --target_col delta_age \
#       --num_bins 8

import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Default feature list (30 clinical indicators)
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_COLUMNS = [
    'BMI', 'SBP', 'DBP', 'MCHC', 'Creatinine', 'WBC', 'PDW',
    'HDL-C', 'FBG', 'Lymphocyte_Count', 'Neutrophil_Count', 'LDL-C',
    'MCH', 'RDW-CV', 'RBC', 'Eosinophil_Count', 'PCT', 'MPV', 'PLT',
    'HCT', 'Monocyte_Count', 'BUN', 'Basophil_Count', 'MCV', 'TG', 'TC',
    'HGB', 'UA', 'Urine_pH', 'USG'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supp. Fig. 1: Δage octile violin/box plots across clinical indicators"
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to the master clinical table (.xlsx or .csv).'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./outputs/octile_violin',
        help='Directory to save per-feature violin+box plots.'
    )
    parser.add_argument(
        '--target_col', type=str, default='delta_age',
        help='Column name for Δage (default: delta_age).'
    )
    parser.add_argument(
        '--num_bins', type=int, default=8,
        help='Number of quantile bins (default: 8 for octile).'
    )
    parser.add_argument(
        '--auto_zoom', action='store_true', default=True,
        help='Zoom the Y-axis to the IQR ± 1.5×IQR range (default: True).'
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='Figure DPI (default: 150).'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"Loading data from '{args.data}'...")
    try:
        if args.data.endswith('.csv'):
            df = pd.read_csv(args.data)
        else:
            df = pd.read_excel(args.data)
    except FileNotFoundError:
        print(f"Error: file not found → {args.data}")
        return

    if args.target_col not in df.columns:
        print(f"Error: target column '{args.target_col}' not found in data.")
        print(f"  Available columns: {list(df.columns)}")
        return

    # ── 2. Quantile binning ───────────────────────────────────────────────────
    print(f"Binning '{args.target_col}' into {args.num_bins} quantile groups...")
    try:
        quantiles = [i / args.num_bins for i in range(args.num_bins + 1)]
        bins = df[args.target_col].quantile(quantiles).tolist()
        labels = [
            f'Q{i + 1}\n({bins[i]:.2f} to {bins[i + 1]:.2f})'
            for i in range(args.num_bins)
        ]
        group_col = f'{args.target_col}_Group'
        df[group_col] = pd.cut(
            df[args.target_col], bins=bins, labels=labels,
            include_lowest=True, duplicates='drop'
        )
        n_groups = df[group_col].nunique()
        print(f"Divided into {n_groups} groups.")
    except Exception as e:
        print(f"Error: failed to bin '{args.target_col}'. {e}")
        return

    # ── 3. Filter feature columns present in the data ─────────────────────────
    feature_cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    missing = set(DEFAULT_FEATURE_COLUMNS) - set(feature_cols)
    if missing:
        print(f"⚠️  Columns not found in data (skipped): {sorted(missing)}")
    print(f"Generating plots for {len(feature_cols)} features...")

    stats_lines = [
        f"Δage Octile Stratification Report — '{args.target_col}'\n{'=' * 60}\n\n"
    ]

    # ── 4. Generate per-feature violin + box plots ────────────────────────────
    for feature in feature_cols:
        # Statistics
        stats = df.groupby(group_col, observed=True)[feature].describe()
        stats_lines.append(f"--- {args.target_col} vs {feature} ---\n")
        stats_lines.append(stats.to_string())
        stats_lines.append(f"\n{'-' * 50}\n\n")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            f"Distribution of {feature} across {args.target_col} Groups",
            fontsize=16
        )

        sns.boxplot(ax=axes[0], x=group_col, y=feature, data=df)
        axes[0].set_title('Box Plot', fontsize=14)
        axes[0].set_xlabel(f'{args.target_col} (Grouped by {args.num_bins}-ile)')
        axes[0].set_ylabel(feature)

        sns.violinplot(ax=axes[1], x=group_col, y=feature, data=df)
        axes[1].set_title('Violin Plot', fontsize=14)
        axes[1].set_xlabel(f'(Grouped by {args.num_bins}-ile)')
        axes[1].set_ylabel('')

        if args.auto_zoom:
            try:
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                padding = IQR * 0.1
                y_lo = Q1 - 1.5 * IQR - padding
                y_hi = Q3 + 1.5 * IQR + padding
                axes[0].set_ylim(y_lo, y_hi)
                axes[1].set_ylim(y_lo, y_hi)
            except Exception as e:
                print(f"  Warning: auto-zoom failed for '{feature}'. {e}")

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        out_path = os.path.join(
            args.output_dir,
            f'octile_{args.target_col}_vs_{feature}.png'
        )
        plt.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    # ── 5. Save statistics report ─────────────────────────────────────────────
    report_path = os.path.join(args.output_dir, 'delta_age_stratification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("".join(stats_lines))
    print(f"\nStatistics report saved → {report_path}")
    print("✅ Octile violin/box analysis complete!")


if __name__ == '__main__':
    main()
