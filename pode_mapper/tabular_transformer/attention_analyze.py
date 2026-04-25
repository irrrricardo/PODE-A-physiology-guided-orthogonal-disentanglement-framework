# attention_analyze.py - Generate Supp Fig 2: Tabular Transformer Attention Heatmap
#
# Paper description (Supp Fig 2):
#   "Attention heatmap from a Tabular Transformer analyzing associations between
#    multi-dimensional indicators and predicted FundusAge.
#    Y-axis: 'query' indicator; X-axis: 'attended-to' indicators.
#    FundusAge exhibited low association with most other indicators,
#    suggesting it is a novel and independent biomarker."
#
# Usage (from the repository root):
#   python -m PODE.pode_mapper.tabular_transformer.attention_analyze \
#       --data data/full_age_02.xlsx \
#       --checkpoint outputs/tab_transformer/best_model.pth \
#       --scaler outputs/tab_transformer/scaler.pkl \
#       --output_dir outputs/tab_transformer/attn

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from .model import load_checkpoint
from .dataset import TabularDataset
from .data_utils import load_data, load_scaler, DEFAULT_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Display names for clinical indicators (cleaner labels for the heatmap)
# Order must match DEFAULT_FEATURE_NAMES in data_utils.py
# ---------------------------------------------------------------------------
DISPLAY_NAMES = [
    'BMI', 'SBP', 'DBP', 'AS Level', 'MCHC', 'Creatinine',
    'WBC', 'PDW', 'HDL-C', 'FBG', 'Lymphocyte',
    'Neutrophil', 'LDL-C', 'MCH', 'RDW-CV', 'RBC',
    'Eosinophil', 'PCT', 'MPV', 'PLT', 'HCT',
    'Monocyte', 'BUN', 'Basophil', 'MCV',
    'TG', 'TC', 'HGB', 'UA', 'Urine pH', 'USG',
    'FundusAge'   # ← should show LOW attention with others
]


def compute_attention_matrix(model, dataloader, device, max_batches=None):
    """
    Run the full dataloader through the model and compute the
    dataset-averaged attention weight matrix.

    Returns:
        attn_matrix: numpy array of shape (n_features, n_features)
                     attn_matrix[i, j] = how much feature i (query) attends
                     to feature j (key), averaged over all layers and samples.
    """
    print("Computing attention matrix over dataset...")
    attn_tensor = model.get_attention_matrix(
        dataloader, device, n_batches=max_batches
    )
    attn_np = attn_tensor.numpy()
    print(f"Attention matrix shape: {attn_np.shape}")
    return attn_np


def plot_attention_heatmap(
    attn_matrix: np.ndarray,
    feature_labels: list,
    output_path: str,
    title: str = "Tabular Transformer Attention Heatmap\n(Clinical Indicators × FundusAge)",
    figsize=(14, 12),
    cmap: str = "YlOrRd",
    annotate: bool = False,
    dpi: int = 300,
):
    """
    Plot the attention weight matrix as a heatmap (Supp Fig 2).

    Args:
        attn_matrix: (n_features, n_features) array
        feature_labels: list of label strings
        output_path: path to save PNG
        title: plot title
        figsize: figure size
        cmap: colormap
        annotate: whether to show numeric values in cells
        dpi: output resolution
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize to [0, 1] for clearer visualization
    attn_norm = attn_matrix / (attn_matrix.max() + 1e-8)

    sns.heatmap(
        attn_norm,
        ax=ax,
        cmap=cmap,
        xticklabels=feature_labels,
        yticklabels=feature_labels,
        vmin=0, vmax=1,
        linewidths=0.3,
        linecolor='lightgray',
        annot=annotate,
        fmt='.2f' if annotate else '',
        cbar_kws={'label': 'Normalized Attention Weight', 'shrink': 0.8},
    )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Attended-to Indicator (Key)", fontsize=12)
    ax.set_ylabel("Query Indicator", fontsize=12)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    # Highlight FundusAge row and column
    if 'FundusAge' in feature_labels:
        fa_idx = feature_labels.index('FundusAge')
        n = len(feature_labels)
        # Draw a rectangle around FundusAge column (key)
        ax.add_patch(mpatches.Rectangle(
            (fa_idx, 0), 1, n,
            fill=False, edgecolor='blue', linewidth=2.5, zorder=5
        ))
        # Draw a rectangle around FundusAge row (query)
        ax.add_patch(mpatches.Rectangle(
            (0, fa_idx), n, 1,
            fill=False, edgecolor='blue', linewidth=2.5, zorder=5
        ))

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Attention heatmap saved to: {output_path}")


def save_attention_csv(attn_matrix, feature_labels, output_path):
    """Save the raw attention matrix as a CSV for further analysis."""
    df = pd.DataFrame(attn_matrix, index=feature_labels, columns=feature_labels)
    df.to_csv(output_path)
    print(f"Attention matrix CSV saved to: {output_path}")


def print_fundusage_summary(attn_matrix, feature_labels):
    """Print summary of FundusAge attention values vs other features."""
    if 'FundusAge' not in feature_labels:
        return
    fa_idx = feature_labels.index('FundusAge')

    # Row: FundusAge as query (attends to others)
    fa_as_query = attn_matrix[fa_idx, :]
    # Col: FundusAge as key (others attend to FundusAge)
    fa_as_key = attn_matrix[:, fa_idx]

    others = [f for f in feature_labels if f != 'FundusAge']
    other_idx = [i for i, f in enumerate(feature_labels) if f != 'FundusAge']

    print("\n" + "=" * 55)
    print("FundusAge Attention Summary")
    print("=" * 55)
    print("FundusAge as KEY (how much each feature attends TO FundusAge):")
    sorted_key = sorted(zip(others, fa_as_key[other_idx]),
                        key=lambda x: x[1], reverse=True)
    for name, val in sorted_key[:5]:
        print(f"  {name:25s} → {val:.4f}")
    mean_key = np.mean(fa_as_key[other_idx])
    print(f"  Mean attention TO FundusAge: {mean_key:.4f}")

    print("\nFundusAge as QUERY (how much FundusAge attends to others):")
    sorted_query = sorted(zip(others, fa_as_query[other_idx]),
                          key=lambda x: x[1], reverse=True)
    for name, val in sorted_query[:5]:
        print(f"  {name:25s} → {val:.4f}")
    mean_query = np.mean(fa_as_query[other_idx])
    print(f"  Mean attention FROM FundusAge: {mean_query:.4f}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Supp Fig 2: Tabular Transformer Attention Heatmap"
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to full_age_02.xlsx (clinical + FundusAge + Age)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--scaler', type=str, required=True,
                        help='Path to saved scaler (.pkl)')
    parser.add_argument('--output_dir', type=str, default='./attention_output',
                        help='Directory to save heatmap and CSV')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Must match training split (default 0.2)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--use_full_data', action='store_true',
                        help='Use all data for attention analysis (not just test set)')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit number of batches for faster analysis')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--annotate', action='store_true',
                        help='Show numeric values in heatmap cells (slower)')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y = load_data(args.data, feature_names=DEFAULT_FEATURE_NAMES)

    if args.use_full_data:
        X_analysis = X
        y_analysis = y
        print(f"Using full dataset: {len(X_analysis)} samples")
    else:
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )
        X_analysis = X_test
        y_analysis = y_test
        print(f"Using test set: {len(X_analysis)} samples")

    scaler, medians = load_scaler(args.scaler)
    X_filled = X_analysis.fillna(medians)
    X_sc = scaler.transform(X_filled).astype('float32')

    y_np = y_analysis.values.astype('float32') if hasattr(y_analysis, 'values') else y_analysis.astype('float32')
    dataset = TabularDataset(X_sc, y_np)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # ── 2. Load model ─────────────────────────────────────────────────────────
    model = load_checkpoint(args.checkpoint, device)
    model.eval()

    # ── 3. Compute attention matrix ───────────────────────────────────────────
    attn_matrix = compute_attention_matrix(
        model, dataloader, device, max_batches=args.max_batches
    )

    # ── 4. Print FundusAge summary ────────────────────────────────────────────
    print_fundusage_summary(attn_matrix, DISPLAY_NAMES)

    # ── 5. Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'attention_matrix.csv')
    save_attention_csv(attn_matrix, DISPLAY_NAMES, csv_path)

    # ── 6. Plot heatmap (Supp Fig 2) ──────────────────────────────────────────
    heatmap_path = os.path.join(args.output_dir, 'normative_architecture.png')
    plot_attention_heatmap(
        attn_matrix,
        feature_labels=DISPLAY_NAMES,
        output_path=heatmap_path,
        annotate=args.annotate,
        dpi=args.dpi,
    )

    print(f"\n✅ Supp Fig 2 saved to: {heatmap_path}")


if __name__ == '__main__':
    main()
