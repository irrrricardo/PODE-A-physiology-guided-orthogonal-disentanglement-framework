# evaluate.py

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# --- 【New import】 ---
from sklearn.model_selection import train_test_split

# Import necessary modules from your project
from ..model import AgePredictionViT
from ...shared.data_utils import AgeDataset, get_transforms, collate_fn_skip_corrupted
from .evaluation_utils import (
    calculate_regression_metrics,
    plot_scatter,
    plot_residual,
    plot_age_gap_histogram,
    plot_bland_altman
)


def run_inference(model, loader, device):
    """Run inference on dataset and return all predictions and ground truth labels"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for images, ages in pbar:
            if images.nelement() == 0:
                continue

            images = images.to(device)
            predictions = model(images)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(ages.numpy())

    # Check if any data was collected
    if not all_preds:
        print("Warning: No valid prediction results collected during inference.")
        return np.array([]), np.array([])

    return np.vstack(all_preds), np.concatenate(all_targets)


def main():
    parser = argparse.ArgumentParser(description="PODE-Base Model Evaluation Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights (.pth file)')

    # --- 【Modified: Evaluation also needs the total manifest】 ---
    parser.add_argument('--manifest_path', type=str, required=True,
                        help='Path to CSV manifest containing *all* data (same as used during training)')

    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output folder to save plots')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for input model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size used for evaluation')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')

    # --- 【New parameter: for reproducing data split】 ---
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation set proportion for reproducing data split (must match training)')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test set proportion for reproducing data split (must match training)')
    parser.add_argument('--split_to_evaluate', type=str, default='test', choices=['val', 'test'],
                        help='Choose which data subset to evaluate: "val" or "test"')

    args = parser.parse_args()

    # --- 1. Preparations ---
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Load model ---
    print(f"Loading model from {args.model_path}...")
    model = AgePredictionViT(model_name=args.model_name)
    state_dict = torch.load(args.model_path, map_location=device)

    # Handle "module." prefix caused by DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # Remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    print("Model loaded successfully!")

    # --- 3. Load data (reproduce train.py's split logic) ---
    print(f"--- Loading total data manifest from {args.manifest_path} ---")
    all_data_df = pd.read_csv(args.manifest_path)

    print(f"--- Reproducing Train/Val/Test split (random_state=42) ---")
    print(f"--- Validation set proportion: {args.val_split}, Test set proportion: {args.test_split} ---")

    # --- Key: Reproduce stratified logic exactly as in train.py ---
    try:
        # Ensure 'age' column is numeric type
        all_data_df['age'] = pd.to_numeric(all_data_df['age'], errors='coerce')
        # Remove rows where 'age' is NaN
        all_data_df = all_data_df.dropna(subset=['age'])

        bins = range(0, 101, 10)
        age_bins = pd.cut(all_data_df['age'], bins=bins, right=False, labels=False)
        # Check if all ages are within bin range
        if age_bins.isnull().any():
            print("Warning: Some ages are out of [0, 100) range, these samples will not be used for stratification.")
            # Assign a special bin (e.g., -1) to NaN so they can be stratified
            age_bins = age_bins.fillna(-1)

    except Exception as e:
        print(f"Warning: Error creating age bins: {e}. Will try without stratification (stratify=None).")
        age_bins = None

    # --- First split: separate training set and (validation set + test set) ---
    train_df, temp_df = train_test_split(
        all_data_df,
        test_size=(args.val_split + args.test_split),
        random_state=42,
        stratify=age_bins if age_bins is not None else None
    )

    # --- Second split: separate validation set and test set from temp_df ---
    # Recalculate age bins for temp set
    try:
        # Ensure temp_df is not empty
        if not temp_df.empty:
            temp_age_bins = pd.cut(temp_df['age'], bins=bins, right=False, labels=False)
            if temp_age_bins.isnull().any():
                temp_age_bins = temp_age_bins.fillna(-1)
        else:
            temp_age_bins = None
    except Exception as e:
        print(f"Warning: Error creating temp age bins (possibly due to insufficient data): {e}. Will try without stratification.")
        temp_age_bins = None

    # Only calculate relative size when (val_split + test_split) > 0
    total_split_size = args.val_split + args.test_split
    relative_test_size = args.test_split / total_split_size if total_split_size > 0 else 0

    # Check if stratification is valid
    stratify_valid = temp_age_bins is not None and temp_age_bins.value_counts().min() >= 2 if temp_age_bins is not None else False
    
    if relative_test_size > 0 and relative_test_size < 1.0 and not temp_df.empty and stratify_valid:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=42,
            stratify=temp_age_bins
        )
    elif not temp_df.empty:
        # Handle edge cases without stratification
        if relative_test_size == 1.0:  # Only test set, no validation set
            val_df = pd.DataFrame(columns=temp_df.columns)
            test_df = temp_df
        elif relative_test_size == 0:  # Only validation set, no test set
            val_df = temp_df
            test_df = pd.DataFrame(columns=temp_df.columns)
        else:
            val_df, test_df = train_test_split(temp_df, test_size=relative_test_size, random_state=42)
    else:
        # temp_df is empty
        val_df = pd.DataFrame(columns=temp_df.columns)
        test_df = pd.DataFrame(columns=temp_df.columns)

    print(f"Dataset split reproduced: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # --- 【Key: Choose which DataFrame to evaluate based on parameter】 ---
    if args.split_to_evaluate == 'val':
        eval_df = val_df
        print(f"--- Target: Evaluate 'Validation' dataset ({len(eval_df)} samples) ---")
    else:
        eval_df = test_df
        print(f"--- Target: Evaluate 'Test' dataset ({len(eval_df)} samples) ---")

    if eval_df.empty:
        print(f"Error: '{args.split_to_evaluate}' dataset is empty, cannot continue evaluation.")
        return

    # Use is_train=False to get transforms without data augmentation
    dataset = AgeDataset(eval_df, transform=get_transforms(args.image_size, is_train=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn_skip_corrupted,
        shuffle=False  # No shuffling needed during evaluation
    )

    # --- 4. Run inference ---
    y_pred, y_true = run_inference(model, loader, device)

    if y_pred.size == 0 or y_true.size == 0:
        print("Evaluation failed: No valid data obtained from data loader.")
        return

    y_pred = y_pred.flatten()  # Ensure it's a 1D array

    # --- 5. Calculate metrics and draw plots ---
    print("\n--- Calculating metrics and generating plots ---")

    # Dynamically create output directory based on evaluated subset
    split_output_dir = os.path.join(args.output_dir, f"{args.split_to_evaluate}_split_results")
    os.makedirs(split_output_dir, exist_ok=True)
    print(f"Results will be saved to: {split_output_dir}")

    # ==================== 【New code: Save prediction results to CSV】 ====================
    print("--- Saving detailed prediction data ---")
    results_df = pd.DataFrame({
        'True_Age': y_true,  # True age column
        'Predicted_Age': y_pred,  # Predicted age column
        'Error': y_pred - y_true  # Additional error column (predicted - true), convenient for you to directly sort and view
    })

    # Construct save path, ensure it's in the same folder as images
    csv_save_path = os.path.join(split_output_dir, 'inference_details.csv')

    # Save as CSV
    results_df.to_csv(csv_save_path, index=False)
    print(f"✅ Detailed prediction CSV saved to: {csv_save_path}")
    # ==================== 【New code end】 ====================

    calculate_regression_metrics(y_true, y_pred)

    plot_scatter(y_true, y_pred, os.path.join(split_output_dir, 'scatter_plot.png'))
    plot_residual(y_true, y_pred, os.path.join(split_output_dir, 'residual_plot.png'))
    plot_age_gap_histogram(y_true, y_pred, os.path.join(split_output_dir, 'age_gap_histogram.png'))
    plot_bland_altman(y_true, y_pred, os.path.join(split_output_dir, 'bland_altman_plot.png'))

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
