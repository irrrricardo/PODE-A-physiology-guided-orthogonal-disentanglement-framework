# evaluate.py

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import timm
from typing import Dict, Any, List
from tqdm import tqdm

# --- Import components from this module ---
from .model import DisentangledVisionFM_V2
from .metrics import MultiTaskMetricsLogger
from .train import PHYSIO_SUBGROUPS, MultiTargetDataset, collate_fn_multi_target

# --- Imports from the parent project ---
from ..shared.data_utils import get_transforms

def main():
    parser = argparse.ArgumentParser(description="Standalone evaluation of a trained disentanglement model")
    
    # --- Core arguments ---
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model weights file (.pth)")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the .xlsx data file used for evaluation")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the evaluation report (evaluation_log.csv)")
    parser.add_argument('--scaler_path', type=str, required=True, help="Path to the scaler_params.pth file saved during training")
    parser.add_argument('--model_config_path', type=str, required=True, help="Path to the model_config.pth file saved during training")
    
    # --- Data-related arguments (must match training) ---
    parser.add_argument('--image_col_left', type=str, required=True, help="Column name of the left-eye image path")
    parser.add_argument('--image_col_right', type=str, required=True, help="Column name of the right-eye image path")
    parser.add_argument('--age_col', type=str, required=True, help="Column name of the age label")
    
    # --- Model and training arguments (must match training) ---
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Data preparation and preprocessing (aligned with train.py) ---
    print(f"Loading evaluation data from {args.data_path}...")
    eval_df = pd.read_excel(args.data_path)

    # Identify valid physiological indicator groups
    actual_physio_groups = {}
    all_physio_cols = []
    for group_name, cols in PHYSIO_SUBGROUPS.items():
        existing_cols = [col for col in cols if col in eval_df.columns]
        if existing_cols:
            actual_physio_groups[group_name] = existing_cols
            all_physio_cols.extend(existing_cols)

    # Load the standardization parameters saved during training
    print(f"Loading standardization parameters from {args.scaler_path}...")
    scaler_params = torch.load(args.scaler_path)
    means = pd.Series(scaler_params['means'])
    stds = pd.Series(scaler_params['stds'])
    
    # Make sure the loaded means and stds only contain columns that exist in the current data
    means = means[means.index.isin(all_physio_cols)]
    stds = stds[stds.index.isin(all_physio_cols)]

    # Median imputation (using training-set statistics)
    # Note: we assume the evaluation set also needs imputation, using the training mean/median is standard practice.
    # For simplicity we use mean imputation here; a more strict approach would be to save the median.
    eval_df[all_physio_cols] = eval_df[all_physio_cols].fillna(means)
    print("Filled missing values in evaluation data with the training-set means.")

    # Standardization (using training-set statistics)
    eval_df[all_physio_cols] = (eval_df[all_physio_cols] - means) / (stds + 1e-6)
    print("Standardized the evaluation data using the training-set parameters.")

    # --- 2. Dynamic model initialization (aligned with train.py) ---
    print("Rebuilding model structure from the configuration file...")
    feature_groups_config = torch.load(args.model_config_path)
    backbone = timm.create_model(args.model_name, pretrained=False, num_classes=0)
    
    model = DisentangledVisionFM_V2(
        backbone=backbone,
        embed_dim=int(backbone.num_features),  # type: ignore
        feature_groups=feature_groups_config,
        head_dropout_rate=0.0  # disable Dropout during inference
    )
    
    # --- 3. Load model weights ---
    print(f"Loading weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 4. Prepare evaluation dataset ---
    eval_dataset = MultiTargetDataset(eval_df, args.image_col_left, args.image_col_right, args.age_col, actual_physio_groups, args.image_size, transform=get_transforms(args.image_size, is_train=False))
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_multi_target)

    # --- 5. Run evaluation ---
    all_predictions = {key: [] for key in feature_groups_config.keys()}
    all_targets = {key: [] for key in feature_groups_config.keys()}

    print("Starting inference on the evaluation set...")
    with torch.no_grad():
        for images_left, images_right, targets in tqdm(eval_loader, desc="Evaluating"):
            if images_left is None:
                continue
            
            images_left = images_left.to(device)
            images_right = images_right.to(device)
            
            predictions = model(images_left, images_right)
            
            for task_name in targets.keys():
                if f"pred_{task_name}" in predictions:
                    all_predictions[task_name].append(predictions[f"pred_{task_name}"].cpu())
                    all_targets[task_name].append(targets[task_name].cpu())

    # --- 6. Compute and save metrics ---
    print("Inference finished. Computing and saving evaluation metrics...")
    metrics_logger = MultiTaskMetricsLogger(args.output_dir)
    metrics_logger.log(epoch=-1, phase='evaluate', all_predictions=all_predictions, all_targets=all_targets)
    
    print(f"Evaluation complete! Detailed metrics have been saved to: {os.path.join(args.output_dir, 'evaluation_log.csv')}")

if __name__ == "__main__":
    # Example run command:
    # python -m PODE.pode_splitter.evaluate \
    #   --model_path outputs/pode_splitter/best_model_v2.pth \
    #   --data_path data/full_age_02.xlsx \
    #   --output_dir outputs/pode_splitter/eval \
    #   --scaler_path outputs/pode_splitter/scaler_params.pth \
    #   --model_config_path outputs/pode_splitter/model_config_v2.pth \
    #   --image_col_left lefteye_path \
    #   --image_col_right righteye_path \
    #   --age_col Age
    main()
