# train_dp.py

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# --- Import from our custom modules ---
from .model import AgePredictionViT, load_mae_weights
from ..shared.metrics import MetricsLogger
# --- Key modification: import collate_fn ---
from ..shared.data_utils import AgeDataset, get_transforms, collate_fn_skip_corrupted


def run_epoch(phase, loader, model, criterion, optimizer, device, epoch_num):
    """Run one training or validation/test epoch (DP version)"""
    is_train = phase == 'train'
    if is_train:
        model.train()
    else:
        model.eval()

    all_preds, all_targets = [], []
    desc = f"Epoch {epoch_num + 1} [{phase.upper()}]" if epoch_num >= 0 else f"Final Test [{phase.upper()}]"

    pbar = tqdm(loader, desc=desc)

    for images, ages in pbar:
        # If the batch is empty (all samples corrupted), skip it
        if images.nelement() == 0:
            continue

        images = images.to(device)
        ages = ages.to(device).float().unsqueeze(1)

        with torch.set_grad_enabled(is_train):
            predictions = model(images)
            loss = criterion(predictions, ages)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        all_preds.append(predictions.detach().cpu())
        all_targets.append(ages.detach().cpu())

    if not all_preds:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(all_preds), torch.cat(all_targets)


def main():
    """Main training function (DP version)"""
    parser = argparse.ArgumentParser(description="MAE ViT Fine-tuning Training Script (DataParallel version)")

    # --- Parameter definitions (unchanged) ---
    parser.add_argument('--manifest_path', type=str, required=True)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--mae_weights_path', type=str, required=True)
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to model checkpoint for resuming training (.pth). If provided, --mae_weights_path will be ignored')
    parser.add_argument('--output_dir', type=str, default='./dp_finetune_output')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--head_dropout_rate', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128, help='This is the total batch size across all GPUs')
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--early_stop_delta', type=float, default=0.01)

    args = parser.parse_args()

    # --- Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Detected {torch.cuda.device_count()} GPUs, will use DataParallel.")
    else:
        print("No GPU detected, will run on CPU.")

    # --- Data loading and splitting ---
    print("--- Preparing dataset ---")
    all_data_df = pd.read_csv(args.manifest_path)
    age_bins = pd.cut(all_data_df['age'], bins=range(0, 101, 10), right=False, labels=False)
    train_df, temp_df = train_test_split(all_data_df, test_size=(args.val_split + args.test_split), random_state=42,
                                         stratify=age_bins)
    temp_age_bins = pd.cut(temp_df['age'], bins=range(0, 101, 10), right=False, labels=False)
    # Check if all age bins have at least 2 samples for stratification
    stratify_valid = temp_age_bins.value_counts().min() >= 2
    relative_test_size = args.test_split / (args.val_split + args.test_split) if (args.val_split + args.test_split) > 0 else 0
    if stratify_valid:
        val_df, test_df = train_test_split(temp_df, test_size=relative_test_size, random_state=42, stratify=temp_age_bins)
    else:
        val_df, test_df = train_test_split(temp_df, test_size=relative_test_size, random_state=42)

    print(f"Dataset split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_dataset = AgeDataset(train_df, transform=get_transforms(args.image_size, is_train=True))
    val_dataset = AgeDataset(val_df, transform=get_transforms(args.image_size, is_train=False))
    test_dataset = AgeDataset(test_df, transform=get_transforms(args.image_size, is_train=False))

    # ==================== 【Key modification: start】 ====================
    # Purpose: Create a weighted sampler for the training set to handle data imbalance

    # 1. Calculate the number of samples in each age bin in the training set
    # We need to reset the index of train_df so that we can use .loc to access it later
    train_df = train_df.reset_index(drop=True)
    train_age_bins = pd.cut(train_df['age'], bins=range(0, 101, 10), right=False, labels=False)
    class_counts = train_age_bins.value_counts().sort_index()

    # 2. Calculate the weight for each class (weight = 1 / class_count)
    class_weights = 1.0 / class_counts

    # 3. Assign each sample in the training set the weight of its corresponding class
    # .map() will look up the corresponding weight in class_weights for each value in train_age_bins (0, 1, 2...)
    sample_weights = train_age_bins.map(class_weights).to_numpy()

    # 4. Create a weighted random sampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    print("Weighted sampler created for training set to balance age distribution.")
    # ==================== 【Key modification: end】 ====================

    # --- DataLoader ---
    # --- Key modification: specify collate_fn in DataLoader ---
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_skip_corrupted,
        persistent_workers=True,
        # --- 【Core fix】---
        sampler=sampler,  # <-- Enable the weighted sampler you created
        shuffle=False  # <-- Must set shuffle to False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn_skip_corrupted
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn_skip_corrupted
    )

    # --- Model initialization ---
    model_without_dp = AgePredictionViT(
        model_name=args.model_name,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        head_dropout_rate=args.head_dropout_rate
    )

    model_without_dp = load_mae_weights(model_without_dp, args.mae_weights_path)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model_without_dp)
    else:
        model = model_without_dp
    model.to(device)

    # --- Optimizer and logger ---
    param_groups = [
        {'params': model_without_dp.vit.parameters(), 'lr': args.backbone_lr},
        {'params': model_without_dp.regression_head.parameters(), 'lr': args.head_lr}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    metrics_logger = MetricsLogger(args.output_dir)
    best_val_mae, patience_counter = float('inf'), 0

    # --- Training loop ---
    for epoch in range(args.epochs):
        train_preds, train_targets = run_epoch('train', train_loader, model, criterion, optimizer, device, epoch)
        scheduler.step()

        val_preds, val_targets = run_epoch('val', val_loader, model, criterion, None, device, epoch)

        # Check if there are valid validation results
        if val_preds.nelement() == 0:
            print(f"Epoch {epoch + 1} - Validation set has no valid data, skipping evaluation.")
            continue

        print("-" * 60)
        metrics_logger.log(epoch, 'train', train_preds, train_targets)
        current_val_mae = metrics_logger.log(epoch, 'val', val_preds, val_targets)
        print("-" * 60)

        if current_val_mae < best_val_mae - args.early_stop_delta:
            best_val_mae = current_val_mae
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "best_model_dp.pth")
            torch.save(model_without_dp.state_dict(), save_path)
            print(f"New best model! Validation MAE: {best_val_mae:.4f}, saved to '{save_path}'")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print(f"Validation MAE has not improved for {args.early_stop_patience} consecutive epochs, triggering early stopping.")
            break

    # --- Final test ---
    print("\n" + "=" * 20 + " Final Test " + "=" * 20)
    best_model_path = os.path.join(args.output_dir, "best_model_dp.pth")
    if os.path.exists(best_model_path):
        model_without_dp.load_state_dict(torch.load(best_model_path, map_location=device))
        test_preds, test_targets = run_epoch('test', test_loader, model, criterion, None, device, -1)
        if test_preds.nelement() > 0:
            metrics_logger.log(-1, 'test', test_preds, test_targets)
        else:
            print("Test set has no valid data, cannot perform final evaluation.")
    else:
        print("Warning: Best model file not found, skipping final test.")
    print("=" * 50)


if __name__ == "__main__":
    main()