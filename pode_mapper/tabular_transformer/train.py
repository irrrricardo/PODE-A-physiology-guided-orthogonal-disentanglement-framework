# train.py - Training script for Tabular Transformer
# Trains on clinical indicators + FundusAge → predict True Age
# Paper: 37,175 train / 9,295 test (80/20 split of mixed cohort)

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

from .model import TabularTransformer, save_checkpoint
from .dataset import TabularDataset
from .data_utils import load_data, preprocess_features, DEFAULT_FEATURE_NAMES


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, preds_all, targets_all = 0.0, [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds_all.extend(pred.detach().cpu().numpy().flatten())
        targets_all.extend(y_batch.cpu().numpy().flatten())

    mae = mean_absolute_error(targets_all, preds_all)
    return total_loss / len(loader.dataset), mae


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, targets_all = 0.0, [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds_all.extend(pred.cpu().numpy().flatten())
            targets_all.extend(y_batch.cpu().numpy().flatten())

    mae = mean_absolute_error(targets_all, preds_all)
    return total_loss / len(loader.dataset), mae


def main():
    parser = argparse.ArgumentParser(
        description="Train Tabular Transformer on clinical indicators + FundusAge"
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to full_age_02.xlsx (clinical + FundusAge + Age)')
    parser.add_argument('--output_dir', type=str, default='./tab_transformer_output',
                        help='Directory to save model and logs')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test fraction (default 0.2 → 37175/9295 split)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=64,
                        help='Transformer embedding dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y = load_data(args.data, feature_names=DEFAULT_FEATURE_NAMES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"Split: Train={len(X_train)}, Test={len(X_test)}")

    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    X_train_sc, _, X_test_sc, _ = preprocess_features(
        X_train, X_test=X_test, scaler_path=scaler_path
    )

    train_ds = TabularDataset(X_train_sc, y_train)
    test_ds  = TabularDataset(X_test_sc, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

    # ── 2. Build model ────────────────────────────────────────────────────────
    model = TabularTransformer(
        feature_names=DEFAULT_FEATURE_NAMES,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── 3. Training loop ──────────────────────────────────────────────────────
    log_rows = []
    best_test_mae = float('inf')
    best_ckpt_path = os.path.join(args.output_dir, 'best_model.pth')

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train MAE':>9} | {'Test Loss':>9} | {'Test MAE':>8}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_mae = eval_epoch(model, test_loader, criterion, device)
        scheduler.step()

        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_mae:>9.4f} | "
              f"{te_loss:>9.4f} | {te_mae:>8.4f}")

        log_rows.append({
            'epoch': epoch, 'train_loss': tr_loss, 'train_mae': tr_mae,
            'test_loss': te_loss, 'test_mae': te_mae
        })

        if te_mae < best_test_mae:
            best_test_mae = te_mae
            save_checkpoint(model, best_ckpt_path, extra_info={
                'best_test_mae': best_test_mae, 'epoch': epoch
            })
            print(f"  ↳ New best Test MAE: {best_test_mae:.4f} — saved to {best_ckpt_path}")

    # ── 4. Save training log ──────────────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    log_df.to_csv(log_path, index=False)
    print(f"\nTraining complete. Best Test MAE: {best_test_mae:.4f}")
    print(f"Training log: {log_path}")
    print(f"Best model:   {best_ckpt_path}")


if __name__ == '__main__':
    main()
