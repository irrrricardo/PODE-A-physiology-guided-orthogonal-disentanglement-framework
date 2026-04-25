# train_v2.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import timm
from typing import Dict, Any, List
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json

# --- Import V2 components ---
from .model import DisentangledVisionFM_V2
from .loss import DisentanglementLoss_V2
from .metrics import MultiTaskMetricsLogger

# --- Imports from the parent project ---
from ..shared.data_utils import get_transforms

# ===================================================================
# 1. Early stopping helper class (unchanged)
# ===================================================================
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience, self.verbose, self.delta, self.path, self.trace_func = patience, verbose, delta, path, trace_func
        self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, float('inf')

    def reset(self):
        """Reset the early-stopping counter and state for a new training stage."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        if self.verbose:
            self.trace_func("--- Early stopping counter and state has been reset for the new stage. ---")

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose: self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), self.path)
        self.val_loss_min = val_loss

# ===================================================================
# 2. V2 weight loading scheme (unchanged)
# ===================================================================
def load_pretrained_weights_v2(model: DisentangledVisionFM_V2, checkpoint_path: str, device: torch.device, is_resume: bool) -> DisentangledVisionFM_V2:
    print(f"--- Loading weights from '{checkpoint_path}' ---")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Smartly determine whether this is resume or initialization from a pretrained model
    is_v2_checkpoint = any(k.startswith('projectors.hemo') or k.startswith('age_delta_heads') for k in checkpoint.keys())

    if is_resume and is_v2_checkpoint:
        # --- Scenario 1: resume from a V2 model checkpoint ---
        print("V2 model checkpoint detected, performing resume loading...")
        # Use non-strict mode for better compatibility
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        if missing_keys:
            print(f"⚠️ Missing keys detected during resume loading: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys detected during resume loading: {unexpected_keys}")
        print("✅ V2 model state has been successfully loaded.")
    else:
        # --- Scenario 2: initialize from a standard age prediction model ---
        print("Standard age prediction model detected, performing initialization loading...")
        backbone_weights = {k.replace('vit.', ''): v for k, v in checkpoint.items() if k.startswith('vit.')}
        if not backbone_weights:
            raise ValueError("No backbone weights with 'vit.' prefix found in the pretrained model!")
        model.backbone.load_state_dict(backbone_weights, strict=True)
        print("✅ Backbone (ViT) weights have been successfully loaded.")

        age_head_weights = {k.replace('regression_head.', ''): v for k, v in checkpoint.items() if k.startswith('regression_head.')}
        if age_head_weights:
            model.heads['age'].load_state_dict(age_head_weights, strict=True)
            print("✅ The original Age Head weights have been successfully loaded into the new 'heads.age'.")
        else:
            print("⚠️ No 'regression_head' weights found, 'heads.age' will be trained from scratch.")

        with torch.no_grad():
            age_projector_linear = model.projectors['age']
            assert isinstance(age_projector_linear, nn.Linear)
            age_projector_linear.weight.copy_(torch.eye(age_projector_linear.in_features))
            if age_projector_linear.bias is not None:
                age_projector_linear.bias.zero_()
        print("✅ The linear layer of 'projectors.age' has been initialized to the identity matrix (approximate identity mapping).")
        print("--- V2 model initialization loading complete ---")

    return model

# ===================================================================
# 3. Dataset and training loop (unchanged)
# ===================================================================
# ===================================================================
# New physiological subgroup classification (grouped by physiological system)
# ===================================================================
PHYSIO_SUBGROUPS = {
    # 1. Hemodynamic: pure physical pressure
    'hemodynamic': ['SBP', 'DBP'],

    # 2. Metabolic: glucose & lipid metabolism (Sugar & Fat)
    'metabolic': ['BMI', 'FBG', 'HbA1c', 'TG', 'TC', 'LDL-C', 'HDL-C'],

    # 3. Renal: kidney filtration and excretion (Kidney Function)
    'renal': ['Creatinine', 'BUN', 'UA', 'Urine_pH', 'USG'],

    # 4. Hematologic: blood components (Blood Cells)
    'hematologic': ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW-CV', 'PLT', 'MPV', 'PDW', 'PCT'],

    # 5. Immune: immune inflammation (Inflammation)
    'immune': ['WBC', 'Neutrophil_Count', 'Lymphocyte_Count', 'Monocyte_Count', 'Eosinophil_Count', 'Basophil_Count']
}

class MultiTargetDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_col_left: str, image_col_right: str, age_col: str, physio_groups_cols: Dict[str, List[str]], image_size: int, transform=None, teacher_age_col: str = None):
        self.df, self.image_col_left, self.image_col_right, self.age_col, self.physio_groups_cols, self.image_size, self.transform, self.teacher_age_col = \
            dataframe, image_col_left, image_col_right, age_col, physio_groups_cols, image_size, transform, teacher_age_col
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        path_left, path_right = record[self.image_col_left], record[self.image_col_right]
        try:
            img_left = Image.open(path_left).convert('RGB') if pd.notna(path_left) else Image.new('RGB', (self.image_size, self.image_size))
            img_right = Image.open(path_right).convert('RGB') if pd.notna(path_right) else Image.new('RGB', (self.image_size, self.image_size))
        except Exception as e:
            print(f"Warning: error loading image: {path_left}, {path_right}. Error: {e}. Will be skipped.")
            return None
        if self.transform: img_left, img_right = self.transform(img_left), self.transform(img_right)
        targets = {'age': torch.tensor(record[self.age_col], dtype=torch.float32)}
        
        # [New] Read the Teacher model's prediction
        if self.teacher_age_col and self.teacher_age_col in record and pd.notna(record[self.teacher_age_col]):
            targets['teacher_age'] = torch.tensor(record[self.teacher_age_col], dtype=torch.float32)

        for group_name, cols in self.physio_groups_cols.items():
            targets[group_name] = torch.tensor([record[col] for col in cols], dtype=torch.float32)
        if 'sample_weight' in record: targets['sample_weight'] = torch.tensor(record['sample_weight'], dtype=torch.float32)
        return (img_left, img_right), targets

def collate_fn_multi_target(batch):
    batch = list(filter(None, batch))
    if not batch: return None, None, None
    images_left = torch.stack([item[0][0] for item in batch])
    images_right = torch.stack([item[0][1] for item in batch])
    targets_list = [item[1] for item in batch]
    targets_batch = {}
    if targets_list:
        for key in targets_list[0].keys():
            if all(key in d for d in targets_list): targets_batch[key] = torch.stack([d[key] for d in targets_list])
    return images_left, images_right, targets_batch

def run_epoch_v2(phase, loader, model, criterion, optimizer, device, epoch_num, writer, global_step, scaler, enable_amp, current_stage):
    is_train = phase == 'train'
    model.train(is_train)
    activate_orth_loss = (current_stage >= 2)
    local_rank = dist.get_rank() if dist.is_initialized() else -1
    pbar = tqdm(loader, desc=f"Epoch {epoch_num + 1} [Stage {current_stage}] [{phase.upper()}]", disable=local_rank > 0)
    epoch_losses, all_predictions, all_targets = {}, {}, {}
    for i, (images_left, images_right, targets) in enumerate(pbar):
        if images_left is None: continue
        images_left, images_right = images_left.to(device), images_right.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # The autocast context manager is only used when AMP is enabled
        autocast_kwargs = {'device_type': 'cuda', 'dtype': torch.float16, 'enabled': enable_amp}
        with torch.autocast(**autocast_kwargs):
            with torch.set_grad_enabled(is_train):
                predictions = model(images_left, images_right)
                loss_breakdown = criterion(predictions, targets, activate_orth_loss=activate_orth_loss)
                loss = loss_breakdown['total_loss']

        if is_train:
            optimizer.zero_grad()
            if scaler: # scaler is not None means we are using AMP on CUDA
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # either no AMP or on CPU
                loss.backward()
                optimizer.step()
        if writer and local_rank in [-1, 0]:
            for key, value in loss_breakdown.items():
                scalar_value = value.item() if torch.is_tensor(value) else value
                writer.add_scalar(f'step_{phase}/s{current_stage}_{key}', scalar_value, global_step + i)
                epoch_losses[key] = epoch_losses.get(key, 0.0) + scalar_value
        if not is_train:
            for task_name in targets.keys():
                if f"pred_{task_name}" in predictions:
                    if task_name not in all_predictions: all_predictions[task_name], all_targets[task_name] = [], []
                    all_predictions[task_name].append(predictions[f"pred_{task_name}"].cpu())
                    all_targets[task_name].append(targets[task_name].cpu())
    avg_total_loss = 0
    if local_rank in [-1, 0] and epoch_losses:
        log_str = f"Epoch {epoch_num + 1} [Stage {current_stage}] [{phase.upper()}] "
        for key, value in epoch_losses.items():
            avg_loss = value / len(loader)
            writer.add_scalar(f'epoch_{phase}/s{current_stage}_{key}_avg', avg_loss, epoch_num)
            log_str += f"{key}: {avg_loss:.4f} | "
            if key == 'total_loss': avg_total_loss = avg_loss
        print(log_str)
    return (avg_total_loss, all_predictions, all_targets) if not is_train else (0, None, None)

# ===================================================================
# 4. V2 main training function (refactored)
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="V2 - Three-stage training script for the fully orthogonal disentanglement model")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--image_col_left', type=str, required=True)
    parser.add_argument('--image_col_right', type=str, required=True)
    parser.add_argument('--age_col', type=str, required=True)
    parser.add_argument('--teacher_age_col', type=str, default=None, help="[New] The column name for the teacher model's predictions.")
    parser.add_argument('--pretrained_age_model', type=str, help="Path to the standard age prediction model for initialization.")
    parser.add_argument('--resume_from_checkpoint', type=str, help="Path to a V2 model checkpoint to resume training.")
    parser.add_argument('--output_dir', type=str, default='./disentangled_output_v2')
    parser.add_argument('--lambda_age_orth', type=float, default=1.0, help="Weight for age vs. physio orthogonal loss")
    parser.add_argument('--lambda_physio_orth', type=float, default=0.1, help="Weight for physio vs. physio orthogonal loss")
    parser.add_argument('--task_weights', type=str, default='{}', help='JSON string for task weights. e.g., \'{"age": 1.0, "hemo": 2.0}\'')
    parser.add_argument('--base_age_loss_weight', type=float, default=1.0, help="Weight for the base_age loss component.")
    parser.add_argument('--final_age_loss_weight', type=float, default=1.0, help="Weight for the final_age loss component.")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--stage1_epochs', type=int, default=5)
    parser.add_argument('--stage2_epochs', type=int, default=10)
    parser.add_argument('--stage3_epochs', type=int, default=15)
    parser.add_argument('--stage1_lr', type=float, default=1e-4)
    parser.add_argument('--stage2_lr', type=float, default=1e-5)
    parser.add_argument('--stage3_lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--enable_amp', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    if not args.pretrained_age_model and not args.resume_from_checkpoint:
        raise ValueError("Either --pretrained_age_model (for initialization) or --resume_from_checkpoint (for resuming) must be provided.")

    if 'LOCAL_RANK' in os.environ: args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.local_rank != -1 and not dist.is_initialized():
        torch.cuda.set_device(args.local_rank); dist.init_process_group(backend='nccl', init_method='env://')
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda', args.local_rank) if args.local_rank != -1 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs')) if args.local_rank in [-1, 0] else None

    all_data_df = pd.read_excel(args.data_path)
    all_data_df.dropna(subset=[args.age_col, args.image_col_left, args.image_col_right], how='any', inplace=True)
    actual_physio_groups = {g: [c for c in cs if c in all_data_df.columns] for g, cs in PHYSIO_SUBGROUPS.items()}
    actual_physio_groups = {g: cs for g, cs in actual_physio_groups.items() if cs}
    all_physio_cols = [c for cs in actual_physio_groups.values() for c in cs]
    train_df, val_df = train_test_split(all_data_df, test_size=0.2, random_state=42)
    for col in all_physio_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce'); val_df[col] = pd.to_numeric(val_df[col], errors='coerce')
    # Note: NaN imputation is now handled by masking inside the loss function.
    # medians = train_df[all_physio_cols].median()
    # train_df.fillna(medians, inplace=True); val_df.fillna(medians, inplace=True)
    means, stds = train_df[all_physio_cols].mean(), train_df[all_physio_cols].std()
    train_df[all_physio_cols] = (train_df[all_physio_cols] - means) / (stds + 1e-6)
    val_df[all_physio_cols] = (val_df[all_physio_cols] - means) / (stds + 1e-6)
    torch.save({'means': means.to_dict(), 'stds': stds.to_dict()}, os.path.join(args.output_dir, 'scaler_params.pth'))
    assert isinstance(args.image_size, int)
    train_dataset = MultiTargetDataset(train_df, args.image_col_left, args.image_col_right, args.age_col, actual_physio_groups, args.image_size, transform=get_transforms(args.image_size, is_train=True), teacher_age_col=args.teacher_age_col)
    val_dataset = MultiTargetDataset(val_df, args.image_col_left, args.image_col_right, args.age_col, actual_physio_groups, args.image_size, transform=get_transforms(args.image_size, is_train=False), teacher_age_col=args.teacher_age_col)
    train_sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.local_rank != -1 else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, collate_fn=collate_fn_multi_target, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn_multi_target, sampler=val_sampler)

    backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    embed_dim = int(backbone.num_features)
    feature_groups_config = {'age': {'dim': embed_dim, 'output_dim': 1}}

    # Hard-code the projection dimension for each subspace
    subspace_dims = {
        'hemodynamic': 512,  # blood pressure: SBP, DBP
        'metabolic': 512,    # glucose & lipid metabolism: BMI, FBG, HbA1c, TG, TC, LDL-C, HDL-C
        'renal': 512,        # kidney function: Creatinine, BUN, UA, Urine_pH, USG
        'hematologic': 512,  # blood components: 11 indicators
        'immune': 512        # immune inflammation: 6 indicators
    }
    
    for group_name, cols in actual_physio_groups.items():
        # Get the dimension from the dictionary, fall back to default 128 if not specified
        proj_dim = subspace_dims.get(group_name, 128)
        feature_groups_config[group_name] = {'dim': proj_dim, 'output_dim': len(cols)}
        
    model = DisentangledVisionFM_V2(backbone=backbone, embed_dim=embed_dim, feature_groups=feature_groups_config) # type: ignore
    torch.save(feature_groups_config, os.path.join(args.output_dir, 'model_config_v2.pth'))

    if args.resume_from_checkpoint:
        model = load_pretrained_weights_v2(model, args.resume_from_checkpoint, device, is_resume=True)
    elif args.pretrained_age_model:
        model = load_pretrained_weights_v2(model, args.pretrained_age_model, device, is_resume=False)
    
    model.to(device)

    task_weights = json.loads(args.task_weights)
    criterion = DisentanglementLoss_V2(
        lambda_age_orth=args.lambda_age_orth,
        lambda_physio_orth=args.lambda_physio_orth,
        task_weights=task_weights,
        base_age_loss_weight=args.base_age_loss_weight,
        final_age_loss_weight=args.final_age_loss_weight
    )
    metrics_logger = MultiTaskMetricsLogger(args.output_dir)
    early_stopper = EarlyStopping(patience=args.early_stopping_patience, verbose=True, path=os.path.join(args.output_dir, "best_model_v2.pth"))
    
    # Fix the GradScaler deprecation warning
    # In CPU mode, scaler should be None
    use_cuda = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=args.enable_amp) if use_cuda and args.enable_amp else None
    
    total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs
    optimizer = None # Will be created at the start of the first stage

    for epoch in range(total_epochs):
        current_stage = 0
        if epoch < args.stage1_epochs: current_stage = 1
        elif epoch < args.stage1_epochs + args.stage2_epochs: current_stage = 2
        else: current_stage = 3

        if epoch == 0 or epoch == args.stage1_epochs or epoch == args.stage1_epochs + args.stage2_epochs:
            print(f"\n--- Epoch {epoch+1}/{total_epochs}: Entering Stage {current_stage} ---")
            if current_stage == 1:
                lr = args.stage1_lr
                for name, param in model.named_parameters():
                    param.requires_grad = not ('backbone' in name or 'projectors.age' in name or 'heads.age' in name)
            elif current_stage == 2:
                lr = args.stage2_lr
                for name, param in model.named_parameters():
                    param.requires_grad = 'projectors.age' in name
            else: # Stage 3
                lr = args.stage3_lr
                for param in model.parameters():
                    param.requires_grad = True
            
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if args.local_rank in [-1, 0]: print(f"Stage {current_stage}: Optimizer created for {len(trainable_params)} trainable parameters.")
            optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - epoch)
            
            # [New] Reset the early-stopping counter when a new stage starts
            if early_stopper and epoch > 0:
                early_stopper.reset()

        model_for_training = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True) if args.local_rank != -1 else model
        if train_sampler: train_sampler.set_epoch(epoch)

        run_epoch_v2('train', train_loader, model_for_training, criterion, optimizer, device, epoch, writer, 0, scaler, args.enable_amp, current_stage)
        avg_val_loss, predictions, targets = run_epoch_v2('val', val_loader, model_for_training, criterion, None, device, epoch, writer, 0, scaler, args.enable_amp, current_stage)
        
        if predictions and targets and args.local_rank in [-1, 0]:
            metrics_logger.log(epoch, 'val', predictions, targets)
            if early_stopper:
                early_stopper(avg_val_loss, model)
                if early_stopper.early_stop: print("Early stopping triggered!"); break
        
        if args.local_rank != -1: dist.barrier(device_ids=[args.local_rank])
        scheduler.step()

    if args.local_rank in [-1, 0]: print("--- Training finished ---")
    if args.local_rank != -1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
