# pode_splitter/visualize_feature_space.py

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import umap
from tqdm import tqdm
import timm
import torchvision.transforms as T
from typing import Dict, List, Tuple

# --- Add the project root to sys.path so module imports work ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- Import the model definitions from the project ---
from pode_base.model import AgePredictionViT as OldModel
from pode_splitter.model import DisentangledVisionFM_V2 as NewModel

# --- Default mapping from subspace to representative indicator ---
DEFAULT_COLOR_MAP = {
    'hemo': 'SBP',
    'blood': 'HGB',
    'immune': 'WBC',
    'organ': 'Creatinine'
}

# --- Data loading ---
class VisualizationDataset(Dataset):
    """Simple dataset for visualization."""
    def __init__(self, dataframe, image_col_left, image_col_right, color_cols: List[str]):
        self.df = dataframe
        self.image_col_left = image_col_left
        self.image_col_right = image_col_right
        self.color_cols = color_cols
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        path_left = record[self.image_col_left]
        path_right = record[self.image_col_right]
        
        try:
            img_left = Image.open(path_left).convert('RGB') if pd.notna(path_left) else Image.new('RGB', (224, 224))
            img_right = Image.open(path_right).convert('RGB') if pd.notna(path_right) else Image.new('RGB', (224, 224))
        except Exception:
            img_left, img_right = Image.new('RGB', (224, 224)), Image.new('RGB', (224, 224))

        img_left = self.transform(img_left)
        img_right = self.transform(img_right)
        
        color_values = {col: record[col] for col in self.color_cols}
        
        return (img_left, img_right), color_values

# --- Model loading ---
def load_old_model(model_path, device):
    model = OldModel(model_name='vit_base_patch16_224')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_new_model(model_path, config_path, device):
    config = torch.load(config_path, map_location=device)
    backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    model = NewModel(
        backbone=backbone,
        embed_dim=backbone.num_features, # type: ignore
        feature_groups=config,
        head_dropout_rate=0.0
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# --- Plotting function ---
def plot_comparison(embedding_old, embedding_new, color_values, subspace, color_by_col, output_path, new_feature_dim):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    norm = mcolors.Normalize(vmin=color_values.min(), vmax=color_values.max())
    cmap = "viridis"

    sns.scatterplot(
        x=embedding_old[:, 0], y=embedding_old[:, 1],
        hue=color_values, palette=cmap, s=10, alpha=0.7, ax=ax1, legend=False, hue_norm=norm
    )
    ax1.set_title(f'Old Model: Global Feature Space (768-dim)\nColored by {color_by_col}')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    
    sns.scatterplot(
        x=embedding_new[:, 0], y=embedding_new[:, 1],
        hue=color_values, palette=cmap, s=10, alpha=0.7, ax=ax2, legend=False, hue_norm=norm
    )
    ax2.set_title(f'New Model: Disentangled "{subspace}" Space ({new_feature_dim}-dim)\nColored by {color_by_col}')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes((0.88, 0.15, 0.03, 0.7))
    fig.colorbar(sm, cax=cbar_ax, label=color_by_col)
    
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Comparison plot saved to: {output_path}")

# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="Visualization script comparing the feature spaces of the old and new models.")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the preprocessed data table (.csv).")
    parser.add_argument('--old_model_path', type=str, required=True, help="Path to the old model (AgePredictionViT) weights.")
    parser.add_argument('--new_model_path', type=str, required=True, help="Path to the new model (DisentangledVisionFM) weights (best_model.pth).")
    parser.add_argument('--new_model_config_path', type=str, required=True, help="Path to the new model's config file (model_config.pth).")
    parser.add_argument('--image_col_left', type=str, required=True, help="Column name of the left-eye image.")
    parser.add_argument('--image_col_right', type=str, required=True, help="Column name of the right-eye image.")
    parser.add_argument('--output_dir', type=str, required=True, help="Target folder for the output comparison plots.")
    # Optional arguments for manual mode
    parser.add_argument('--subspace', type=str, default=None, help="[Manual mode] The subspace to visualize (e.g. 'hemo').")
    parser.add_argument('--color_by_col', type=str, default=None, help="[Manual mode] Column name used for coloring (e.g. 'SBP').")
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Decide what tasks to run ---
    tasks_to_run: Dict[str, str] = {}
    if args.subspace and args.color_by_col:
        print(f"--- Manual mode: only generating images for subspace '{args.subspace}' ---")
        tasks_to_run[args.subspace] = args.color_by_col
    else:
        print("--- Auto mode: generating images for all default subspaces ---")
        tasks_to_run = DEFAULT_COLOR_MAP

    # --- Load data ---
    df = pd.read_csv(args.data_file)
    color_cols = list(tasks_to_run.values())
    df = df.dropna(subset=color_cols)
    dataset = VisualizationDataset(df, args.image_col_left, args.image_col_right, color_cols)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # --- Load models ---
    print("Loading models...")
    old_model = load_old_model(args.old_model_path, device)
    new_model = load_new_model(args.new_model_path, args.new_model_config_path, device)
    print("All models loaded.")

    # --- Extract all the features we need in one pass ---
    print("Extracting all features in a single pass...")
    all_old_features, all_new_features, all_color_values = [], {}, {col: [] for col in color_cols}
    for subspace in tasks_to_run.keys():
        all_new_features[subspace] = []

    with torch.no_grad():
        for (img_left, img_right), colors_batch in tqdm(loader, desc="Extracting features"):
            img_left, img_right = img_left.to(device), img_right.to(device)
            
            z_global_old = old_model.vit((img_left + img_right) / 2.0)
            all_old_features.append(z_global_old.cpu())
            
            output_new = new_model(img_left, img_right)
            for subspace in tasks_to_run.keys():
                z_subspace_new = output_new[f'z_{subspace}']
                all_new_features[subspace].append(z_subspace_new.cpu())
            
            for col in color_cols:
                all_color_values[col].append(colors_batch[col].numpy())

    all_old_features = torch.cat(all_old_features, dim=0).numpy()
    for subspace in tasks_to_run.keys():
        all_new_features[subspace] = torch.cat(all_new_features[subspace], dim=0).numpy()
    for col in color_cols:
        all_color_values[col] = np.concatenate(all_color_values[col], axis=0)
    print("Feature extraction complete.")

    # --- Loop through each task ---
    for subspace, color_by_col in tasks_to_run.items():
        print(f"\n--- Processing: subspace='{subspace}', coloring indicator='{color_by_col}' ---")
        
        # Prepare the data for the current task
        old_features = all_old_features
        new_features = all_new_features[subspace]
        color_values = all_color_values[color_by_col]
        
        # UMAP dimensionality reduction
        print("Running UMAP dimensionality reduction...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_old = reducer.fit_transform(old_features)
        embedding_new = reducer.fit_transform(new_features)
        print("UMAP dimensionality reduction complete.")

        # Plot
        output_path = os.path.join(args.output_dir, f"comparison_{subspace}_vs_{color_by_col}.png")
        plot_comparison(embedding_old, embedding_new, color_values, subspace, color_by_col, output_path, new_features.shape[1])

    print("\nAll visualization tasks finished!")

if __name__ == '__main__':
    main()
