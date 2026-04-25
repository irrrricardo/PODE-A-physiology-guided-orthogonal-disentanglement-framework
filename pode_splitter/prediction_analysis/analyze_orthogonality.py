# pode_splitter/prediction_analysis/analyze_orthogonality.py

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Use relative imports to make sure we load from the V2 module ---
from ..model import DisentangledVisionFM_V2
from ...shared.data_utils import get_transforms
from .predict import InferenceDataset # reuse the Dataset from predict.py

def analyze_and_plot_orthogonality(args: argparse.Namespace):
    """
    Load model and data, extract all Z feature vectors, compute their pairwise correlations,
    and plot the heatmap.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting orthogonality analysis, using device: {device} ---")

    # --- 1. Load data ---
    print(f"Loading data from {args.data_path}...")
    df = pd.read_excel(args.data_path)
    # To save computation time, we may use only a subset of the data for analysis
    if len(df) > args.num_samples_for_analysis:
        df = df.sample(n=args.num_samples_for_analysis, random_state=42)
        print(f"Randomly sampled {args.num_samples_for_analysis} samples for analysis.")
    
    dataset = InferenceDataset(df, args.image_col_left, args.image_col_right, transform=get_transforms(args.image_size, is_train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # --- 2. Load V2 model ---
    print("Loading the V2 model...")
    model_config = torch.load(args.model_config_path, map_location=device)
    backbone = timm.create_model(args.model_name, pretrained=False, num_classes=0)
    # Make sure DisentangledVisionFM_V2 is used
    model = DisentangledVisionFM_V2(backbone=backbone, embed_dim=int(backbone.num_features), feature_groups=model_config, head_dropout_rate=0.0) # type: ignore
    
    print(f"Loading V2 model weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 3. Extract all Z feature vectors ---
    print("Starting to extract Z feature vectors...")
    all_z_features = {}
    with torch.no_grad():
        for images_left, images_right in tqdm(loader, desc="Extracting Features"):
            images_left, images_right = images_left.to(device), images_right.to(device)
            predictions = model(images_left, images_right)
            
            for key, value in predictions.items():
                if key.startswith('z_'):
                    if key not in all_z_features:
                        all_z_features[key] = []
                    all_z_features[key].append(value.cpu())

    # Concatenate the lists into full tensors
    for key in all_z_features:
        all_z_features[key] = torch.cat(all_z_features[key], dim=0)
    
    z_keys = sorted(all_z_features.keys())
    print(f"Successfully extracted the following features: {z_keys}")

    # --- 4. Compute correlation matrix ---
    print("Computing the correlation matrix between features...")
    num_features = len(z_keys)
    corr_matrix = np.zeros((num_features, num_features))

    for i in range(num_features):
        for j in range(num_features):
            if i < j:
                # Standardize features
                z1 = all_z_features[z_keys[i]]
                z2 = all_z_features[z_keys[j]]
                z1_norm = z1 - z1.mean(dim=0)
                z2_norm = z2 - z2.mean(dim=0)
                
                # Compute the Pearson correlation matrix, then take the mean of absolute values
                # This is one way to measure the overall correlation between two high-dimensional spaces
                cov = (z1_norm.T @ z2_norm) / (len(z1_norm) - 1)
                std1 = z1_norm.std(dim=0).unsqueeze(1)
                std2 = z2_norm.std(dim=0).unsqueeze(0)
                pearson_corr = cov / (std1 @ std2 + 1e-6)
                
                mean_abs_corr = torch.mean(torch.abs(pearson_corr)).item()
                corr_matrix[i, j] = mean_abs_corr
                corr_matrix[j, i] = mean_abs_corr
            elif i == j:
                corr_matrix[i, j] = 1.0

    # --- 5. Plot the heatmap ---
    print("Plotting the heatmap...")
    plt.figure(figsize=(10, 8))

    # --- Custom colormap ---
    cmap = LinearSegmentedColormap.from_list("custom_blue_red", ['#4169E1', '#FFFFFF', '#94070A'])

    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=.5,
        vmin=0,
        vmax=1,
        xticklabels=[key.replace('z_', '') for key in z_keys],
        yticklabels=[key.replace('z_', '') for key in z_keys]
    )
    #plt.title('Mean Absolute Correlation between Z Subspaces', fontsize=16)#
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    output_path = os.path.join(args.output_dir, "z_subspace_orthogonality_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Orthogonality analysis heatmap saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and visualize the orthogonality of Z feature subspaces")
    
    # --- Core arguments ---
    parser.add_argument('--data_path', type=str, required=True, help="Path to the .xlsx data file used for analysis")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model weights file (.pth)")
    parser.add_argument('--model_config_path', type=str, required=True, help="Path to the model_config.pth file saved during training")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the heatmap")
    
    # --- Data column-name arguments ---
    parser.add_argument('--image_col_left', type=str, required=True, help="Column name of the left-eye image path")
    parser.add_argument('--image_col_right', type=str, required=True, help="Column name of the right-eye image path")
    
    # --- Analysis and loading arguments ---
    parser.add_argument('--num_samples_for_analysis', type=int, default=2000, help="Number of samples used for analysis to speed things up")
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    analyze_and_plot_orthogonality(args)
