# pode_splitter/prediction_analysis/predict.py

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np
from typing import Dict, List

from ..model import DisentangledVisionFM_V2
from ...shared.data_utils import get_transforms

class InferenceDataset(Dataset):
    """A simple inference-only dataset."""
    def __init__(self, dataframe, image_col_left, image_col_right, transform=None):
        self.df = dataframe
        self.image_col_left = image_col_left
        self.image_col_right = image_col_right
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        path_left = record.get(self.image_col_left)
        path_right = record.get(self.image_col_right)
        
        try:
            img_left = Image.open(path_left).convert('RGB') if pd.notna(path_left) else Image.new('RGB', (224, 224))
            img_right = Image.open(path_right).convert('RGB') if pd.notna(path_right) else Image.new('RGB', (224, 224))
        except Exception as e:
            print(f"Warning: cannot load image {path_left} or {path_right}. Error: {e}. Using a blank image instead.")
            img_left, img_right = Image.new('RGB', (224, 224)), Image.new('RGB', (224, 224))

        if self.transform:
            img_left = self.transform(img_left)
            img_right = self.transform(img_right)
            
        return img_left, img_right

def predict_age_v2(dataframe: pd.DataFrame, args: argparse.Namespace) -> Dict[str, List[float]]:
    """
    (V2 Version) Run prediction with the trained model and return a dictionary of all predictions.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing image paths.
        args (argparse.Namespace): Namespace containing all required paths and parameters.

    Returns:
        Dict[str, List[float]]: A dictionary whose keys are column names ('Predicted_Age',
                                'Predicted_hemo_0', ...) and values are the corresponding
                                lists of predictions.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Starting V2 prediction pipeline, using device: {device} ---")

    # --- 1. Prepare the data loader ---
    inference_dataset = InferenceDataset(
        dataframe, 
        args.image_col_left, 
        args.image_col_right, 
        transform=get_transforms(args.image_size, is_train=False)
    )
    inference_loader = DataLoader(
        inference_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False
    )

    # --- 2. Load the model ---
    print("Rebuilding V2 model structure from the configuration file...")
    model_config = torch.load(args.model_config_path, map_location=device)
    backbone = timm.create_model(args.model_name, pretrained=False, num_classes=0)
    
    # Safely retrieve embed_dim from the model config; this matches the logic in the training script.
    # The dimension of the 'age' subspace is set equal to the backbone's embed_dim.
    embed_dim = model_config.get('age', {}).get('dim')
    if embed_dim is None:
        raise ValueError("Cannot find the 'age' subspace dimension ('dim') in model_config.pth. Please make sure the configuration file is valid.")

    model = DisentangledVisionFM_V2(
        backbone=backbone,
        embed_dim=embed_dim,
        feature_groups=model_config,
        head_dropout_rate=0.0
    )
    
    print(f"Loading V2 weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 3. Run inference ---
    predictions_collector: Dict[str, list] = {}

    print("Starting inference on the dataset...")
    with torch.no_grad():
        for images_left, images_right in tqdm(inference_loader, desc="V2 Predicting"):
            images_left = images_left.to(device)
            images_right = images_right.to(device)
            
            model_output = model(images_left, images_right)
            
            # --- Core change: collect all relevant model outputs ---
            # Including final age, base age, all deltas and other predictions
            keys_to_collect = [
                'final_age', 'pred_age', 'total_delta'
            ]
            # Dynamically add all delta_* and pred_* keys
            for k in model_output.keys():
                if k.startswith('delta_') or k.startswith('pred_'):
                    if k != 'pred_age': # pred_age is already included
                        keys_to_collect.append(k)
            
            for key in keys_to_collect:
                if key in model_output:
                    value = model_output[key]
                    if key not in predictions_collector:
                        predictions_collector[key] = []
                    predictions_collector[key].append(value.cpu().numpy())

    print("Inference finished.")
    
    # --- 4. Format the output ---
    final_predictions: Dict[str, list] = {}
    for key, batches in predictions_collector.items():
        # Concatenate the list of batches into a single big numpy array
        full_array = np.concatenate(batches, axis=0)
        
        # --- Core change: define clear column names for the different output types ---
        col_name = ""
        if key == 'final_age':
            col_name = 'Predicted_Age'
        elif key == 'pred_age':
            col_name = 'Base_Age'
        elif key == 'total_delta':
            col_name = 'Total_Age_Delta'
        elif key.startswith('delta_'):
            col_name = f"Age_Delta_{key.replace('delta_', '')}"
        elif key.startswith('pred_'):
            # Handle multi-dimensional physiological-indicator predictions
            if full_array.ndim > 1 and full_array.shape[1] > 1:
                for i in range(full_array.shape[1]):
                    multi_dim_col_name = f"Predicted_{key.replace('pred_', '')}_{i}"
                    final_predictions[multi_dim_col_name] = full_array[:, i].tolist()
                continue # skip the subsequent single-column processing
            else:
                col_name = f"Predicted_{key.replace('pred_', '')}"
        
        if col_name:
            final_predictions[col_name] = full_array.flatten().tolist()

    return final_predictions
