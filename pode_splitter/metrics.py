# metrics.py

import os
import csv
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List

class MultiTaskMetricsLogger:
    """
    An evaluation logger for multi-task regression models.

    This class is able to:
    1. Receive a dictionary of model predictions and a dictionary of targets.
    2. Independently compute a full set of regression metrics for each task (key) in the dictionaries.
    3. Clearly print the performance of each task to the console.
    4. Log all evaluation results in a structured CSV file for later analysis.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the logger.

        Args:
            output_dir (str): Directory where the log file will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.log_file = os.path.join(output_dir, "evaluation_log.csv")
        self.fieldnames = [
            'epoch', 'phase', 'task_name', 'mae', 'mse', 'rmse',
            'pearson_corr', 'r2_score'
        ]

        # Write the header of the CSV file
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Compute all required metrics for a single task.
        Adds NaN handling.
        """
        # --- Core change: use masking to handle NaN ---
        # 1. Build a boolean mask, True means valid (non-NaN)
        valid_mask = ~np.isnan(targets)

        # 2. If there is no valid value at all, return a set of default values
        if not np.any(valid_mask):
            return {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'pearson_corr': 0.0, 'r2_score': 0.0}

        # 3. Filter valid predictions and targets according to the mask
        valid_preds = predictions[valid_mask].flatten()
        valid_targets = targets[valid_mask].flatten()

        # 4. Compute metrics on the valid values
        mae = mean_absolute_error(valid_targets, valid_preds)
        mse = mean_squared_error(valid_targets, valid_preds)
        rmse = np.sqrt(mse)
        
        # Compute Pearson correlation, handling the special case where the std is 0
        pearson_corr_val: float = 0.0
        if np.std(valid_preds) > 0 and np.std(valid_targets) > 0:
            # pearsonr returns a tuple (correlation, p-value)
            corr, _ = pearsonr(valid_targets, valid_preds)
            pearson_corr_val = float(corr)
        else:
            pearson_corr_val = 0.0

        r2 = r2_score(valid_targets, valid_preds)

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'pearson_corr': pearson_corr_val,
            'r2_score': float(r2)
        }

    def log(self,
            epoch: int,
            phase: str,
            all_predictions: Dict[str, List[torch.Tensor]],
            all_targets: Dict[str, List[torch.Tensor]]):
        """
        Evaluate, print and log all tasks for one epoch.

        Args:
            epoch (int): Current epoch number.
            phase (str): 'train' or 'val'.
            all_predictions (Dict): Dictionary containing the list of predictions for each task.
            all_targets (Dict): Dictionary containing the list of targets for each task.
        """
        print("-" * 80)
        print(f"Epoch {epoch + 1} - {phase.upper()} Phase Evaluation")
        print("-" * 80)

        # Iterate over all tasks for evaluation
        for task_name in all_targets.keys():
            if task_name not in all_predictions or not all_predictions[task_name]:
                print(f"  Task: {task_name: <20} | No valid predictions found, skipping evaluation.")
                continue

            # Concatenate the list of tensors into a single big tensor and convert to numpy
            preds_tensor = torch.cat(all_predictions[task_name], dim=0).cpu().numpy()
            targets_tensor = torch.cat(all_targets[task_name], dim=0).cpu().numpy()

            # If a task has multiple output dimensions (e.g., hemodynamics), evaluate each separately
            num_dims = targets_tensor.shape[1] if targets_tensor.ndim > 1 else 1
            
            for i in range(num_dims):
                dim_preds = preds_tensor[:, i] if num_dims > 1 else preds_tensor
                dim_targets = targets_tensor[:, i] if num_dims > 1 else targets_tensor
                
                metrics = self.calculate_metrics(dim_preds, dim_targets)
                
                # Determine the task name used in the log
                log_task_name = f"{task_name}_{i}" if num_dims > 1 else task_name

                # Print to console
                print(f"  Task: {log_task_name: <20} | "
                      f"MAE: {metrics['mae']:.4f} | "
                      f"RMSE: {metrics['rmse']:.4f} | "
                      f"Pearson: {metrics['pearson_corr']:.4f} | "
                      f"R²: {metrics['r2_score']:.4f}")

                # Log to the CSV file
                log_data = {
                    'epoch': epoch + 1,
                    'phase': phase,
                    'task_name': log_task_name,
                    **metrics
                }
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writerow(log_data)
        print("-" * 80)
