# metrics.py

import os
import csv
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MetricsLogger:
    """
    A class for calculating, printing, and recording metrics for regression tasks.
    """

    def __init__(self, output_dir):
        """
        Initialize the logger.

        Args:
            output_dir (str): Directory for saving log files.
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.log_file = os.path.join(output_dir, "training_log.csv")
        self.fieldnames = [
            'epoch', 'phase', 'mae', 'mse', 'rmse',
            'pearson_corr', 'r2_score'
        ]

        # Write CSV file header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def calculate_metrics(self, predictions, targets):
        """
        Calculate all required metrics.

        Args:
            predictions (np.array): Model's predicted values.
            targets (np.array): Ground truth labels.

        Returns:
            dict: Dictionary containing all metric names and values.
        """
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        # Calculate Pearson correlation coefficient, return 0 if standard deviation is 0
        if np.std(predictions) > 0 and np.std(targets) > 0:
            pearson_corr, _ = pearsonr(targets.flatten(), predictions.flatten())
        else:
            pearson_corr = 0.0

        r2 = r2_score(targets, predictions)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'pearson_corr': pearson_corr,
            'r2_score': r2
        }

    def log(self, epoch, phase, predictions, targets):
        """
        Calculate, print, and log metrics for one epoch.

        Args:
            epoch (int): Current epoch number.
            phase (str): 'train' or 'val'.
            predictions (torch.Tensor): Model's predicted tensor.
            targets (torch.Tensor): Ground truth tensor.

        Returns:
            float: Returns MAE, typically used for early stopping strategy and model saving.
        """
        # Convert Tensor to Numpy array
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Calculate metrics
        metrics = self.calculate_metrics(preds_np, targets_np)

        # Print to console
        print(f"Epoch {epoch + 1} - {phase.upper()} | "
              f"MAE: {metrics['mae']:.4f} | "
              f"RMSE: {metrics['rmse']:.4f} | "
              f"Pearson Corr: {metrics['pearson_corr']:.4f} | "
              f"R² Score: {metrics['r2_score']:.4f}")

        # Log to CSV file
        log_data = {
            'epoch': epoch + 1,
            'phase': phase,
            **metrics
        }
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(log_data)

        return metrics['mae']