# dataset.py - Tabular Dataset for Tabular Transformer

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular clinical data.

    Args:
        X: numpy array of shape (n_samples, n_features) — normalized features
        y: numpy array of shape (n_samples,) — target values (True Age)
    """

    def __init__(self, X: np.ndarray, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Accept numpy array, pandas Series, or any array-like
        y_array = np.array(y, dtype=np.float32)
        self.y = torch.tensor(y_array, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
