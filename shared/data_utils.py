# data_utils.py

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageFile  # Import ImageFile
import torch
import torchvision.transforms as T

# --- Key modification: Enable support for truncated images globally ---
# Warning: This may load incomplete images with gray bars,
# which may have a negative impact on model training.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AgeDataset(Dataset):
    """
    PyTorch Dataset class for age prediction task.
    """
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a sample according to index idx.
        Modified: Return None if image loading fails.
        """
        record = self.df.iloc[idx]
        image_path = record['image_path']
        age = torch.tensor(record['age'], dtype=torch.float32)

        try:
            # Load image from local path
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # If loading still fails even after enabling LOAD_TRUNCATED_IMAGES
            print(f"Warning: Failed to load image: {image_path}. Error: {e}. This sample will be skipped.")
            return None  # Return None, handled by collate_fn

        # Apply image transforms
        if self.transform:
            image = self.transform(image)

        return image, age


def get_transforms(image_size: int, is_train: bool = True):
    """
    Get image transformation pipeline for training or validation/testing.
    Modified: Only use random horizontal and vertical flips during training.
    """
    # During validation and testing, only resize and normalize are needed
    if not is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # During training, use simpler augmentation more suitable for medical imaging
    return T.Compose([
        # Note: If the second part of offline preprocessing is executed,
        # T.Resize here is actually processing already scaled images,
        # its role is to ensure the size is exactly consistent, with minimal overhead.
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),  # <-- New vertical flip
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.RandomRotation(degrees=(-15, 15)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def collate_fn_skip_corrupted(batch):
    """
    Custom collate_fn to filter out corrupted samples that return None in __getitem__.
    """
    # Filter out items that are None in the batch
    batch = list(filter(lambda x: x is not None, batch))
    # If the entire batch is empty after filtering, return empty tensors
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    # Otherwise, use default collate behavior to pack remaining samples
    return torch.utils.data.dataloader.default_collate(batch)