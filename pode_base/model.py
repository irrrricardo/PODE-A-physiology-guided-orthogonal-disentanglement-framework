# model.py

import torch
import torch.nn as nn
import timm


class AgePredictionViT(nn.Module):
    """
    ViT model architecture for age prediction, with configurable Dropout.
    """

    def __init__(self,
                 model_name='vit_base_patch16_224',
                 pretrained=False,
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 head_dropout_rate=0.5):
        """
        Args:
            model_name (str): Name of the ViT model in the timm library.
            pretrained (bool): Whether to load ImageNet pretrained weights.
            drop_rate (float): Dropout rate for Attention and MLP blocks in ViT backbone.
            drop_path_rate (float): Stochastic Depth droppath rate.
            head_dropout_rate (float): Dropout rate in the custom regression head.
        """
        super().__init__()

        # ------------------ 1. Load ViT backbone with Dropout support ------------------
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Return feature vector instead of classification results
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )

        # ------------------ 2. Define regression head with Dropout ------------------
        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.vit.embed_dim),
            nn.Dropout(head_dropout_rate),
            nn.Linear(self.vit.embed_dim, 1)
        )

    def forward(self, x):
        features = self.vit(x)
        predicted_age = self.regression_head(features)
        return predicted_age


def load_mae_weights(model, mae_weights_path):
    """
    A more robust weight loading function for loading complex self-supervised learning checkpoints,
    capable of precisely handling various prefixes like 'module.' and 'backbone.'.
    """
    print(f"Loading MAE pretrained weights from '{mae_weights_path}'...")

    checkpoint = torch.load(mae_weights_path, map_location='cpu', weights_only=False)

    # --- 1. Smart extraction of weight dictionary ---
    # Priority order: student > teacher > model > state_dict > encoder > bare dict
    if 'student' in checkpoint:
        print("Found 'student' key in checkpoint, will use weights from this part.")
        weights = checkpoint['student']
    elif 'teacher' in checkpoint:
        print("Found 'teacher' key in checkpoint, will use weights from this part.")
        weights = checkpoint['teacher']
    elif 'model' in checkpoint:
        print("Found 'model' key in checkpoint, will use weights from this part.")
        weights = checkpoint['model']
    elif 'state_dict' in checkpoint:
        # Common in PyTorch-Lightning, timm, and many other frameworks
        print("Found 'state_dict' key in checkpoint, will use weights from this part.")
        weights = checkpoint['state_dict']
    elif 'encoder' in checkpoint:
        # Some MAE implementations save the encoder separately
        print("Found 'encoder' key in checkpoint, will use weights from this part.")
        weights = checkpoint['encoder']
    else:
        print("Common nested keys not found, will try to load the entire file as weights directly.")
        weights = checkpoint

    # --- 2. Key fix: Robustly clean up weight key names ---
    # Strip common prefixes used by different training frameworks, then check if
    # the resulting key exists in the target ViT backbone state dict.
    KNOWN_PREFIXES = [
        'module.backbone.',   # DDP + backbone wrapper
        'backbone.',          # generic backbone wrapper
        'module.encoder.',    # DDP + encoder wrapper
        'encoder.',           # MAE encoder-only saves
        'module.base_model.', # DDP + HuggingFace-style wrapper
        'base_model.',        # HuggingFace PEFT style
        'module.vit.',        # DDP + our own PODE-Base format
        'vit.',               # our own PODE-Base format (regression_head excluded)
        'module.',            # plain DDP
    ]

    cleaned_weights = {}
    for k, v in weights.items():
        new_key = k
        # Try each prefix in order; stop at the first match
        for prefix in KNOWN_PREFIXES:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break  # only strip one prefix layer

        # Accept the key if it exists verbatim in the target ViT state dict
        if new_key in model.vit.state_dict():
            cleaned_weights[new_key] = v

    if not cleaned_weights:
        print("Critical warning: No weight keys matching the model were found after cleanup. Please check the weight file and model structure!")
        return model

    # --- 3. Load the processed weights ---
    # Now we only load exactly matching backbone weights to the vit part of the model
    missing_keys, unexpected_keys = model.vit.load_state_dict(cleaned_weights, strict=False)

    print("Weight loading complete.")
    if missing_keys:
        # In this precise loading mode, theoretically there should be no missing keys belonging to the ViT backbone
        print(f"Warning: Still missing the following ViT backbone keys: {missing_keys}")
    if unexpected_keys:
        # Having unexpected keys is normal, because cleaned_weights may contain parts not needed by the model
        # (although our logic has filtered most of them, there may still be some)
        print(f"Found unexpected keys (usually normal): {unexpected_keys}")

    # Check loading results
    if not missing_keys:
        print("✅ All weights successfully matched and loaded!")
    elif len(missing_keys) < 10:  # If only a few missing, loading is basically successful
        print("✅ Most weights loaded successfully!")
    else:
        print("❌ Warning: It seems most weights failed to load. Please carefully check the model architecture and weight file again.")

    return model