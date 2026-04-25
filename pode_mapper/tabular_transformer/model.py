# model.py - Tabular Transformer for clinical indicator analysis
# Supp Fig 2: Attention heatmap showing feature-to-feature associations

import torch
import torch.nn as nn
import math


class FeatureTokenizer(nn.Module):
    """
    Convert each scalar tabular feature into a d_model-dimensional token embedding.
    Each feature gets its own independent linear projection.
    """
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        # One Linear(1 → d_model) per feature
        self.embeddings = nn.ModuleList([
            nn.Linear(1, d_model, bias=True) for _ in range(n_features)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_features)  — raw normalized feature values
        Returns:
            tokens: (batch_size, n_features, d_model)
        """
        tokens = [embed(x[:, i:i+1]) for i, embed in enumerate(self.embeddings)]
        tokens = torch.stack(tokens, dim=1)   # (B, n_features, d_model)
        return self.norm(tokens)


class AttentionBlock(nn.Module):
    """
    A single Transformer encoder block that explicitly stores attention weights.
    Attention weights can be extracted after each forward pass.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        # Stored after each forward pass — shape: (batch, n_features, n_features)
        self.last_attn_weights: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x: same shape
        """
        attn_out, attn_weights = self.attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=True   # average over heads → (B, seq, seq)
        )
        # Store for later extraction
        self.last_attn_weights = attn_weights.detach().cpu()

        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TabularTransformer(nn.Module):
    """
    Tabular Transformer for clinical indicator analysis.

    Architecture:
        1. FeatureTokenizer: each feature → d_model embedding
        2. N AttentionBlocks: multi-head self-attention across feature tokens
        3. Mean pooling over feature tokens
        4. Regression head → predict target (True Age)

    After training, use get_attention_matrix() to extract the
    averaged self-attention weights for Supp Fig 2.
    """

    def __init__(
        self,
        feature_names: list,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.d_model = d_model

        self.tokenizer = FeatureTokenizer(self.n_features, d_model)

        self.blocks = nn.ModuleList([
            AttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_features) — normalized feature values
        Returns:
            pred: (batch_size, 1)
        """
        tokens = self.tokenizer(x)          # (B, n_features, d_model)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = tokens.mean(dim=1)         # (B, d_model)
        return self.head(pooled)            # (B, 1)

    def get_attention_matrix(
        self,
        dataloader,
        device: torch.device,
        n_batches: int = None
    ) -> torch.Tensor:
        """
        Run inference on dataloader and return the averaged attention weight matrix.

        Args:
            dataloader: DataLoader of TabularDataset (no shuffle)
            device: torch device
            n_batches: if set, only use first N batches (for speed)

        Returns:
            avg_attn: (n_features, n_features) — averaged over layers, heads, samples
        """
        self.eval()
        # Accumulate per-layer attention: list of (n_features, n_features)
        layer_attn_sum = [
            torch.zeros(self.n_features, self.n_features)
            for _ in self.blocks
        ]
        total_samples = 0

        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                if n_batches is not None and i >= n_batches:
                    break
                x = x.to(device)
                _ = self(x)
                batch_size = x.size(0)
                total_samples += batch_size

                for layer_idx, block in enumerate(self.blocks):
                    # block.last_attn_weights: (B, n_features, n_features)
                    layer_attn_sum[layer_idx] += (
                        block.last_attn_weights.sum(dim=0)  # sum over batch
                    )

        # Average over samples and layers
        avg_attn = torch.zeros(self.n_features, self.n_features)
        for layer_sum in layer_attn_sum:
            avg_attn += layer_sum / total_samples
        avg_attn /= len(self.blocks)

        return avg_attn   # (n_features, n_features)


def save_checkpoint(model: TabularTransformer, path: str, extra_info: dict = None):
    """Save model checkpoint with metadata."""
    ckpt = {
        'state_dict': model.state_dict(),
        'feature_names': model.feature_names,
        'd_model': model.d_model,
        'config': {
            'n_features': model.n_features,
            'd_model': model.d_model,
        }
    }
    if extra_info:
        ckpt.update(extra_info)
    torch.save(ckpt, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, device: torch.device = None) -> TabularTransformer:
    """Load model from checkpoint."""
    if device is None:
        device = torch.device('cpu')
    ckpt = torch.load(path, map_location=device)
    model = TabularTransformer(
        feature_names=ckpt['feature_names'],
        d_model=ckpt['config']['d_model'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    print(f"Model loaded from {path}")
    return model
