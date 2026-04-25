import torch
import torch.nn as nn
from typing import Dict, Any

class Identity(nn.Module):
    """An identity mapping module, used to keep the original features when frozen."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class DisentangledVisionFM_V2(nn.Module):
    """
    Fully Orthogonal Disentanglement Model - V2 architecture.
    All feature dimensions (including age) are treated equally, each having its own
    orthogonal subspace.
    """

    def __init__(self,
                 backbone: nn.Module,
                 embed_dim: int,
                 feature_groups: Dict[str, Dict[str, Any]],
                 head_dropout_rate: float = 0.5):
        """
        Args:
            backbone (nn.Module): Feature extractor (ViT).
            embed_dim (int): Output feature dimension of the backbone (e.g., 768).
            feature_groups (Dict): Configuration dictionary for the disentangled subspaces.
                                   'age' is now also a member.
            head_dropout_rate (float): Dropout probability of the prediction heads.
        """
        super().__init__()
        self.backbone = backbone
        self.feature_groups_config = feature_groups

        print("--- Initializing DisentangledVisionFM (V2 - Fully Orthogonal Disentanglement Architecture) ---")

        self.projectors = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.age_delta_heads = nn.ModuleDict()

        for name, config in feature_groups.items():
            proj_dim = config['dim']
            output_dim = config['output_dim']

            # --- 1. Create projector ---
            if name == 'age':
                if proj_dim != embed_dim:
                    raise ValueError(f"Age projector 'dim' must be equal to embed_dim ({embed_dim})")
                projector = nn.Linear(embed_dim, proj_dim)
                print(f"  - Created Age subspace 'age': Z_dim={proj_dim} (trainable identity mapping, no LayerNorm)")
            else:
                projector = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, proj_dim),
                    nn.GELU(),
                    nn.LayerNorm(proj_dim)
                )
                print(f"  - Created physiological subspace '{name}': Z_dim={proj_dim}")
            self.projectors[name] = projector

            # --- 2. Create main task prediction head ---
            head = nn.Sequential(
                nn.LayerNorm(proj_dim),
                nn.Dropout(head_dropout_rate),
                nn.Linear(proj_dim, output_dim)
            )
            self.heads[name] = head
            print(f"    - Created main-task head for '{name}': Input_dim={proj_dim}, Output_dim={output_dim}")

            # --- 3. Create Age Delta prediction head for physiological subspaces ---
            if name != 'age':
                age_delta_head = nn.Sequential(
                    nn.LayerNorm(proj_dim),
                    nn.Linear(proj_dim, 1)
                )
                self.age_delta_heads[name] = age_delta_head
                print(f"    - Created Age Delta head for '{name}': Input_dim={proj_dim}, Output_dim=1")

        print("----------------------------------------------------------")

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass of the V2 architecture: parallel projection, independent prediction.
        """
        # --- 1. Extract global features ---
        z_left = self.backbone(x_left)
        z_right = self.backbone(x_right)
        z_global = (z_left + z_right) / 2.0

        output_dict = {'feature_groups': list(self.feature_groups_config.keys())}
        
        # --- 2. Project to all subspaces in parallel ---
        for name, projector in self.projectors.items():
            z_subspace = projector(z_global)
            output_dict[f'z_{name}'] = z_subspace

        # --- 3. Independently predict from each subspace ---
        for name, head in self.heads.items():
            z_subspace = output_dict[f'z_{name}']
            prediction = head(z_subspace)
            
            # Keep the same output format as the previous version
            if prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            
            output_dict[f'pred_{name}'] = prediction

        # --- 4. Compute Age Deltas and final age ---
        base_age = output_dict.get('pred_age', torch.tensor(0.0, device=z_global.device))
        total_delta = torch.zeros_like(base_age)

        for name, delta_head in self.age_delta_heads.items():
            z_subspace = output_dict[f'z_{name}']
            delta = delta_head(z_subspace).squeeze(-1)
            output_dict[f'delta_{name}'] = delta
            total_delta += delta
        
        output_dict['total_delta'] = total_delta
        output_dict['final_age'] = base_age + total_delta
        
        return output_dict
