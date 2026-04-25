import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from typing import Dict, Optional

def orthogonal_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute the orthogonal loss between two batches of feature vectors.
    """
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    cross_corr_matrix = torch.matmul(z1.T, z2)
    loss = cross_corr_matrix.pow(2).sum()
    loss = loss / z1.size(0)
    return loss

class DisentanglementLoss_V2(nn.Module):
    """
    Total loss for the V2 architecture.
    Supports pairwise orthogonality constraints and configurable task weights.
    """
    def __init__(self,
                 lambda_age_orth: float = 1.0,
                 lambda_physio_orth: float = 0.1,
                 task_weights: Optional[Dict[str, float]] = None,
                 task_loss_fn=None,
                 base_age_loss_weight: float = 1.0,
                 final_age_loss_weight: float = 1.0):
        """
        Args:
            lambda_age_orth (float): Weight of the orthogonal loss between the Age subspace
                                     and the other physiological subspaces.
            lambda_physio_orth (float): Weight of the orthogonal loss between physiological
                                        subspaces themselves.
            task_weights (Dict[str, float], optional): A dictionary specifying loss weights
                                                       for each task. Default is None, i.e.
                                                       all weights are 1.0.
            task_loss_fn (nn.Module, optional): Loss function for the supervised tasks.
                                                Default is nn.MSELoss(reduction='none').
            base_age_loss_weight (float): Weight for the base_age loss.
            final_age_loss_weight (float): Weight for the final_age loss.
        """
        super().__init__()
        self.lambda_age_orth = lambda_age_orth
        self.lambda_physio_orth = lambda_physio_orth
        self.task_weights = task_weights if task_weights is not None else {}
        self.task_loss_fn = task_loss_fn if task_loss_fn is not None else nn.MSELoss(reduction='none')
        self.base_age_loss_weight = base_age_loss_weight
        self.final_age_loss_weight = final_age_loss_weight
        print(f"DisentanglementLoss_V2 initialized: lambda_age_orth={self.lambda_age_orth}, lambda_physio_orth={self.lambda_physio_orth}, "
              f"base_age_weight={self.base_age_loss_weight}, final_age_weight={self.final_age_loss_weight}, task_weights={self.task_weights}")

    def forward(self, predictions: dict, targets: dict, activate_orth_loss: bool = True) -> dict:
        """
        Compute the total loss.

        Args:
            predictions (dict): Output dictionary from the model.
            targets (dict): Dictionary of ground-truth labels.
            activate_orth_loss (bool): Whether to activate the orthogonal loss. Used for
                                       three-stage training.
        """
        device = next((v.device for v in predictions.values() if isinstance(v, torch.Tensor)), 'cpu')
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # --- 1. Compute the weighted task loss (Weighted Task Loss) ---
        task_loss = torch.tensor(0.0, device=device)
        
        # Specifically handle the age losses (base_age and final_age)
        if 'age' in targets:
            age_target = targets['age']
            total_age_loss = torch.tensor(0.0, device=device)
            sample_weights = targets.get('sample_weight')

            # Compute base_age loss
            if 'pred_age' in predictions and self.base_age_loss_weight > 0:
                base_age_pred = predictions['pred_age']
                loss_base_age_unreduced = self.task_loss_fn(base_age_pred, age_target)

                if sample_weights is not None:
                    if sample_weights.shape != loss_base_age_unreduced.shape:
                        sample_weights = sample_weights.squeeze()
                    weighted_loss_base_age = loss_base_age_unreduced * sample_weights
                    loss_base_age = weighted_loss_base_age.mean()
                else:
                    loss_base_age = loss_base_age_unreduced.mean()

                weighted_loss_base = self.base_age_loss_weight * loss_base_age
                total_age_loss += weighted_loss_base
                loss_dict['loss_base_age'] = loss_base_age.item()
                loss_dict['w_loss_base_age'] = weighted_loss_base.item()

            # Compute final_age loss (anchored to Teacher Pred)
            if 'final_age' in predictions and 'teacher_age' in targets and self.final_age_loss_weight > 0:
                final_age_pred = predictions['final_age']
                # [Core change] The supervision target is switched to teacher_age
                teacher_target = targets['teacher_age']
                loss_final_age_unreduced = self.task_loss_fn(final_age_pred, teacher_target)

                if sample_weights is not None:
                    if sample_weights.shape != loss_final_age_unreduced.shape:
                        sample_weights = sample_weights.squeeze()
                    weighted_loss_final_age = loss_final_age_unreduced * sample_weights
                    loss_final_age = weighted_loss_final_age.mean()
                else:
                    loss_final_age = loss_final_age_unreduced.mean()

                weighted_loss_final = self.final_age_loss_weight * loss_final_age
                total_age_loss += weighted_loss_final
                # Update the keys in loss_dict to reflect the supervision target
                loss_dict['loss_final_age_vs_teacher'] = loss_final_age.item()
                loss_dict['w_loss_final_age_vs_teacher'] = weighted_loss_final.item()

            task_loss += total_age_loss
            loss_dict['task_loss_age_total'] = total_age_loss.item()

        # Handle losses for the other physiological indicators
        for group_name in predictions.get('feature_groups', []):
            if group_name == 'age':  # Age loss has already been handled above
                continue

            pred_key = f'pred_{group_name}'
            target_key = group_name
            
            if pred_key in predictions and target_key in targets:
                pred = predictions[pred_key]
                target = targets[target_key]

                # --- Core change: use masking to handle NaN ---
                # 1. Build a boolean mask, True means valid (non-NaN)
                valid_mask = ~torch.isnan(target)

                # 2. If the entire batch has no valid value, skip the loss for this task
                if not valid_mask.any():
                    loss_dict[f'loss_{group_name}'] = 0.0
                    continue

                # 3. Filter valid predictions and targets according to the mask
                valid_pred = pred[valid_mask]
                valid_target = target[valid_mask]

                # 4. Compute the loss on the valid values
                # Note: since we have already filtered the elements, the reduction here should be 'mean'
                loss_fn_masked = nn.MSELoss(reduction='mean')
                final_loss_group = loss_fn_masked(valid_pred, valid_target)
                
                # Apply the task weight
                task_weight = self.task_weights.get(group_name, 1.0)
                weighted_task_loss = task_weight * final_loss_group
                
                task_loss += weighted_task_loss
                loss_dict[f'loss_{group_name}'] = final_loss_group.item() # record the original loss
                if task_weight != 1.0:
                    loss_dict[f'w_loss_{group_name}'] = weighted_task_loss.item() # record the weighted loss

        loss_dict['task_loss_total'] = task_loss.item()
        total_loss += task_loss

        # --- 2. Compute the hierarchical orthogonal loss (Hierarchical Orthogonal Loss) ---
        if activate_orth_loss:
            total_ortho_loss = torch.tensor(0.0, device=device)
            
            # Part A: Age vs. Physio Orthogonal Loss
            if self.lambda_age_orth > 0 and 'z_age' in predictions:
                age_ortho_loss = torch.tensor(0.0, device=device)
                z_age = predictions['z_age']
                physio_names = [name for name in predictions.get('feature_groups', []) if name != 'age']
                
                for name in physio_names:
                    z_physio_key = f'z_{name}'
                    if z_physio_key in predictions:
                        z_physio = predictions[z_physio_key]
                        loss_pair = orthogonal_loss(z_age, z_physio)
                        age_ortho_loss += loss_pair
                        loss_dict[f'ortho_age_{name}'] = loss_pair.item()
                
                if age_ortho_loss > 0:
                    loss_dict['ortho_loss_age_vs_physio'] = age_ortho_loss.item()
                    total_ortho_loss += self.lambda_age_orth * age_ortho_loss

            # Part B: Physio vs. Physio Orthogonal Loss
            if self.lambda_physio_orth > 0:
                physio_ortho_loss = torch.tensor(0.0, device=device)
                physio_names = [name for name in predictions.get('feature_groups', []) if name != 'age']

                for name1, name2 in combinations(physio_names, 2):
                    z1_key, z2_key = f'z_{name1}', f'z_{name2}'
                    if z1_key in predictions and z2_key in predictions:
                        z1, z2 = predictions[z1_key], predictions[z2_key]
                        loss_pair = orthogonal_loss(z1, z2)
                        physio_ortho_loss += loss_pair
                        loss_dict[f'ortho_{name1}_{name2}'] = loss_pair.item()

                if physio_ortho_loss > 0:
                    loss_dict['ortho_loss_physio_vs_physio'] = physio_ortho_loss.item()
                    total_ortho_loss += self.lambda_physio_orth * physio_ortho_loss
            
            if total_ortho_loss > 0:
                loss_dict['ortho_loss_total'] = total_ortho_loss.item()
                total_loss += total_ortho_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict
