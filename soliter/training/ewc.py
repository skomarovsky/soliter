"""
Elastic Weight Consolidation (EWC) Loss.

Combines task loss with EWC penalty to prevent catastrophic forgetting.
"""

import torch
import torch.nn as nn
from typing import Dict

from ..memory.fisher_matrix import FisherInformationMatrix


class EWCLoss:
    """
    EWC loss wrapper that combines task loss with Fisher-weighted penalty.
    
    Total Loss = Task Loss + (λ/2) × Σ F_i × (θ_i - θ*_i)²
    
    Args:
        fisher_matrix: FisherInformationMatrix instance
        lambda_ewc: EWC strength (higher = more protection against forgetting)
    """
    
    def __init__(
        self,
        fisher_matrix: FisherInformationMatrix,
        lambda_ewc: float = 155000.0,
    ):
        self.fisher_matrix = fisher_matrix
        self.lambda_ewc = lambda_ewc
    
    def compute_loss(
        self,
        model: nn.Module,
        task_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total loss = task_loss + EWC_penalty.
        
        Args:
            model: Neural network with current weights
            task_loss: The primary task loss (e.g., PPO loss)
            
        Returns:
            Total loss tensor
        """
        ewc_penalty = self.fisher_matrix.get_ewc_loss(model, self.lambda_ewc)
        return task_loss + ewc_penalty
    
    def get_ewc_penalty(self, model: nn.Module) -> torch.Tensor:
        """Get just the EWC penalty term."""
        return self.fisher_matrix.get_ewc_loss(model, self.lambda_ewc)
    
    def get_loss_components(
        self,
        model: nn.Module,
        task_loss: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Get individual loss components for logging.
        
        Returns:
            Dictionary with task_loss, ewc_loss, and total_loss
        """
        ewc_loss = self.fisher_matrix.get_ewc_loss(model, self.lambda_ewc)
        total_loss = task_loss + ewc_loss
        
        return {
            'task_loss': task_loss.item() if torch.is_tensor(task_loss) else task_loss,
            'ewc_loss': ewc_loss.item(),
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
        }
    
    def update_lambda(self, new_lambda: float) -> None:
        """Update the EWC strength parameter."""
        self.lambda_ewc = new_lambda