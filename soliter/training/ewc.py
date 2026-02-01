"""
Elastic Weight Consolidation (EWC) Loss.

Prevents catastrophic forgetting by penalizing changes to important weights.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..memory.fisher_matrix import FisherInformationMatrix


class EWCLoss:
    """
    EWC loss computation combining task loss and weight protection.
    
    Total Loss = Task Loss + λ * EWC Penalty
    """
    
    def __init__(
        self,
        fisher_matrix: Optional[FisherInformationMatrix] = None,
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
        Compute total loss with EWC penalty.
        
        Args:
            model: Neural network
            task_loss: Current task loss (e.g., MSE, RL loss)
            
        Returns:
            Total loss = task_loss + ewc_penalty
        """
        if self.fisher_matrix is None:
            return task_loss
        
        ewc_penalty = self.fisher_matrix.get_ewc_loss(model, self.lambda_ewc)
        
        return task_loss + ewc_penalty
    
    def get_loss_components(
        self,
        model: nn.Module,
        task_loss: torch.Tensor,
    ) -> dict:
        """Get individual loss components for logging."""
        if self.fisher_matrix is None:
            return {
                'task_loss': task_loss.item(),
                'ewc_loss': 0.0,
                'total_loss': task_loss.item(),
            }
        
        ewc_penalty = self.fisher_matrix.get_ewc_loss(model, self.lambda_ewc)
        
        return {
            'task_loss': task_loss.item(),
            'ewc_loss': ewc_penalty.item(),
            'total_loss': (task_loss + ewc_penalty).item(),
        }
