"""
Fisher Information Matrix for Elastic Weight Consolidation (EWC).

Tracks which weights are important for previously learned tasks
and penalizes changes to those weights.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import copy


class FisherInformationMatrix:
    """
    Computes and stores Fisher Information Matrix for EWC.
    
    The Fisher diagonal approximates the importance of each weight:
    - High Fisher value = important for past tasks (should be preserved)
    - Low Fisher value = less important (can be modified)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        self.device = device
        
        # Store Fisher diagonal and optimal weights
        self.fisher_diagonal: Dict[str, torch.Tensor] = {}
        self.optimal_weights: Dict[str, torch.Tensor] = {}
        
        # Initialize to zeros
        self._initialize_fisher(model)
    
    def _initialize_fisher(self, model: nn.Module) -> None:
        """Initialize Fisher diagonal to zeros."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.fisher_diagonal[name] = torch.zeros_like(param, device=self.device)
                self.optimal_weights[name] = param.data.clone()
    
    def compute_fisher(
        self,
        model: nn.Module,
        dataloader,
        num_samples: int = 1000,
    ) -> None:
        """
        Compute Fisher Information diagonal using sampled gradients.
        
        Fisher approximation: E[(∇log p(y|x))²]
        
        Args:
            model: Neural network
            dataloader: Iterator of (state, action) pairs
            num_samples: Number of samples to use
        """
        model.eval()
        
        # Reset Fisher diagonal
        for name in self.fisher_diagonal:
            self.fisher_diagonal[name].zero_()
        
        samples_processed = 0
        
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            states = batch[0].to(self.device)
            batch_size = states.shape[0]
            
            # Forward pass
            outputs, _ = model(states)
            
            # For each output dimension, compute gradient
            for i in range(outputs.shape[1]):
                model.zero_grad()
                
                # Gradient of output w.r.t. weights
                outputs[:, i].sum().backward(retain_graph=(i < outputs.shape[1] - 1))
                
                # Accumulate squared gradients (Fisher diagonal)
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher_diagonal[name] += param.grad.data ** 2
            
            samples_processed += batch_size
        
        # Normalize by number of samples
        for name in self.fisher_diagonal:
            self.fisher_diagonal[name] /= samples_processed
    
    def update_optimal_weights(self, model: nn.Module) -> None:
        """Store current weights as optimal (after consolidation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_weights[name] = param.data.clone()
    
    def decay_fisher(self, decay_rate: float = 0.77) -> None:
        """
        Decay Fisher values over time.
        
        This implements "Fisher saturation" - old tasks become less protected
        to allow new learning.
        
        Args:
            decay_rate: Multiplicative decay factor (0.77 from POC experiments)
        """
        for name in self.fisher_diagonal:
            self.fisher_diagonal[name] *= decay_rate
    
    def get_ewc_loss(
        self,
        model: nn.Module,
        lambda_ewc: float = 155000.0,
    ) -> torch.Tensor:
        """
        Compute EWC loss: λ/2 * Σ F_i * (θ_i - θ*_i)²
        
        Args:
            model: Current model
            lambda_ewc: EWC strength (155,000 from POC experiments)
            
        Returns:
            EWC penalty loss
        """
        loss = torch.tensor(0.0, device=self.device)
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher_diagonal:
                fisher = self.fisher_diagonal[name]
                optimal = self.optimal_weights[name]
                
                # EWC penalty: F * (θ - θ*)²
                loss += (fisher * (param - optimal) ** 2).sum()
        
        return (lambda_ewc / 2) * loss
    
    def get_stats(self) -> Dict:
        """Get Fisher statistics."""
        if not self.fisher_diagonal:
            return {}
        
        # Concatenate all Fisher values
        all_fisher = torch.cat([f.flatten() for f in self.fisher_diagonal.values()])
        
        return {
            'mean_fisher': all_fisher.mean().item(),
            'max_fisher': all_fisher.max().item(),
            'min_fisher': all_fisher.min().item(),
            'std_fisher': all_fisher.std().item(),
            'num_saturated': (all_fisher > 1000).sum().item(),  # High protection
            'num_params': len(all_fisher),
        }
    
    def save(self, path: str) -> None:
        """Save Fisher matrix to file."""
        torch.save({
            'fisher_diagonal': self.fisher_diagonal,
            'optimal_weights': self.optimal_weights,
        }, path)
    
    def load(self, path: str) -> None:
        """Load Fisher matrix from file."""
        checkpoint = torch.load(path)
        self.fisher_diagonal = checkpoint['fisher_diagonal']
        self.optimal_weights = checkpoint['optimal_weights']
