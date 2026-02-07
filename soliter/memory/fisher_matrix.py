"""
Fisher Information Matrix for Elastic Weight Consolidation (EWC).

FIXED VERSION: 
- Fisher now ACCUMULATES over sleep cycles instead of being overwritten.
- get_ewc_loss() now accepts lambda_ewc parameter for proper scaling.

Formula per sleep cycle:
    1. Decay existing Fisher: F = decay * F
    2. Compute new Fisher from current data: F_new
    3. Accumulate: F = F + F_new

EWC Loss Formula:
    L_EWC = (λ/2) × Σ_i F_i × (θ_i - θ*_i)²
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Iterable


class FisherInformationMatrix:
    """
    Computes and stores the diagonal Fisher Information Matrix.
    
    The Fisher Information approximates the curvature of the loss landscape,
    indicating which parameters are important for previously learned tasks.
    
    For EWC, we use this to penalize changes to important parameters,
    preventing catastrophic forgetting.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        self.device = device
        self.model_param_names = [name for name, _ in model.named_parameters()]
        
        # Initialize Fisher diagonal to zeros
        self.fisher_diagonal: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self.fisher_diagonal[name] = torch.zeros_like(param, device=device)
        
        # Store optimal weights (θ* in EWC formula)
        self.optimal_weights: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self.optimal_weights[name] = param.detach().clone().to(device)
        
        # Track computation history for debugging
        self._compute_count = 0
        self._last_computed_mean = 0.0
    
    def compute_fisher(
        self,
        model: nn.Module,
        dataloader: Iterable,
        num_samples: int = 1000,
    ) -> None:
        """
        Compute Fisher Information and ACCUMULATE with existing Fisher.
        
        Uses the empirical Fisher approximation:
        F_ii = E[(∂L/∂θ_i)²]
        
        IMPORTANT: This ADDS to existing Fisher, not replaces!
        Call decay_fisher() BEFORE this to prevent unbounded growth.
        
        Args:
            model: The neural network
            dataloader: Iterator yielding (state,) tuples
            num_samples: Number of samples to use
        """
        model.eval()
        
        # Temporary storage for this computation
        new_fisher: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            new_fisher[name] = torch.zeros_like(param, device=self.device)
        
        samples_processed = 0
        
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            states = batch[0].to(self.device)
            batch_size = states.shape[0]
            
            # Reset hidden state for clean forward pass
            if hasattr(model, 'reset_hidden'):
                model.reset_hidden(batch_size=batch_size, device=self.device)
            
            # Forward pass
            model.zero_grad()
            output, _ = model(states)
            
            # Use output variance as proxy for log-likelihood
            # (For policy networks, this approximates the Fisher)
            loss = output.pow(2).mean()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    new_fisher[name] += param.grad.detach().pow(2) * batch_size
            
            samples_processed += batch_size
        
        # Normalize by number of samples
        if samples_processed > 0:
            for name in new_fisher:
                new_fisher[name] /= samples_processed
        
        # ACCUMULATE: Add new Fisher to existing (decayed) Fisher
        for name in self.fisher_diagonal:
            self.fisher_diagonal[name] = self.fisher_diagonal[name] + new_fisher[name]
        
        # Track for debugging
        self._compute_count += 1
        all_new = torch.cat([f.flatten() for f in new_fisher.values()])
        self._last_computed_mean = all_new.mean().item()
        
        # Reset hidden state after computation
        if hasattr(model, 'reset_hidden'):
            model.reset_hidden(batch_size=1, device=self.device)
        
        model.train()
    
    def decay_fisher(self, decay_factor: float = 0.9) -> None:
        """
        Apply exponential decay to Fisher values.
        
        This prevents Fisher from growing unboundedly and allows
        the network to gradually "forget" very old task importance.
        
        CALL THIS BEFORE compute_fisher() each sleep cycle:
            1. decay_fisher(0.77)  # Decay old importance
            2. compute_fisher()    # Add new importance
        
        Args:
            decay_factor: Multiply all Fisher values by this (0-1)
        """
        for name in self.fisher_diagonal:
            self.fisher_diagonal[name] *= decay_factor
    
    def update_optimal_weights(self, model: nn.Module) -> None:
        """
        Update the optimal weights (θ*) to current model weights.
        
        Called after successful training/consolidation to mark
        current weights as the "reference point" for EWC penalty.
        """
        for name, param in model.named_parameters():
            self.optimal_weights[name] = param.detach().clone().to(self.device)
    
    def get_ewc_loss(
        self, 
        model: nn.Module, 
        lambda_ewc: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the EWC penalty term.
        
        L_EWC = (λ/2) × Σ_i F_i × (θ_i - θ*_i)²
        
        This penalizes deviations from optimal weights, weighted
        by parameter importance (Fisher) and scaled by lambda.
        
        Args:
            model: The neural network with current weights
            lambda_ewc: EWC strength (typically 1000-500000)
                       Higher = stronger protection against forgetting
                       
        Returns:
            Scalar tensor representing EWC loss
        """
        ewc_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in model.named_parameters():
            if name in self.fisher_diagonal and name in self.optimal_weights:
                fisher = self.fisher_diagonal[name]
                optimal = self.optimal_weights[name]
                
                # F_i * (θ_i - θ*_i)²
                ewc_loss += (fisher * (param - optimal).pow(2)).sum()
        
        # Apply lambda scaling (λ/2 factor from original EWC paper)
        ewc_loss = (lambda_ewc / 2.0) * ewc_loss
        
        return ewc_loss
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics about Fisher values for monitoring."""
        all_fisher = torch.cat([f.flatten() for f in self.fisher_diagonal.values()])
        
        return {
            'mean_fisher': all_fisher.mean().item(),
            'max_fisher': all_fisher.max().item(),
            'min_fisher': all_fisher.min().item(),
            'std_fisher': all_fisher.std().item(),
            'nonzero_ratio': (all_fisher > 1e-10).float().mean().item(),
            'compute_count': self._compute_count,
            'last_computed_mean': self._last_computed_mean,
        }
    
    def get_importance_ranking(self, top_k: int = 10) -> Dict[str, float]:
        """Get the most important parameters by Fisher value."""
        importance = {}
        for name, fisher in self.fisher_diagonal.items():
            importance[name] = fisher.mean().item()
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )
        return sorted_importance