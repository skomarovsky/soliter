"""
Replay Buffer with Epistemic Pruning.

The buffer stores recent experiences and prunes them based on:
1. Epistemic uncertainty (model confidence)
2. TD-error (prediction accuracy)

Experiences are removed when the model has "consolidated" them into weights.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from dataclasses import dataclass
import random
import copy

if TYPE_CHECKING:
    from ..memory.fisher_matrix import FisherInformationMatrix


@dataclass
class Transition:
    """Single experience transition."""
    state: torch.Tensor  # Sensor readings
    action: torch.Tensor  # Motor outputs
    reward: float  # Survival reward
    next_state: torch.Tensor
    done: bool
    
    # Metadata
    tick: int = 0
    uncertainty: float = 1.0  # Epistemic uncertainty (1.0 = maximum uncertainty)
    td_error: float = 1.0  # TD-error


class ReplayBuffer:
    """
    Experience replay buffer with epistemic pruning.
    
    Unlike standard FIFO buffers, this prunes based on consolidation:
    - Keep experiences with high uncertainty (not yet learned)
    - Remove experiences with low uncertainty + low TD-error (consolidated)
    
    Uncertainty Estimation Strategy:
    --------------------------------
    Since CfC networks don't have dropout, we use Fisher-informed weight
    perturbation. The Fisher Information Matrix tells us which weights are
    important (high Fisher) vs unimportant (low Fisher).
    
    A consolidated memory should be ROBUST to perturbations of low-Fisher
    weights (the unimportant ones). If perturbing unimportant weights 
    changes the output significantly, the memory is NOT well consolidated.
    
    This is more principled than random noise because:
    1. It directly uses the same importance measure as EWC
    2. It tests if the memory relies on stable (high-Fisher) pathways
    3. It aligns with the biological intuition of synaptic consolidation
    """
    
    def __init__(
        self,
        capacity: int = 1_000_000,
        prune_threshold_uncertainty: float = 0.1,
        prune_threshold_td_error: float = 0.05,
    ):
        self.capacity = capacity
        self.prune_threshold_uncertainty = prune_threshold_uncertainty
        self.prune_threshold_td_error = prune_threshold_td_error
        
        self.buffer: List[Transition] = []
        self.position = 0
        
        # Statistics
        self.total_added = 0
        self.total_pruned = 0
    
    def push(self, transition: Transition) -> None:
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Circular buffer: overwrite oldest
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
        self.total_added += 1
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch from buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def update_uncertainties(
        self,
        model: nn.Module,
        num_samples: int = 10,
        fisher_matrix: Optional['FisherInformationMatrix'] = None,
        perturbation_scale: float = 0.1,
    ) -> None:
        """
        Update epistemic uncertainty using Fisher-informed weight perturbation.
        
        Strategy: Perturb LOW-Fisher weights (unimportant ones) and measure
        output variance. Well-consolidated memories are robust to these 
        perturbations; uncertain memories are sensitive.
        
        Args:
            model: The neural network (CfC)
            num_samples: Number of forward passes with perturbed weights
            fisher_matrix: Fisher Information Matrix (if None, falls back to random)
            perturbation_scale: Scale of perturbation for low-Fisher weights
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Get original weights
        original_state = copy.deepcopy(model.state_dict())
        
        # Compute Fisher-based masks if available
        if fisher_matrix is not None and fisher_matrix.fisher_diagonal:
            perturbation_masks = self._compute_fisher_masks(
                model, fisher_matrix, perturbation_scale
            )
        else:
            perturbation_masks = None
        
        for transition in self.buffer:
            state = transition.state.unsqueeze(0).to(device)
            
            predictions = []
            with torch.no_grad():
                for sample_idx in range(num_samples):
                    # Apply Fisher-informed perturbation to weights
                    if perturbation_masks is not None:
                        self._apply_perturbation(model, perturbation_masks)
                    else:
                        # Fallback: perturb all weights slightly
                        self._apply_random_perturbation(model, perturbation_scale)
                    
                    # Reset hidden state for clean forward pass
                    if hasattr(model, 'reset_hidden'):
                        hidden = model.reset_hidden(batch_size=1, device=device)
                    else:
                        hidden = None
                    
                    output, _ = model(state, hidden=hidden, return_hidden=True)
                    predictions.append(output.clone())
                    
                    # Restore original weights
                    model.load_state_dict(original_state)
            
            # Calculate variance across predictions
            predictions = torch.stack(predictions)
            variance = predictions.var(dim=0).mean().item()
            
            # Normalize to [0, 1] range
            # Higher variance = higher uncertainty
            transition.uncertainty = min(1.0, variance / (variance + 0.01))
        
        # Ensure original weights are restored
        model.load_state_dict(original_state)
    
    def _compute_fisher_masks(
        self,
        model: nn.Module,
        fisher_matrix: 'FisherInformationMatrix',
        perturbation_scale: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute perturbation masks based on Fisher Information.
        
        Low-Fisher weights get larger perturbations (they're "weak"/unimportant).
        High-Fisher weights get smaller perturbations (they're "strong"/important).
        
        Returns:
            Dictionary mapping parameter names to perturbation standard deviations
        """
        masks = {}
        
        for name, param in model.named_parameters():
            if name in fisher_matrix.fisher_diagonal:
                fisher = fisher_matrix.fisher_diagonal[name]
                
                # Inverse Fisher scaling: low Fisher → high perturbation
                # Add small epsilon to avoid division by zero
                # Normalize Fisher values to [0, 1] range first
                fisher_normalized = fisher / (fisher.max() + 1e-8)
                
                # Perturbation scale: high for low-Fisher, low for high-Fisher
                # Scale = perturbation_scale * (1 - normalized_fisher)
                perturbation_std = perturbation_scale * (1.0 - fisher_normalized)
                
                masks[name] = perturbation_std
            else:
                # No Fisher info: use uniform perturbation
                masks[name] = torch.full_like(param, perturbation_scale)
        
        return masks
    
    def _apply_perturbation(
        self,
        model: nn.Module,
        perturbation_masks: Dict[str, torch.Tensor],
    ) -> None:
        """Apply Fisher-informed perturbations to model weights."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in perturbation_masks:
                    noise = torch.randn_like(param) * perturbation_masks[name]
                    param.add_(noise)
    
    def _apply_random_perturbation(
        self,
        model: nn.Module,
        scale: float,
    ) -> None:
        """Apply uniform random perturbations (fallback when no Fisher available)."""
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * scale
                param.add_(noise)
    
    def update_uncertainties_batch(
        self,
        model: nn.Module,
        batch_size: int = 64,
        num_samples: int = 10,
        fisher_matrix: Optional['FisherInformationMatrix'] = None,
        perturbation_scale: float = 0.1,
    ) -> None:
        """
        Batch version of update_uncertainties for efficiency.
        
        Args:
            model: The neural network (CfC)
            batch_size: Number of transitions to process at once
            num_samples: Number of forward passes with perturbed weights
            fisher_matrix: Fisher Information Matrix
            perturbation_scale: Scale of perturbation for low-Fisher weights
        """
        model.eval()
        device = next(model.parameters()).device
        
        original_state = copy.deepcopy(model.state_dict())
        
        if fisher_matrix is not None and fisher_matrix.fisher_diagonal:
            perturbation_masks = self._compute_fisher_masks(
                model, fisher_matrix, perturbation_scale
            )
        else:
            perturbation_masks = None
        
        for batch_start in range(0, len(self.buffer), batch_size):
            batch_end = min(batch_start + batch_size, len(self.buffer))
            batch_transitions = self.buffer[batch_start:batch_end]
            current_batch_size = len(batch_transitions)
            
            states = torch.stack([t.state for t in batch_transitions]).to(device)
            
            all_predictions = []
            with torch.no_grad():
                for _ in range(num_samples):
                    if perturbation_masks is not None:
                        self._apply_perturbation(model, perturbation_masks)
                    else:
                        self._apply_random_perturbation(model, perturbation_scale)
                    
                    if hasattr(model, 'reset_hidden'):
                        hidden = model.reset_hidden(batch_size=current_batch_size, device=device)
                    else:
                        hidden = None
                    
                    output, _ = model(states, hidden=hidden, return_hidden=True)
                    all_predictions.append(output.clone())
                    
                    model.load_state_dict(original_state)
            
            predictions = torch.stack(all_predictions)
            variances = predictions.var(dim=0)
            mean_variances = variances.mean(dim=-1)
            
            for i, transition in enumerate(batch_transitions):
                var = mean_variances[i].item()
                transition.uncertainty = min(1.0, var / (var + 0.01))
        
        model.load_state_dict(original_state)
    
    def update_td_errors(
        self,
        model: nn.Module,
        gamma: float = 0.99,
    ) -> None:
        """
        Update TD-errors for all transitions.
        
        TD-error = |r + γ·V(s') - V(s)|
        
        For simplicity, we use action magnitude as proxy for value.
        """
        model.eval()
        device = next(model.parameters()).device
        
        for transition in self.buffer:
            if transition.done:
                td_error = abs(transition.reward)
            else:
                with torch.no_grad():
                    if hasattr(model, 'reset_hidden'):
                        hidden = model.reset_hidden(batch_size=1, device=device)
                    else:
                        hidden = None
                    
                    state = transition.state.unsqueeze(0).to(device)
                    next_state = transition.next_state.unsqueeze(0).to(device)
                    
                    state_output, new_hidden = model(state, hidden=hidden, return_hidden=True)
                    next_state_output, _ = model(next_state, hidden=new_hidden, return_hidden=True)
                    
                    state_value = -state_output[0, 0].item()
                    next_state_value = -next_state_output[0, 0].item()
                    
                    td_error = abs(transition.reward + gamma * next_state_value - state_value)
            
            transition.td_error = td_error
    
    def update_td_errors_batch(
        self,
        model: nn.Module,
        batch_size: int = 64,
        gamma: float = 0.99,
    ) -> None:
        """Batch version of update_td_errors for efficiency."""
        model.eval()
        device = next(model.parameters()).device
        
        for batch_start in range(0, len(self.buffer), batch_size):
            batch_end = min(batch_start + batch_size, len(self.buffer))
            batch_transitions = self.buffer[batch_start:batch_end]
            current_batch_size = len(batch_transitions)
            
            states = torch.stack([t.state for t in batch_transitions]).to(device)
            next_states = torch.stack([t.next_state for t in batch_transitions]).to(device)
            rewards = torch.tensor([t.reward for t in batch_transitions], device=device)
            dones = torch.tensor([t.done for t in batch_transitions], device=device)
            
            with torch.no_grad():
                if hasattr(model, 'reset_hidden'):
                    hidden = model.reset_hidden(batch_size=current_batch_size, device=device)
                else:
                    hidden = None
                
                state_outputs, new_hidden = model(states, hidden=hidden, return_hidden=True)
                next_state_outputs, _ = model(next_states, hidden=new_hidden, return_hidden=True)
                
                state_values = -state_outputs[:, 0]
                next_state_values = -next_state_outputs[:, 0]
                
                td_errors = torch.abs(rewards + gamma * next_state_values * (~dones) - state_values)
            
            for i, transition in enumerate(batch_transitions):
                transition.td_error = td_errors[i].item()
    
    def prune_consolidated(self) -> int:
        """
        Remove consolidated experiences (low uncertainty AND low TD-error).
        
        Returns:
            Number of transitions pruned
        """
        original_size = len(self.buffer)
        
        self.buffer = [
            t for t in self.buffer
            if not (
                t.uncertainty < self.prune_threshold_uncertainty and
                t.td_error < self.prune_threshold_td_error
            )
        ]
        
        pruned = original_size - len(self.buffer)
        self.total_pruned += pruned
        
        self.position = len(self.buffer) % self.capacity
        
        return pruned
    
    def get_consolidation_candidates(self) -> List[Transition]:
        """Get transitions that are close to being consolidated (for debugging)."""
        return [
            t for t in self.buffer
            if t.uncertainty < self.prune_threshold_uncertainty * 2 or
               t.td_error < self.prune_threshold_td_error * 2
        ]
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'total_added': self.total_added,
                'total_pruned': self.total_pruned,
            }
        
        uncertainties = [t.uncertainty for t in self.buffer]
        td_errors = [t.td_error for t in self.buffer]
        
        near_consolidated = sum(
            1 for t in self.buffer
            if t.uncertainty < self.prune_threshold_uncertainty * 2 and
               t.td_error < self.prune_threshold_td_error * 2
        )
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'total_added': self.total_added,
            'total_pruned': self.total_pruned,
            'avg_uncertainty': float(np.mean(uncertainties)),
            'avg_td_error': float(np.mean(td_errors)),
            'min_uncertainty': float(np.min(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties)),
            'min_td_error': float(np.min(td_errors)),
            'max_td_error': float(np.max(td_errors)),
            'near_consolidated': near_consolidated,
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.position = 0
