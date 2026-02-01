"""
Closed-form Continuous-time (CfC) Network for Soliter Agent.

This module implements the brain architecture using CfC units organized
in a Neural Circuit Policy (NCP) topology.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from ncps.torch import CfC
from ncps.wirings import NCP

from .ncp_wiring import create_soliter_wiring


class CfCBrain(nn.Module):
    """
    CfC-based brain for the Soliter agent.
    
    Architecture:
        Sensory (41) → Interneurons (256) → Command (64) → Motor (3)
    
    The interneurons use CfC dynamics for continuous-time processing,
    allowing the network to handle irregular sensor inputs (hallucinations)
    and maintain temporal coherence.
    
    Args:
        sensory_size: Number of sensory inputs (default: 41)
        inter_size: Number of interneurons (default: 256)
        command_size: Number of command neurons (default: 64)
        motor_size: Number of motor outputs (default: 3)
        time_constant_min: Minimum time constant for CfC (default: 0.5)
        time_constant_max: Maximum time constant for CfC (default: 5.0)
        return_sequences: Whether to return full sequence (default: False)
    """
    
    def __init__(
        self,
        sensory_size: int = 41,
        inter_size: int = 256,
        command_size: int = 64,
        motor_size: int = 3,
        time_constant_min: float = 0.5,
        time_constant_max: float = 5.0,
        return_sequences: bool = False,
    ):
        super().__init__()
        
        self.sensory_size = sensory_size
        self.inter_size = inter_size
        self.command_size = command_size
        self.motor_size = motor_size
        
        # Create NCP wiring topology
        self.wiring = create_soliter_wiring(
            sensory_size=sensory_size,
            inter_size=inter_size,
            command_size=command_size,
            motor_size=motor_size,
        )
        
        # CfC RNN layer with NCP wiring
        self.cfc = CfC(
            input_size=sensory_size,
            units=self.wiring.units,
            proj_size=None,  # Use wiring-defined output size
            return_sequences=return_sequences,
            mixed_memory=True,  # Enable both short and long time constants
            mode="default",
            activation="lecun_tanh",
        )
        
        # Note: CfC with NCP wiring automatically handles output selection
        # The wiring.output_dim already equals motor_size (3)
        # No additional projection needed - just verify dimensions match
        assert self.wiring.output_dim == motor_size,             f"Wiring output ({self.wiring.output_dim}) != motor size ({motor_size})"
        
        # Identity mapping (no additional projection needed)
        self.output_projection = nn.Identity()
        
        # Hidden state (persistent across forward passes)
        self.hidden_state: Optional[torch.Tensor] = None
        
        # Activity tracking for homeostatic scaling
        self.register_buffer('mean_activity', torch.zeros(inter_size))
        self.register_buffer('activity_momentum', torch.tensor(0.999))
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the CfC brain.
        
        Args:
            x: Input tensor of shape (batch_size, sensory_size) or 
               (batch_size, seq_len, sensory_size)
            hidden: Optional hidden state from previous step
            return_hidden: Whether to return the hidden state
            
        Returns:
            motor_output: Tensor of shape (batch_size, motor_size)
            hidden_state: Optional tensor of shape (batch_size, units)
        """
        # Handle single-step vs sequence input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Use provided hidden state or stored state
        if hidden is None:
            hidden = self.hidden_state
        
        # Forward through CfC
        cfc_output, new_hidden = self.cfc(x, hidden)
        
        # Store hidden state for next forward pass
        if not return_hidden:
            # Handle tuple hidden state (mixed_memory=True)
            if isinstance(new_hidden, tuple):
                self.hidden_state = tuple(h.detach() for h in new_hidden)
            else:
                self.hidden_state = new_hidden.detach()
        
        # Extract output from wiring-defined output neurons
        if squeeze_output:
            cfc_output = cfc_output.squeeze(1)
        
        # Project to motor commands
        motor_output = self.output_projection(cfc_output)
        
        # Apply motor constraints
        # velocity: [0, 1], turn: [-1, 1], sleep: [0, 1]
        motor_output = torch.cat([
            torch.sigmoid(motor_output[..., 0:1]),  # velocity
            torch.tanh(motor_output[..., 1:2]),     # turn
            torch.sigmoid(motor_output[..., 2:3]),  # sleep
        ], dim=-1)
        
        # Update activity tracking (for homeostatic scaling)
        if self.training:
            with torch.no_grad():
                # Extract interneuron activations
                inter_activity = self._get_interneuron_activity(new_hidden)
                self.mean_activity = (
                    self.activity_momentum * self.mean_activity +
                    (1 - self.activity_momentum) * inter_activity.mean(0)
                )
        
        if return_hidden:
            return motor_output, new_hidden
        return motor_output, None
    
    def _get_interneuron_activity(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Extract interneuron layer activations from hidden state.
        
        The hidden state contains all neuron activations. We need to
        extract just the interneurons for homeostatic scaling.
        
        Args:
            hidden: Full hidden state tensor
            
        Returns:
            Interneuron activations only
        """
        # In NCP wiring, interneurons come after sensory but before command
        start_idx = self.sensory_size
        end_idx = start_idx + self.inter_size
        return hidden[..., start_idx:end_idx]
    
    def reset_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """Reset the hidden state (e.g., at start of episode)."""
        if device is None:
            device = next(self.parameters()).device
        # CfC with mixed_memory=True requires tuple (h_state, c_state)
        h_state = torch.zeros(batch_size, self.wiring.units, device=device)
        c_state = torch.zeros(batch_size, self.wiring.units, device=device)
        self.hidden_state = (h_state, c_state)
    def get_mean_activity(self) -> torch.Tensor:
        """Get the current mean activity of interneurons."""
        return self.mean_activity.clone()
    
    def apply_homeostatic_scaling(
        self,
        target_activity: float = 0.5,
        scaling_rate: float = 0.01,
    ) -> float:
        """
        Apply homeostatic synaptic scaling to prevent saturation.
        
        This is called during the sleep phase (NREM). It multiplicatively
        scales weights to bring neuron activity back toward a target setpoint.
        
        Args:
            target_activity: Target mean firing rate
            scaling_rate: How aggressively to scale (γ in the paper)
            
        Returns:
            The scaling factor applied
        """
        current_activity = self.mean_activity.mean().item()
        
        # Calculate scaling factor
        # scale = 1 + γ * (target - current)
        error = target_activity - current_activity
        scale_factor = 1.0 + scaling_rate * error
        
        # Apply multiplicative scaling to interneuron weights
        with torch.no_grad():
            for name, param in self.cfc.named_parameters():
                if 'weight' in name and param.requires_grad:
                    # Only scale weights, not biases
                    param.data *= scale_factor
        
        return scale_factor
    
    def get_weight_norm(self) -> float:
        """Get the total weight norm (for monitoring saturation)."""
        total_norm = 0.0
        for param in self.cfc.parameters():
            if param.requires_grad:
                total_norm += param.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def predict_future_self(
        self,
        current_state: torch.Tensor,
        horizon: int = 10,
    ) -> torch.Tensor:
        """
        Predict future states (for self-model coherence measurement).
        
        This is used in consciousness experiments to test if the agent
        can predict its own future behavior better than an external observer.
        
        Args:
            current_state: Current sensory input
            horizon: Number of steps to predict
            
        Returns:
            Predicted future states
        """
        predictions = []
        hidden = self.hidden_state.clone() if self.hidden_state is not None else None
        state = current_state
        
        with torch.no_grad():
            for _ in range(horizon):
                output, hidden = self.forward(state, hidden=hidden, return_hidden=True)
                predictions.append(output)
                # For next prediction, use output as pseudo-state
                # (In real implementation, you'd have environment model)
                state = output
        
        return torch.stack(predictions, dim=1)


class SimpleMLP(nn.Module):
    """
    Simple MLP baseline for comparison experiments.
    
    This is used as a control to compare CfC performance.
    It has similar parameter count but no temporal dynamics.
    """
    
    def __init__(
        self,
        input_size: int = 41,
        hidden_size: int = 256,
        output_size: int = 3,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Forward pass (no hidden state for MLP)."""
        output = self.net(x)
        
        # Apply same motor constraints as CfC
        output = torch.cat([
            torch.sigmoid(output[..., 0:1]),  # velocity
            torch.tanh(output[..., 1:2]),     # turn
            torch.sigmoid(output[..., 2:3]),  # sleep
        ], dim=-1)
        
        return output, None
    
    def reset_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """Reset the hidden state (e.g., at start of episode)."""
        if device is None:
            device = next(self.parameters()).device
        # CfC with mixed_memory=True requires tuple (h_state, c_state)
        h_state = torch.zeros(batch_size, self.wiring.units, device=device)
        c_state = torch.zeros(batch_size, self.wiring.units, device=device)
        self.hidden_state = (h_state, c_state)
