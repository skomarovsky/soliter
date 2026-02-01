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
        
        # Build the wiring with input size to create connection matrices
        self.wiring.build(sensory_size)
        
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
        assert self.wiring.output_dim == motor_size, \
            f"Wiring output ({self.wiring.output_dim}) != motor size ({motor_size})"
        
        # Identity mapping (no additional projection needed)
        self.output_projection = nn.Identity()
        
        # Hidden state (persistent across forward passes)
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        # Activity tracking for homeostatic scaling
        self.register_buffer('mean_activity', torch.zeros(inter_size))
        self.register_buffer('activity_momentum', torch.tensor(0.999))
    
    def _get_hidden_batch_size(self, hidden) -> int:
        """Extract batch size from hidden state."""
        if hidden is None:
            return 0
        if isinstance(hidden, tuple):
            return hidden[0].shape[0]
        return hidden.shape[0]
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
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
        
        batch_size = x.shape[0]
        
        # Use provided hidden state or stored state
        if hidden is None:
            hidden = self.hidden_state
        
        # Validate hidden state batch size matches input
        # This is critical for sleep-wake transitions where batch sizes change
        if hidden is not None:
            hidden_batch_size = self._get_hidden_batch_size(hidden)
            if hidden_batch_size != batch_size:
                # Batch size mismatch - must reinitialize hidden state
                hidden = None
        
        # Initialize hidden state if needed
        if hidden is None:
            hidden = self.reset_hidden(batch_size, device=x.device)
        
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
    
    def _get_interneuron_activity(self, hidden) -> torch.Tensor:
        """Extract interneuron activations from hidden state."""
        # Handle tuple hidden state (h, c) from mixed_memory
        if isinstance(hidden, tuple):
            h_state = hidden[0]
        else:
            h_state = hidden
            
        # Interneurons are indices 0 to inter_size-1
        start_idx = 0
        end_idx = len(self.wiring._inter_neurons)
        return h_state[..., start_idx:end_idx]
    
    def reset_hidden(
        self, 
        batch_size: int = 1, 
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset the hidden state (e.g., at start of episode or batch size change).
        
        Args:
            batch_size: Number of parallel sequences
            device: Device to create tensors on
            
        Returns:
            Tuple of (h_state, c_state) for mixed_memory CfC
        """
        if device is None:
            device = next(self.parameters()).device
        
        # CfC with mixed_memory=True requires tuple (h_state, c_state)
        h_state = torch.zeros(batch_size, self.wiring.units, device=device)
        c_state = torch.zeros(batch_size, self.wiring.units, device=device)
        self.hidden_state = (h_state, c_state)
        return self.hidden_state
    
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
        
        # Clone hidden state to avoid modifying the original
        if self.hidden_state is not None:
            if isinstance(self.hidden_state, tuple):
                hidden = tuple(h.clone() for h in self.hidden_state)
            else:
                hidden = self.hidden_state.clone()
        else:
            hidden = None
            
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
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )
        
        # For API compatibility with CfCBrain
        self.hidden_state = None
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass (no hidden state for MLP).
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            hidden: Ignored (for API compatibility)
            return_hidden: Ignored (for API compatibility)
            
        Returns:
            output: Motor commands of shape (batch_size, output_size)
            None: No hidden state for MLP
        """
        # Handle sequence input by taking last timestep
        if x.dim() == 3:
            x = x[:, -1, :]  # Take last timestep
        
        output = self.net(x)
        
        # Apply same motor constraints as CfC
        output = torch.cat([
            torch.sigmoid(output[..., 0:1]),  # velocity
            torch.tanh(output[..., 1:2]),     # turn
            torch.sigmoid(output[..., 2:3]),  # sleep
        ], dim=-1)
        
        return output, None
    
    def reset_hidden(
        self, 
        batch_size: int = 1, 
        device: Optional[torch.device] = None
    ) -> None:
        """
        Reset hidden state (no-op for MLP, maintains API compatibility).
        
        Args:
            batch_size: Ignored
            device: Ignored
            
        Returns:
            None (MLP has no hidden state)
        """
        self.hidden_state = None
        return None
    
    def get_weight_norm(self) -> float:
        """Get the total weight norm (for monitoring)."""
        total_norm = 0.0
        for param in self.parameters():
            if param.requires_grad:
                total_norm += param.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def apply_homeostatic_scaling(
        self,
        target_activity: float = 0.5,
        scaling_rate: float = 0.01,
    ) -> float:
        """
        Apply homeostatic scaling (simplified for MLP).
        
        MLP doesn't track activity, so this applies uniform scaling.
        """
        scale_factor = 1.0  # No activity tracking, so no scaling
        return scale_factor
