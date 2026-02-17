"""
Drive-Modulated NCP Brain for Soliter.

Uses the modified NCP library with internal drive modulation.
Combines:
  - NCP sparse wiring (C. elegans inspired)
  - CfC continuous-time dynamics
  - Drive modulation of interneurons (biological homeostasis)
  
This is the "best of both worlds" architecture!
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# These will use the MODIFIED ncps library
from ncps.torch import CfC
from ncps.wirings import NCP


class DriveModulatedNCPBrain(nn.Module):
    """
    NCP brain with internal drive modulation.
    
    Architecture:
        Sensors (49) → [NCP Wiring] → Motor (3)
        
        NCP Wiring (sparse):
            Sensory → Inter (256) [DRIVE MODULATED!] → Command (64) → Motor (3)
                         ↓
                    4 groups of 64:
                      - Food (0:64): hunger modulation
                      - Water (64:128): thirst modulation
                      - Heat (128:192): cold modulation
                      - Explore (192:256): curiosity modulation
    
    Key features:
        ✅ Sparse connectivity (like C. elegans)
        ✅ Continuous-time dynamics (CfC)
        ✅ Drive modulation (biological homeostasis)
        ✅ End-to-end learnable
    """
    
    def __init__(
        self,
        sensory_size: int = 49,
        inter_size: int = 256,
        command_size: int = 64,
        motor_size: int = 3,
        num_drives: int = 4,
        drive_gain_range: float = 4.0,
        sparsity_level: float = 0.5,
        device: torch.device = None,
    ):
        super().__init__()
        
        assert inter_size % num_drives == 0, \
            f"inter_size ({inter_size}) must be divisible by num_drives ({num_drives})"
        
        self.sensory_size = sensory_size
        self.inter_size = inter_size
        self.command_size = command_size
        self.motor_size = motor_size
        self.num_drives = num_drives
        self.device = device or torch.device('cpu')
        
        # Create NCP wiring (C. elegans inspired sparse connectivity)
        self.wiring = NCP(
            inter_neurons=inter_size,
            command_neurons=command_size,
            motor_neurons=motor_size,
            sensory_fanout=8,              # Each sensor connects to 8 interneurons
            inter_fanout=4,                # Each interneuron connects to 4 command
            recurrent_command_synapses=6,  # Recurrence in command layer
            motor_fanin=8,                 # Each motor gets input from 8 command
            seed=22222,
        )
        
        # Create CfC RNN with DRIVE MODULATION enabled!
        self.cfc = CfC(
            input_size=sensory_size,
            units=self.wiring,
            return_sequences=False,
            batch_first=True,
            mixed_memory=False,
            mode="default",
            enable_drive_modulation=True,  # ← KEY: Enable drive modulation!
            num_drives=num_drives,
            drive_gain_range=drive_gain_range,
        ).to(self.device)
        
        # Hidden state
        self.hidden_state: Optional[torch.Tensor] = None
    
    def forward(
        self,
        sensors: torch.Tensor,
        drives: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with drive modulation.
        
        Args:
            sensors: (batch, 49) or (49,) - sensor inputs
            drives: (batch, 4) or (4,) - [hunger, thirst, cold, curiosity]
            hidden: Optional hidden state
            
        Returns:
            motor_output: (batch, 3) or (3,) - [velocity, turn, sleep]
            hidden_state: Updated hidden state
        """
        # Handle single sample (add batch + sequence dims)
        was_single = sensors.dim() == 1
        if was_single:
            sensors = sensors.unsqueeze(0).unsqueeze(0)  # (1, 1, 49)
            drives = drives.unsqueeze(0).unsqueeze(0)    # (1, 1, 4)
        elif sensors.dim() == 2:
            sensors = sensors.unsqueeze(1)  # Add sequence dim
            drives = drives.unsqueeze(1)
        
        # Use stored hidden state if none provided
        if hidden is None:
            hidden = self.hidden_state
        
        # Forward through NCP with drive modulation
        # The magic happens inside CfC.forward()!
        output, new_hidden = self.cfc(
            input=sensors,
            hx=hidden,
            drives=drives,  # ← Drives modulate interneurons internally!
        )
        
        # Store hidden state
        self.hidden_state = new_hidden
        
        # Remove batch/sequence dims if input was single
        if was_single:
            output = output.squeeze(0)
        
        return output, new_hidden
    
    def reset_hidden(self, batch_size=None, device=None):
        """Reset hidden state."""
        self.hidden_state = None
    
    def get_wiring_stats(self):
        """Get statistics about the NCP wiring."""
        adjacency = self.wiring.adjacency_matrix
        total_possible = adjacency.shape[0] * adjacency.shape[1]
        total_connections = adjacency.sum()
        sparsity = 1 - (total_connections / total_possible)
        
        return {
            'total_units': self.wiring.units,
            'inter_neurons': self.wiring.inter_neurons,
            'command_neurons': self.wiring.command_neurons,
            'motor_neurons': self.wiring.motor_neurons,
            'total_connections': int(total_connections),
            'sparsity': float(sparsity),
        }


def create_drive_modulated_ncp_brain(
    sensory_size: int = 49,
    inter_size: int = 256,
    command_size: int = 64,
    motor_size: int = 3,
    num_drives: int = 4,
    device: torch.device = None,
) -> DriveModulatedNCPBrain:
    """
    Factory function to create drive-modulated NCP brain.
    
    Args:
        sensory_size: Number of sensors (default: 49)
        inter_size: Number of interneurons (default: 256, must be divisible by 4)
        command_size: Number of command neurons (default: 64)
        motor_size: Number of motor outputs (default: 3)
        num_drives: Number of drives (default: 4)
        device: Torch device
        
    Returns:
        Configured DriveModulatedNCPBrain
    """
    return DriveModulatedNCPBrain(
        sensory_size=sensory_size,
        inter_size=inter_size,
        command_size=command_size,
        motor_size=motor_size,
        num_drives=num_drives,
        device=device,
    )
