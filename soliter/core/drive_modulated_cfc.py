"""
Drive-Modulated CfC Brain for Soliter Agent.

This implements Option A: Drives modulate interneuron activity INSIDE the brain,
allowing natural winner-take-all behavior to emerge from the network dynamics.

Architecture:
    Sensors (47) → Interneurons (256) [DRIVE MODULATED] → Command (64) → Motor (3)
    
Key differences from standard CfC:
    1. Sensors DO NOT include drive states (only 47 channels)
    2. Interneurons are organized into drive-specific groups
    3. Each group gets modulated by corresponding drive
    4. Command layer receives modulated interneurons → natural competition
    5. No external winner-take-all, no external action constraints

Biological inspiration:
    - C. elegans interneurons integrate hunger + food smell
    - Drives modulate neural gain, not sensor amplitudes
    - Winner-take-all emerges from lateral inhibition in command layer
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class DriveModulatedCfCBrain(nn.Module):
    """
    CfC brain with drive-modulated interneurons.
    
    Architecture:
        Sensors (49) → Interneurons (256) → [DRIVE MODULATION] → Command (64) → Motor (3)
        
    Drive groups (4 groups of 64 interneurons each):
        - Food neurons (0:64): Modulated by hunger
        - Water neurons (64:128): Modulated by thirst  
        - Heat neurons (128:192): Modulated by cold
        - Exploration neurons (192:256): Modulated by curiosity
        
    Args:
        sensory_size: Number of sensors (default: 49, NO drives!)
        inter_size: Number of interneurons (default: 256, must be divisible by 4)
        command_size: Number of command neurons (default: 64)
        motor_size: Number of motor outputs (default: 3)
        num_drives: Number of drive states (default: 4)
        drive_gain_min: Minimum drive gain (default: 1.0, no boost)
        drive_gain_max: Maximum drive gain (default: 5.0, 5x boost when drive=1.0)
    """
    
    def __init__(
        self,
        sensory_size: int = 49,  # Updated from 47 to include light + ambient temp!
        inter_size: int = 256,
        command_size: int = 64,
        motor_size: int = 3,
        num_drives: int = 4,
        drive_gain_min: float = 1.0,
        drive_gain_max: float = 5.0,
        time_constant_min: float = 0.5,
        time_constant_max: float = 5.0,
    ):
        super().__init__()
        
        assert inter_size % num_drives == 0, \
            f"inter_size ({inter_size}) must be divisible by num_drives ({num_drives})"
        
        self.sensory_size = sensory_size
        self.inter_size = inter_size
        self.command_size = command_size
        self.motor_size = motor_size
        self.num_drives = num_drives
        self.neurons_per_drive = inter_size // num_drives
        
        self.drive_gain_min = drive_gain_min
        self.drive_gain_max = drive_gain_max
        
        # Create interneuron layer (standard CfC units, not NCP wiring)
        # We'll build drive modulation ourselves
        self.interneuron_layer = nn.Linear(sensory_size, inter_size)
        
        # LEARNABLE DRIVE MODULATION WEIGHTS
        # Shape: (num_drives, neurons_per_drive)
        # Each drive has different learned gains for its neuron group
        self.drive_modulation_weights = nn.Parameter(
            torch.ones(num_drives, self.neurons_per_drive)
        )
        
        # Command layer with lateral inhibition (winner-take-all)
        # Input: modulated interneurons (256)
        # Output: command neurons (64)
        self.command_layer = nn.Sequential(
            nn.Linear(inter_size, command_size),
            nn.LayerNorm(command_size),  # Normalize for competition
            nn.Tanh(),  # Bounded activation
        )
        
        # Motor layer
        # Input: command neurons (64)  
        # Output: motor neurons (3)
        self.motor_layer = nn.Linear(command_size, motor_size)
        
        # Hidden state (for recurrence, simple approach)
        self.hidden_state = None
        self.register_buffer('prev_inter', torch.zeros(inter_size))
        
    def forward(
        self,
        sensors: torch.Tensor,
        drives: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with drive modulation.
        
        Args:
            sensors: Input tensor (batch_size, 49) - NO DRIVES!
            drives: Drive tensor (batch_size, 4) - [hunger, thirst, cold, curiosity]
            hidden: Optional hidden state from previous step
            
        Returns:
            motor_output: Tensor (batch_size, 3) - [velocity, turn, sleep]
            hidden_state: Updated hidden state (None for now, simple feedforward)
        """
        # Ensure inputs are 2D
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        if drives.dim() == 1:
            drives = drives.unsqueeze(0)
            
        batch_size = sensors.shape[0]
        
        # Process sensors through interneuron layer
        inter_raw = self.interneuron_layer(sensors)  # (batch_size, 256)
        inter_raw = torch.tanh(inter_raw)  # Non-linearity
        
        # Add simple recurrence (blend with previous state for temporal continuity)
        if batch_size == 1 and self.prev_inter is not None:
            inter_raw = 0.8 * inter_raw + 0.2 * self.prev_inter
            self.prev_inter = inter_raw.detach().clone().squeeze(0)
        
        # Split interneurons into drive-specific groups
        # Shape: (batch_size, num_drives, neurons_per_drive)
        inter_groups = inter_raw.reshape(batch_size, self.num_drives, self.neurons_per_drive)
        
        # Compute drive modulation
        # drives: (batch_size, 4) → (batch_size, 4, 1)
        # modulation_weights: (4, 64)
        # Result: (batch_size, 4, 64)
        drives_expanded = drives.unsqueeze(2)  # (batch_size, 4, 1)
        
        # Modulation = drive_gain_min + drive_value * drive_gain_range * learned_weights
        drive_gain_range = self.drive_gain_max - self.drive_gain_min
        modulation = self.drive_gain_min + drives_expanded * drive_gain_range * self.drive_modulation_weights
        
        # Apply modulation to interneuron groups
        # modulated_groups: (batch_size, 4, 64)
        modulated_groups = inter_groups * modulation
        
        # Flatten back to full interneuron vector
        # Shape: (batch_size, 256)
        modulated_inter = modulated_groups.reshape(batch_size, self.inter_size)
        
        # Command layer - natural competition via LayerNorm + Tanh
        # Strongest drive pathway will dominate here
        command_output = self.command_layer(modulated_inter)  # (batch_size, 64)
        
        # Motor output
        motor_output = self.motor_layer(command_output)  # (batch_size, 3)
        
        return motor_output, None  # No complex hidden state for now
    
    def reset_hidden(self, batch_size=None, device=None):
        """
        Reset hidden state (for new episode).
        
        Args:
            batch_size: Ignored (for compatibility with old interface)
            device: Ignored (for compatibility with old interface)
        """
        self.hidden_state = None
        
    def get_interneuron_activity(
        self,
        sensors: torch.Tensor,
        drives: torch.Tensor,
    ) -> dict:
        """
        Get interneuron activity for debugging/visualization.
        
        Returns dict with:
            - raw_inter: Raw interneuron outputs (batch, 256)
            - modulated_inter: After drive modulation (batch, 256)
            - modulation: Drive modulation factors (batch, 4, 64)
            - command: Command neuron outputs (batch, 64)
        """
        with torch.no_grad():
            if sensors.dim() == 1:
                sensors = sensors.unsqueeze(0)
            if drives.dim() == 1:
                drives = drives.unsqueeze(0)
                
            batch_size = sensors.shape[0]
            
            # Get raw interneurons
            cfc_output, _ = self.cfc(sensors, hx=None)
            inter_neurons = cfc_output[:, :self.inter_size]
            
            # Compute modulation
            inter_groups = inter_neurons.reshape(batch_size, self.num_drives, self.neurons_per_drive)
            drives_expanded = drives.unsqueeze(2)
            drive_gain_range = self.drive_gain_max - self.drive_gain_min
            modulation = self.drive_gain_min + drives_expanded * drive_gain_range * self.drive_modulation_weights
            
            # Apply modulation
            modulated_groups = inter_groups * modulation
            modulated_inter = modulated_groups.reshape(batch_size, self.inter_size)
            
            # Command output
            command_output = self.command_layer(modulated_inter)
            
            return {
                'raw_inter': inter_neurons,
                'modulated_inter': modulated_inter,
                'modulation': modulation,
                'command': command_output,
            }


def create_drive_modulated_brain(
    sensory_size: int = 49,  # 49 channels with light + ambient temp
    inter_size: int = 256,
    command_size: int = 64,
    motor_size: int = 3,
    **kwargs
) -> DriveModulatedCfCBrain:
    """
    Factory function to create drive-modulated brain.
    
    Args:
        sensory_size: Number of sensor inputs (default: 49, NO drives!)
        inter_size: Number of interneurons (default: 256)
        command_size: Number of command neurons (default: 64)
        motor_size: Number of motor outputs (default: 3)
        **kwargs: Additional arguments for DriveModulatedCfCBrain
        
    Returns:
        Configured DriveModulatedCfCBrain instance
    """
    return DriveModulatedCfCBrain(
        sensory_size=sensory_size,
        inter_size=inter_size,
        command_size=command_size,
        motor_size=motor_size,
        **kwargs
    )
