"""
Neural Circuit Policy (NCP) wiring for Soliter.

This module defines the sparse connectivity pattern that mimics
biological neural circuits. The wiring is inspired by C. elegans
connectome topology.
"""

import torch
from typing import Optional
from ncps.wirings import NCP


def create_soliter_wiring(
    sensory_size: int = 41,
    inter_size: int = 256,
    command_size: int = 64,
    motor_size: int = 3,
    sensory_fanout: int = 8,
    inter_fanout: int = 4,
    recurrent_command_synapses: int = 6,
    motor_fanin: int = 8,
    seed: int = 22222,
) -> NCP:
    """
    Create NCP wiring topology for Soliter brain.
    
    This creates a sparse, biologically-inspired connectivity pattern:
    
    Sensory → Interneurons → Command → Motor
       ↓          ↻            ↻
    (feedforward) (recurrent) (recurrent)
    
    Args:
        sensory_size: Number of sensory neurons (not used, NCP infers from input)
        inter_size: Number of interneurons (main memory layer)
        command_size: Number of command neurons (motor planning)
        motor_size: Number of motor neurons
        sensory_fanout: Average outgoing synapses from sensory to inter neurons
        inter_fanout: Average outgoing synapses from inter to command neurons
        recurrent_command_synapses: Average recurrent connections in command layer
        motor_fanin: Average incoming synapses to motor neurons from command
        seed: Random seed for wiring generation
        
    Returns:
        NCP wiring object
    """
    
    wiring = NCP(
        inter_neurons=inter_size,
        command_neurons=command_size,
        motor_neurons=motor_size,
        sensory_fanout=sensory_fanout,
        inter_fanout=inter_fanout,
        recurrent_command_synapses=recurrent_command_synapses,
        motor_fanin=motor_fanin,
        seed=seed,
    )
    
    return wiring


def print_wiring_stats(wiring: NCP) -> None:
    """
    Print statistics about the wiring topology.
    
    Useful for debugging and understanding the connectivity pattern.
    """
    print(f"NCP Wiring Statistics:")
    print(f"  Total units: {wiring.units}")
    print(f"  Sensory neurons: {wiring.input_dim}")
    print(f"  Inter neurons: {wiring.inter_neurons}")
    print(f"  Command neurons: {wiring.command_neurons}")
    print(f"  Motor neurons: {wiring.motor_neurons}")
    print(f"  Output dimension: {wiring.output_dim}")
    
    # Calculate sparsity
    adjacency = wiring.adjacency_matrix
    total_possible = adjacency.shape[0] * adjacency.shape[1]
    total_connections = adjacency.sum()
    sparsity = 1 - (total_connections / total_possible)
    
    print(f"  Total connections: {total_connections}")
    print(f"  Sparsity: {sparsity:.2%}")


def visualize_wiring(
    wiring: NCP,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the wiring topology as a graph.
    
    Requires networkx and matplotlib.
    
    Args:
        wiring: The NCP wiring to visualize
        save_path: Optional path to save the figure
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("networkx and matplotlib required for visualization")
        return
    
    # Create directed graph from adjacency matrix
    G = nx.DiGraph(wiring.adjacency_matrix.numpy())
    
    # Create layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        node_size=20,
        node_color='lightblue',
        edge_color='gray',
        alpha=0.6,
        arrows=True,
        arrowsize=5,
    )
    plt.title("NCP Wiring Topology")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
