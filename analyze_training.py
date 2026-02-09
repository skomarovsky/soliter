#!/usr/bin/env python3
"""
Analyze training results from a JSON log file.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_training_log(filepath: str) -> Dict[str, Any]:
    """Load training log from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_snapshots(snapshots: List[Dict]) -> pd.DataFrame:
    """Convert snapshots to DataFrame and perform basic analysis."""
    df = pd.DataFrame(snapshots)
    
    # Convert to datetime if possible
    if 'tick' in df.columns:
        df = df.sort_values('tick').reset_index(drop=True)
    
    return df


def plot_vitals_over_time(df: pd.DataFrame, save_path: str = None):
    """Plot vital signs over time."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Vital Signs Over Time', fontsize=16)
    
    # Energy
    axes[0, 0].plot(df['tick'], df['energy'], label='Energy', color='red', alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Death Threshold')
    axes[0, 0].set_title('Energy')
    axes[0, 0].set_xlabel('Tick')
    axes[0, 0].set_ylabel('Energy Level')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Hydration
    axes[0, 1].plot(df['tick'], df['hydration'], label='Hydration', color='blue', alpha=0.7)
    axes[0, 1].axhline(y=0, color='blue', linestyle='--', alpha=0.5, label='Death Threshold')
    axes[0, 1].set_title('Hydration')
    axes[0, 1].set_xlabel('Tick')
    axes[0, 1].set_ylabel('Hydration Level')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Temperature
    axes[1, 0].plot(df['tick'], df['temperature'], label='Temperature', color='orange', alpha=0.7)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Hypothermia')
    axes[1, 0].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Hyperthermia')
    axes[1, 0].set_title('Body Temperature')
    axes[1, 0].set_xlabel('Tick')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Wakefulness
    axes[1, 1].plot(df['tick'], df['wakefulness'], label='Wakefulness', color='purple', alpha=0.7)
    axes[1, 1].set_title('Wakefulness')
    axes[1, 1].set_xlabel('Tick')
    axes[1, 1].set_ylabel('Wakefulness Level')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_drive_system(df: pd.DataFrame, save_path: str = None):
    """Plot drive system components over time."""
    if not all(col in df.columns for col in ['hunger', 'thirst', 'cold', 'curiosity']):
        print("Drive system columns not found in data.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Drive System Components Over Time', fontsize=16)
    
    # Hunger
    axes[0, 0].plot(df['tick'], df['hunger'], label='Hunger Drive', color='red', alpha=0.7)
    axes[0, 0].set_title('Hunger Drive')
    axes[0, 0].set_xlabel('Tick')
    axes[0, 0].set_ylabel('Drive Intensity')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Thirst
    axes[0, 1].plot(df['tick'], df['thirst'], label='Thirst Drive', color='blue', alpha=0.7)
    axes[0, 1].set_title('Thirst Drive')
    axes[0, 1].set_xlabel('Tick')
    axes[0, 1].set_ylabel('Drive Intensity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Cold
    axes[1, 0].plot(df['tick'], df['cold'], label='Cold Drive', color='cyan', alpha=0.7)
    axes[1, 0].set_title('Cold Drive')
    axes[1, 0].set_xlabel('Tick')
    axes[1, 0].set_ylabel('Drive Intensity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Curiosity
    axes[1, 1].plot(df['tick'], df['curiosity'], label='Curiosity Drive', color='green', alpha=0.7)
    axes[1, 1].set_title('Curiosity Drive')
    axes[1, 1].set_xlabel('Tick')
    axes[1, 1].set_ylabel('Drive Intensity')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_rewards_and_satisfaction(df: pd.DataFrame, save_path: str = None):
    """Plot reward components over time."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reward Components Over Time', fontsize=16)
    
    # Total Reward
    axes[0, 0].plot(df['tick'], df['reward'], label='Total Reward', color='black', alpha=0.7)
    axes[0, 0].set_title('Total Reward')
    axes[0, 0].set_xlabel('Tick')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Satisfaction
    if 'satisfaction' in df.columns:
        axes[0, 1].plot(df['tick'], df['satisfaction'], label='Satisfaction', color='green', alpha=0.7)
        axes[0, 1].set_title('Satisfaction')
        axes[0, 1].set_xlabel('Tick')
        axes[0, 1].set_ylabel('Satisfaction')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Discomfort
    if 'discomfort' in df.columns:
        axes[1, 0].plot(df['tick'], df['discomfort'], label='Discomfort', color='red', alpha=0.7)
        axes[1, 0].set_title('Discomfort')
        axes[1, 0].set_xlabel('Tick')
        axes[1, 0].set_ylabel('Discomfort')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Surprise
    if 'surprise' in df.columns:
        axes[1, 1].plot(df['tick'], df['surprise'], label='Surprise', color='purple', alpha=0.7)
        axes[1, 1].set_title('Surprise')
        axes[1, 1].set_xlabel('Tick')
        axes[1, 1].set_ylabel('Surprise')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_position_trajectory(df: pd.DataFrame, save_path: str = None):
    """Plot agent position trajectory."""
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(df['x'], df['y'], c=df['tick'], cmap='viridis', s=1, alpha=0.6)
    ax.set_title('Agent Position Trajectory')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.colorbar(scatter, ax=ax, label='Tick')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_deaths(deaths: List[Dict]):
    """Analyze death events."""
    if not deaths:
        print("No deaths recorded in this run.")
        return
    
    print(f"\n=== DEATH ANALYSIS ===")
    print(f"Total deaths: {len(deaths)}")
    
    # Death causes
    causes = [d['cause'] for d in deaths]
    cause_counts = {}
    for cause in causes:
        cause_counts[cause] = cause_counts.get(cause, 0) + 1
    
    print("\nDeath causes:")
    for cause, count in cause_counts.items():
        print(f"  {cause}: {count} ({100*count/len(deaths):.1f}%)")
    
    # Average life duration
    life_durations = [d['life_duration'] for d in deaths]
    avg_life = sum(life_durations) / len(life_durations) if life_durations else 0
    print(f"\nAverage life duration: {avg_life:.1f} ticks")
    
    # Positions of deaths
    death_positions = [(d['x'], d['y']) for d in deaths]
    print(f"Death positions: {death_positions[:5]}..." if len(death_positions) > 5 else f"Death positions: {death_positions}")


def analyze_sleep_events(sleeps: List[Dict]):
    """Analyze sleep events."""
    if not sleeps:
        print("No sleep events recorded in this run.")
        return
    
    print(f"\n=== SLEEP ANALYSIS ===")
    print(f"Total sleep cycles: {len(sleeps)}")
    
    # Buffer sizes
    buffer_sizes = [s['buffer_size'] for s in sleeps]
    avg_buffer_size = sum(buffer_sizes) / len(buffer_sizes) if buffer_sizes else 0
    print(f"Average buffer size: {avg_buffer_size:.1f}")
    
    # Pruned transitions
    pruned_total = sum([s['pruned'] for s in sleeps])
    print(f"Total transitions pruned: {pruned_total}")
    
    # Action std
    action_stds = [s['action_std'] for s in sleeps]
    avg_action_std = sum(action_stds) / len(action_stds) if action_stds else 0
    print(f"Average action std: {avg_action_std:.4f}")
    
    # Consumption events
    if 'total_consumptions' in sleeps[0]:
        consumptions = [s['total_consumptions'] for s in sleeps]
        total_consumptions = sum(consumptions)
        print(f"Total consumption events: {total_consumptions}")


def print_summary(stats: Dict):
    """Print a summary of training statistics."""
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Start time: {stats.get('start_time', 'Unknown')}")
    print(f"Total ticks: {stats.get('total_ticks', 0):,}")
    print(f"Total cycles: {stats.get('total_cycles', 0)}")
    print(f"Total deaths: {stats.get('total_deaths', 0)}")
    print(f"Wall time: {stats.get('wall_time_seconds', 0):.1f}s ({stats.get('total_ticks', 1)/stats.get('wall_time_seconds', 1):.0f} ticks/s)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_training.py <training_log.json>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"Loading training log from: {filepath}")
    
    # Load the training log
    log_data = load_training_log(filepath)
    
    # Print summary
    print_summary(log_data)
    
    # Analyze snapshots if available
    if log_data.get('snapshots'):
        print(f"Processing {len(log_data['snapshots'])} snapshots...")
        df = analyze_snapshots(log_data['snapshots'])
        
        # Create plots
        plot_vitals_over_time(df, filepath.replace('.json', '_vitals.png'))
        plot_drive_system(df, filepath.replace('.json', '_drives.png'))
        plot_rewards_and_satisfaction(df, filepath.replace('.json', '_rewards.png'))
        plot_position_trajectory(df, filepath.replace('.json', '_trajectory.png'))
    else:
        print("No snapshots found in log.")
    
    # Analyze deaths
    analyze_deaths(log_data.get('deaths', []))
    
    # Analyze sleep events
    analyze_sleep_events(log_data.get('sleeps', []))


if __name__ == "__main__":
    main()