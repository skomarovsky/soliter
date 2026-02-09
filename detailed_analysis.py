#!/usr/bin/env python3
"""
Detailed analysis of training results from a JSON log file.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns


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


def plot_comprehensive_analysis(df: pd.DataFrame, save_prefix: str = "analysis"):
    """Create comprehensive plots for the training run."""
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Vitals over time
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(df['tick'], df['energy'], label='Energy', color='red', alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Energy Level Over Time')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Energy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(df['tick'], df['hydration'], label='Hydration', color='blue', alpha=0.7)
    ax2.axhline(y=0, color='blue', linestyle='--', alpha=0.5)
    ax2.set_title('Hydration Level Over Time')
    ax2.set_xlabel('Tick')
    ax2.set_ylabel('Hydration')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(df['tick'], df['temperature'], label='Temperature', color='orange', alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Hypothermia')
    ax3.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Hyperthermia')
    ax3.set_title('Body Temperature Over Time')
    ax3.set_xlabel('Tick')
    ax3.set_ylabel('Temperature (°C)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 2. Drive system
    ax4 = plt.subplot(4, 3, 4)
    ax4.plot(df['tick'], df['hunger'], label='Hunger', color='red', alpha=0.7)
    ax4.set_title('Hunger Drive Over Time')
    ax4.set_xlabel('Tick')
    ax4.set_ylabel('Hunger Drive')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(df['tick'], df['thirst'], label='Thirst', color='blue', alpha=0.7)
    ax5.set_title('Thirst Drive Over Time')
    ax5.set_xlabel('Tick')
    ax5.set_ylabel('Thirst Drive')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    ax6 = plt.subplot(4, 3, 6)
    ax6.plot(df['tick'], df['cold'], label='Cold', color='cyan', alpha=0.7)
    ax6.set_title('Cold Drive Over Time')
    ax6.set_xlabel('Tick')
    ax6.set_ylabel('Cold Drive')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 3. Rewards and satisfaction
    ax7 = plt.subplot(4, 3, 7)
    ax7.plot(df['tick'], df['reward'], label='Total Reward', color='black', alpha=0.7)
    ax7.set_title('Total Reward Over Time')
    ax7.set_xlabel('Tick')
    ax7.set_ylabel('Reward')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    ax8 = plt.subplot(4, 3, 8)
    ax8.plot(df['tick'], df['satisfaction'], label='Satisfaction', color='green', alpha=0.7)
    ax8.set_title('Satisfaction Over Time')
    ax8.set_xlabel('Tick')
    ax8.set_ylabel('Satisfaction')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    ax9 = plt.subplot(4, 3, 9)
    ax9.plot(df['tick'], df['discomfort'], label='Discomfort', color='red', alpha=0.7)
    ax9.set_title('Discomfort Over Time')
    ax9.set_xlabel('Tick')
    ax9.set_ylabel('Discomfort')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    # 4. Position and movement
    ax10 = plt.subplot(4, 3, 10)
    scatter = ax10.scatter(df['x'], df['y'], c=df['tick'], cmap='viridis', s=1, alpha=0.6)
    ax10.set_title('Agent Position Trajectory')
    ax10.set_xlabel('X Position')
    ax10.set_ylabel('Y Position')
    plt.colorbar(scatter, ax=ax10, label='Tick')
    ax10.grid(True, alpha=0.3)
    
    ax11 = plt.subplot(4, 3, 11)
    ax11.plot(df['tick'], df['velocity'], label='Velocity', color='purple', alpha=0.7)
    ax11.set_title('Movement Velocity Over Time')
    ax11.set_xlabel('Tick')
    ax11.set_ylabel('Velocity')
    ax11.grid(True, alpha=0.3)
    ax11.legend()
    
    ax12 = plt.subplot(4, 3, 12)
    ax12.plot(df['tick'], df['wakefulness'], label='Wakefulness', color='magenta', alpha=0.7)
    ax12.set_title('Wakefulness Over Time')
    ax12.set_xlabel('Tick')
    ax12.set_ylabel('Wakefulness')
    ax12.grid(True, alpha=0.3)
    ax12.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close()


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
    if life_durations:
        avg_life = sum(life_durations) / len(life_durations)
        median_life = np.median(life_durations)
        print(f"\nLife duration statistics:")
        print(f"  Average: {avg_life:.1f} ticks")
        print(f"  Median: {median_life:.1f} ticks")
        print(f"  Min: {min(life_durations):.1f} ticks")
        print(f"  Max: {max(life_durations):.1f} ticks")
    
    # Positions of deaths
    death_positions = [(d['x'], d['y']) for d in deaths]
    print(f"\nSample death positions: {death_positions[:5]}")


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
    print(f"Final buffer size: {buffer_sizes[-1] if buffer_sizes else 0}")
    
    # Pruned transitions
    pruned_counts = [s['pruned'] for s in sleeps]
    pruned_total = sum(pruned_counts)
    avg_pruned = sum(pruned_counts) / len(pruned_counts) if pruned_counts else 0
    print(f"Total transitions pruned: {pruned_total}")
    print(f"Average pruned per sleep: {avg_pruned:.1f}")
    
    # Action std
    action_stds = [s['action_std'] for s in sleeps]
    avg_action_std = sum(action_stds) / len(action_stds) if action_stds else 0
    print(f"Average action std: {avg_action_std:.4f}")
    print(f"Final action std: {action_stds[-1] if action_stds else 0:.4f}")
    
    # Consumption events
    if 'total_consumptions' in sleeps[0]:
        consumptions = [s['total_consumptions'] for s in sleeps]
        total_consumptions = sum(consumptions)
        avg_consumptions = sum(consumptions) / len(consumptions) if consumptions else 0
        print(f"Total consumption events: {total_consumptions}")
        print(f"Average consumptions per sleep: {avg_consumptions:.1f}")
        
        # Calculate consumption rate over time
        if len(consumptions) > 1:
            consumption_growth = (consumptions[-1] - consumptions[0]) / len(consumptions) if len(consumptions) > 0 else 0
            print(f"Consumption growth rate: {consumption_growth:.1f} per sleep")


def analyze_performance_over_time(df: pd.DataFrame):
    """Analyze performance trends over the training period."""
    print(f"\n=== PERFORMANCE ANALYSIS ===")
    
    # Calculate rolling averages for key metrics
    window_size = max(1, len(df) // 20)  # Use 1/20 of data for rolling average
    
    df['energy_rolling'] = df['energy'].rolling(window=window_size, min_periods=1).mean()
    df['hydration_rolling'] = df['hydration'].rolling(window=window_size, min_periods=1).mean()
    df['temperature_rolling'] = df['temperature'].rolling(window=window_size, min_periods=1).mean()
    df['reward_rolling'] = df['reward'].rolling(window=window_size, min_periods=1).mean()
    
    # Trends
    energy_trend = (df['energy_rolling'].iloc[-1] - df['energy_rolling'].iloc[0]) / df['energy_rolling'].iloc[0]
    hydration_trend = (df['hydration_rolling'].iloc[-1] - df['hydration_rolling'].iloc[0]) / df['hydration_rolling'].iloc[0]
    temperature_trend = (df['temperature_rolling'].iloc[-1] - df['temperature_rolling'].iloc[0]) / df['temperature_rolling'].iloc[0]
    reward_trend = (df['reward_rolling'].iloc[-1] - df['reward_rolling'].iloc[0]) / df['reward_rolling'].iloc[0]
    
    print(f"Trends over training period:")
    print(f"  Energy: {energy_trend:+.2%}")
    print(f"  Hydration: {hydration_trend:+.2%}")
    print(f"  Temperature: {temperature_trend:+.2%}")
    print(f"  Reward: {reward_trend:+.2%}")
    
    # Survival indicators
    low_energy_periods = (df['energy'] < 20).sum()
    low_hydration_periods = (df['hydration'] < 20).sum()
    dangerous_temperature_periods = ((df['temperature'] < 5) | (df['temperature'] > 95)).sum()
    
    print(f"\nStress indicators:")
    print(f"  Periods with very low energy (<20): {low_energy_periods}/{len(df)} ({100*low_energy_periods/len(df):.1f}%)")
    print(f"  Periods with very low hydration (<20): {low_hydration_periods}/{len(df)} ({100*low_hydration_periods/len(df):.1f}%)")
    print(f"  Periods with dangerous temperature: {dangerous_temperature_periods}/{len(df)} ({100*dangerous_temperature_periods/len(df):.1f}%)")


def print_summary(stats: Dict):
    """Print a summary of training statistics."""
    print(f"\n=== DETAILED TRAINING SUMMARY ===")
    print(f"Start time: {stats.get('start_time', 'Unknown')}")
    print(f"Total ticks: {stats.get('total_ticks', 0):,}")
    print(f"Total cycles: {stats.get('total_cycles', 0)}")
    print(f"Total deaths: {stats.get('total_deaths', 0)}")
    print(f"Wall time: {stats.get('wall_time_seconds', 0):.1f}s ({stats.get('total_ticks', 1)/stats.get('wall_time_seconds', 1):.0f} ticks/s)")
    
    # Calculate survival rate
    if stats.get('total_cycles', 0) > 0:
        survival_rate = (stats.get('total_cycles', 0) - stats.get('total_deaths', 0)) / stats.get('total_cycles', 1) * 100
        print(f"Survival rate: {survival_rate:.1f}% of cycles")


def main():
    if len(sys.argv) < 2:
        print("Usage: python detailed_analysis.py <training_log.json>")
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
        
        # Create comprehensive plots
        plot_comprehensive_analysis(df, filepath.replace('.json', '_analysis'))
        
        # Analyze performance over time
        analyze_performance_over_time(df)
    else:
        print("No snapshots found in log.")
    
    # Analyze deaths
    analyze_deaths(log_data.get('deaths', []))
    
    # Analyze sleep events
    analyze_sleep_events(log_data.get('sleeps', []))
    
    print(f"\nAnalysis complete. Plots saved with prefix: {filepath.replace('.json', '_analysis')}")


if __name__ == "__main__":
    main()