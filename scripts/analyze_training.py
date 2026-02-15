#!/usr/bin/env python3
"""
Analyze training results from JSON log.

Usage:
    python scripts/analyze_training.py experiments/viz_training/training_20260214_180702.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_training(log_file):
    """Analyze training log and generate comprehensive report."""
    
    print("Loading training log...")
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("SOLITER TRAINING ANALYSIS REPORT")
    print("=" * 80)
    print(f"Log file: {log_file}")
    print()
    
    # Overall stats
    total_ticks = data.get('total_ticks', 0)
    snapshots = data.get('snapshots', [])
    deaths = data.get('deaths', [])
    
    if not snapshots:
        print("❌ No snapshots found in log!")
        return
    
    max_cycle = max(s['cycle'] for s in snapshots) if snapshots else 0
    
    print(f"📊 OVERALL STATISTICS")
    print(f"{'─' * 80}")
    print(f"Total ticks:        {total_ticks:,}")
    print(f"Total cycles:       {max_cycle}")
    print(f"Total snapshots:    {len(snapshots):,}")
    print(f"Total deaths:       {len(deaths)}")
    print()
    
    # Consumption analysis
    consumptions = [s for s in snapshots if s.get('consumed', '')]  # Non-empty string
    food_count = sum(1 for s in consumptions if s.get('consumed') == 'food')
    water_count = sum(1 for s in consumptions if s.get('consumed') == 'water')
    heat_count = sum(1 for s in consumptions if s.get('consumed') == 'heat')
    
    print(f"🍽️  CONSUMPTION ANALYSIS")
    print(f"{'─' * 80}")
    print(f"Total consumptions: {len(consumptions)}")
    print(f"  Food:             {food_count:4d} ({food_count/len(consumptions)*100:.1f}%)" if consumptions else "  Food:             0")
    print(f"  Water:            {water_count:4d} ({water_count/len(consumptions)*100:.1f}%)" if consumptions else "  Water:            0")
    print(f"  Heat:             {heat_count:4d} ({heat_count/len(consumptions)*100:.1f}%)" if consumptions else "  Heat:             0")
    print()
    
    # Death analysis
    if deaths:
        print(f"💀 DEATH ANALYSIS")
        print(f"{'─' * 80}")
        death_causes = defaultdict(int)
        for death in deaths:
            cause = death.get('cause', 'unknown')
            death_causes[cause] += 1
        
        for cause, count in sorted(death_causes.items(), key=lambda x: -x[1]):
            pct = count / len(deaths) * 100
            print(f"  {cause:20s}: {count:3d} ({pct:5.1f}%)")
        
        print(f"\nDeath rate: {len(deaths) / total_ticks * 1000:.2f} per 1000 ticks")
        print()
    
    # Behavior metrics by cycle
    print(f"🎯 BEHAVIOR BY CYCLE")
    print(f"{'─' * 80}")
    print(f"{'Cycle':>5} | {'Ticks':>6} | {'Heading Δ':>9} | {'Distance':>8} | {'Cons':>4} | {'Deaths':>6}")
    print(f"{'─' * 80}")
    
    for cycle in range(1, min(max_cycle + 1, 11)):  # First 10 cycles
        cycle_snaps = [s for s in snapshots if s['cycle'] == cycle]
        if not cycle_snaps:
            continue
        
        # Heading changes
        headings = [s['heading'] for s in cycle_snaps]
        heading_changes = []
        for i in range(1, len(headings)):
            change = abs(headings[i] - headings[i-1])
            if change > np.pi:
                change = 2*np.pi - change
            heading_changes.append(change)
        avg_heading = np.degrees(np.mean(heading_changes)) if heading_changes else 0
        
        # Distance traveled
        positions = [(s['x'], s['y']) for s in cycle_snaps]
        total_dist = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_dist += np.sqrt(dx**2 + dy**2)
        
        # Consumptions this cycle
        cycle_consumptions = sum(1 for s in cycle_snaps if s.get('consumed'))
        
        # Deaths this cycle
        cycle_deaths = sum(1 for d in deaths if d['cycle'] == cycle)
        
        # Ticks in cycle
        ticks_in_cycle = len(cycle_snaps)
        
        print(f"{cycle:5d} | {ticks_in_cycle:6d} | {avg_heading:7.1f}° | {total_dist:8.0f} | {cycle_consumptions:4d} | {cycle_deaths:6d}")
    
    print()
    
    # Vital statistics
    print(f"❤️  VITAL STATISTICS")
    print(f"{'─' * 80}")
    
    energies = [s['energy'] for s in snapshots]
    hydrations = [s['hydration'] for s in snapshots]
    temperatures = [s['temperature'] for s in snapshots]
    
    print(f"Energy:       min={min(energies):5.1f}, avg={np.mean(energies):5.1f}, max={max(energies):5.1f}")
    print(f"Hydration:    min={min(hydrations):5.1f}, avg={np.mean(hydrations):5.1f}, max={max(hydrations):5.1f}")
    print(f"Temperature:  min={min(temperatures):5.1f}, avg={np.mean(temperatures):5.1f}, max={max(temperatures):5.1f}")
    print()
    
    # Spatial analysis
    print(f"🗺️  SPATIAL ANALYSIS")
    print(f"{'─' * 80}")
    
    x_positions = [s['x'] for s in snapshots]
    y_positions = [s['y'] for s in snapshots]
    
    print(f"X range: {min(x_positions):.1f} to {max(x_positions):.1f}")
    print(f"Y range: {min(y_positions):.1f} to {max(y_positions):.1f}")
    print(f"Center of activity: ({np.mean(x_positions):.1f}, {np.mean(y_positions):.1f})")
    print()
    
    # Learning indicators
    print(f"📈 LEARNING INDICATORS")
    print(f"{'─' * 80}")
    
    # Compare first 5 cycles to last 5 cycles
    early_cycles = [s for s in snapshots if 1 <= s['cycle'] <= 5]
    late_cycles = [s for s in snapshots if max_cycle - 4 <= s['cycle'] <= max_cycle]
    
    if early_cycles and late_cycles:
        early_cons = sum(1 for s in early_cycles if s.get('consumed', ''))  # Non-empty string
        late_cons = sum(1 for s in late_cycles if s.get('consumed', ''))
        
        early_cons_rate = early_cons / len(early_cycles) * 100
        late_cons_rate = late_cons / len(late_cycles) * 100
        
        print(f"Consumption rate:")
        print(f"  Early (cycles 1-5):     {early_cons_rate:.2f}%")
        print(f"  Late (cycles {max_cycle-4}-{max_cycle}):    {late_cons_rate:.2f}%")
        print(f"  Improvement:            {late_cons_rate - early_cons_rate:+.2f}%")
        print()
        
        # Heading stability
        early_headings = [s['heading'] for s in early_cycles]
        late_headings = [s['heading'] for s in late_cycles]
        
        early_heading_changes = []
        for i in range(1, len(early_headings)):
            change = abs(early_headings[i] - early_headings[i-1])
            if change > np.pi:
                change = 2*np.pi - change
            early_heading_changes.append(change)
        
        late_heading_changes = []
        for i in range(1, len(late_headings)):
            change = abs(late_headings[i] - late_headings[i-1])
            if change > np.pi:
                change = 2*np.pi - change
            late_heading_changes.append(change)
        
        early_stability = np.degrees(np.mean(early_heading_changes)) if early_heading_changes else 0
        late_stability = np.degrees(np.mean(late_heading_changes)) if late_heading_changes else 0
        
        print(f"Movement stability (avg heading change):")
        print(f"  Early:                  {early_stability:.1f}°")
        print(f"  Late:                   {late_stability:.1f}°")
        print(f"  Improvement:            {early_stability - late_stability:+.1f}° (lower is better)")
    
    print()
    print("=" * 80)
    
    # Try to create plots if matplotlib available
    try:
        create_plots(data, log_file)
    except ImportError:
        print("📊 matplotlib not installed - skipping plots")
        print("   Install with: pip install matplotlib")
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

def create_plots(data, log_file):
    """Create visualization plots."""
    import matplotlib.pyplot as plt
    
    # Create output directory
    output_dir = Path(log_file).parent / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    snapshots = data['snapshots']
    
    # Sample for performance (every 10th snapshot)
    sample_interval = max(1, len(snapshots) // 1000)
    sampled = snapshots[::sample_interval]
    
    # Plot 1: Vitals over time
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    ticks = [s['tick'] for s in sampled]
    
    ax1.plot(ticks, [s['energy'] for s in sampled], alpha=0.7, linewidth=1)
    ax1.set_ylabel('Energy')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=20, color='r', linestyle='--', alpha=0.3, label='Critical')
    
    ax2.plot(ticks, [s['hydration'] for s in sampled], alpha=0.7, linewidth=1, color='blue')
    ax2.set_ylabel('Hydration')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=20, color='r', linestyle='--', alpha=0.3)
    
    ax3.plot(ticks, [s['temperature'] for s in sampled], alpha=0.7, linewidth=1, color='red')
    ax3.set_ylabel('Temperature')
    ax3.set_xlabel('Tick')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=20, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vitals_over_time.png', dpi=150)
    plt.close()
    
    # Plot 2: Movement heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x = [s['x'] for s in snapshots[::50]]
    y = [s['y'] for s in snapshots[::50]]
    
    ax.hexbin(x, y, gridsize=30, cmap='YlOrRd', mincnt=1)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Agent Movement Heatmap')
    ax.set_aspect('equal')
    plt.colorbar(ax.collections[0], ax=ax, label='Visits')
    plt.tight_layout()
    plt.savefig(output_dir / 'movement_heatmap.png', dpi=150)
    plt.close()
    
    # Plot 3: Consumptions by cycle
    max_cycle = max(s['cycle'] for s in snapshots)
    cycle_consumptions = []
    for cycle in range(1, max_cycle + 1):
        cycle_snaps = [s for s in snapshots if s['cycle'] == cycle]
        cons = sum(1 for s in cycle_snaps if s.get('consumed'))
        cycle_consumptions.append(cons)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(cycle_consumptions) + 1), cycle_consumptions, alpha=0.7)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Consumptions')
    ax.set_title('Consumptions per Cycle')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'consumptions_by_cycle.png', dpi=150)
    plt.close()
    
    print(f"\n✅ Plots saved to {output_dir}/")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_training.py <training_log.json>")
        print("\nExample:")
        print("  python scripts/analyze_training.py experiments/viz_training/training_20260214_180702.json")
        sys.exit(1)
    
    log_file = sys.argv[1]
    if not Path(log_file).exists():
        print(f"❌ File not found: {log_file}")
        sys.exit(1)
    
    analyze_training(log_file)
