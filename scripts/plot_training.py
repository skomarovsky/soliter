#!/usr/bin/env python3
"""
Plot training logs from train_soliter.py.

Generates:
  1. Trajectory map — agent path + resource positions + death markers
  2. Vitals timeline — energy, hydration, temperature over training
  3. Death analysis — causes, positions, life durations
  4. Learning curves — surprise, reward, buffer size
  5. Resource proximity — how close does agent get to resources?

Usage:
    python scripts/plot_training.py experiments/training_YYYYMMDD_HHMMSS.json
    python scripts/plot_training.py experiments/training_*.json --compare
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)


def load_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_trajectory_map(log: dict, ax: plt.Axes, title: str = "Agent Trajectory"):
    """Plot agent movement path with resources and death locations."""
    
    wc = log['world_config']
    W, H = wc['width'], wc['height']
    
    # Resources
    colors_map = {'feeder': '#4CAF50', 'fountain': '#2196F3', 'heater': '#FF5722'}
    markers_map = {'feeder': 's', 'fountain': 'D', 'heater': '^'}
    
    for r in log['resources']:
        rtype = r['resource_type']
        color = colors_map.get(rtype, 'gray')
        marker = markers_map.get(rtype, 'o')
        circle = Circle((r['x'], r['y']), r['radius'],
                        fill=False, edgecolor=color, linewidth=1.5, 
                        linestyle='--', alpha=0.6)
        ax.add_patch(circle)
        ax.plot(r['x'], r['y'], marker=marker, color=color,
                markersize=8, zorder=5, label=rtype if r['index'] == 0 else "")
    
    # Agent trajectory (color by time)
    if log['snapshots']:
        xs = [s['x'] for s in log['snapshots']]
        ys = [s['y'] for s in log['snapshots']]
        ticks = [s['tick'] for s in log['snapshots']]
        
        # Color segments by time
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(min(ticks), max(ticks))
        lc = LineCollection(segments, cmap='viridis', norm=norm, alpha=0.4, linewidth=0.5)
        lc.set_array(np.array(ticks[:-1]))
        ax.add_collection(lc)
        
        # Start position
        ax.plot(xs[0], ys[0], 'o', color='lime', markersize=10, zorder=10, label='Start')
    
    # Death positions
    if log['deaths']:
        dx = [d['x'] for d in log['deaths']]
        dy = [d['y'] for d in log['deaths']]
        ax.scatter(dx, dy, marker='X', c='red', s=60, zorder=10, 
                   edgecolors='darkred', linewidths=0.5, label='Death')
    
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.15)


def plot_trajectory_by_life(log: dict, ax: plt.Axes):
    """Plot each life as a separate colored trajectory."""
    
    wc = log['world_config']
    W, H = wc['width'], wc['height']
    
    # Resources (faded)
    colors_map = {'feeder': '#4CAF50', 'fountain': '#2196F3', 'heater': '#FF5722'}
    for r in log['resources']:
        circle = Circle((r['x'], r['y']), r['radius'],
                        fill=True, facecolor=colors_map.get(r['resource_type'], 'gray'),
                        alpha=0.15, edgecolor='none')
        ax.add_patch(circle)
    
    # Split trajectory by death events
    death_ticks = set(d['tick'] for d in log['deaths'])
    
    life_segments = []
    current_life = []
    for s in log['snapshots']:
        current_life.append(s)
        if s['tick'] in death_ticks:
            life_segments.append(current_life)
            current_life = []
    if current_life:
        life_segments.append(current_life)
    
    # Color each life differently
    cmap = plt.cm.tab10
    for i, life in enumerate(life_segments):
        if len(life) < 2:
            continue
        xs = [s['x'] for s in life]
        ys = [s['y'] for s in life]
        color = cmap(i % 10)
        ax.plot(xs, ys, '-', color=color, alpha=0.5, linewidth=0.8, label=f'Life {i+1}')
        ax.plot(xs[0], ys[0], 'o', color=color, markersize=5)
        if life[-1]['tick'] in death_ticks:
            ax.plot(xs[-1], ys[-1], 'X', color='red', markersize=8)
    
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_title('Trajectories by Life', fontsize=11)
    ax.grid(True, alpha=0.15)
    if len(life_segments) <= 10:
        ax.legend(loc='upper right', fontsize=6, framealpha=0.8)


def plot_vitals_timeline(log: dict, ax: plt.Axes):
    """Plot vitals over training time."""
    if not log['snapshots']:
        return
    
    ticks = [s['tick'] for s in log['snapshots']]
    energy = [s['energy'] for s in log['snapshots']]
    hydration = [s['hydration'] for s in log['snapshots']]
    temperature = [s['temperature'] for s in log['snapshots']]
    
    ax.plot(ticks, energy, '-', color='#4CAF50', alpha=0.7, linewidth=0.5, label='Energy')
    ax.plot(ticks, hydration, '-', color='#2196F3', alpha=0.7, linewidth=0.5, label='Hydration')
    ax.plot(ticks, temperature, '-', color='#FF5722', alpha=0.7, linewidth=0.5, label='Temperature')
    
    # Death markers
    for d in log['deaths']:
        ax.axvline(d['tick'], color='red', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Tick')
    ax.set_ylabel('Value')
    ax.set_title('Vitals Over Time', fontsize=11)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.15)


def plot_reward_and_surprise(log: dict, axes):
    """Plot reward per cycle and surprise trends."""
    ax1, ax2 = axes
    
    if not log['sleeps']:
        return
    
    cycles = [s['cycle'] for s in log['sleeps']]
    
    # Reward: sum from snapshots per cycle
    cycle_rewards = {}
    for s in log['snapshots']:
        c = s['cycle']
        cycle_rewards[c] = cycle_rewards.get(c, 0) + s['reward']
    
    reward_cycles = sorted(cycle_rewards.keys())
    rewards = [cycle_rewards[c] for c in reward_cycles]
    
    ax1.plot(reward_cycles, rewards, '-o', color='#9C27B0', markersize=2, linewidth=0.8)
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward per Cycle', fontsize=11)
    ax1.grid(True, alpha=0.15)
    
    # Surprise
    surprise_mean = [s['surprise_mean'] for s in log['sleeps']]
    surprise_min = [s['surprise_min'] for s in log['sleeps']]
    surprise_cutoff = [s['surprise_cutoff'] for s in log['sleeps']]
    
    ax2.semilogy(cycles, surprise_mean, '-', color='#FF9800', linewidth=1, label='Mean')
    ax2.semilogy(cycles, surprise_cutoff, '--', color='#795548', linewidth=0.8, label='Cutoff')
    ax2.semilogy(cycles, surprise_min, ':', color='#607D8B', linewidth=0.5, label='Min')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Surprise (log)')
    ax2.set_title('Replay Surprise', fontsize=11)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.15)


def plot_buffer_and_gating(log: dict, axes):
    """Plot buffer size and gating statistics."""
    ax1, ax2 = axes
    
    if not log['sleeps']:
        return
    
    cycles = [s['cycle'] for s in log['sleeps']]
    buf_sizes = [s['buffer_size'] for s in log['sleeps']]
    pruned = [s['pruned'] for s in log['sleeps']]
    gated = [s.get('gated_out_this_cycle', 0) for s in log['sleeps']]
    
    ax1.plot(cycles, buf_sizes, '-', color='#3F51B5', linewidth=1, label='Buffer size')
    ax1.bar(cycles, pruned, color='#F44336', alpha=0.4, width=0.8, label='Pruned')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Count')
    ax1.set_title('Buffer Regulation', fontsize=11)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.15)
    
    ax2.bar(cycles, gated, color='#FF9800', alpha=0.6)
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Gated Out')
    ax2.set_title('Wake Surprise Gating', fontsize=11)
    ax2.grid(True, alpha=0.15)


def plot_resource_proximity(log: dict, ax: plt.Axes):
    """How close does agent get to each resource over time?"""
    if not log['snapshots'] or not log['resources']:
        return
    
    wc = log['world_config']
    W, H = wc['width'], wc['height']
    
    # For each resource, track minimum distance over sliding windows
    resource_distances = {f"{r['resource_type']}_{r['index']}": [] for r in log['resources']}
    
    for snap in log['snapshots']:
        ax_pos = np.array([snap['x'], snap['y']])
        for r in log['resources']:
            r_pos = np.array([r['x'], r['y']])
            # Toroidal distance
            dx = abs(ax_pos[0] - r_pos[0])
            dy = abs(ax_pos[1] - r_pos[1])
            dx = min(dx, W - dx)
            dy = min(dy, H - dy)
            dist = np.sqrt(dx**2 + dy**2)
            key = f"{r['resource_type']}_{r['index']}"
            resource_distances[key].append(dist)
    
    # Plot minimum distance to any resource of each type
    ticks = [s['tick'] for s in log['snapshots']]
    
    type_colors = {'feeder': '#4CAF50', 'fountain': '#2196F3', 'heater': '#FF5722'}
    
    for rtype in ['feeder', 'fountain', 'heater']:
        # Get min distance across all resources of this type per tick
        type_keys = [k for k in resource_distances if k.startswith(rtype)]
        if not type_keys:
            continue
        min_dists = []
        for i in range(len(ticks)):
            d = min(resource_distances[k][i] for k in type_keys)
            min_dists.append(d)
        
        ax.plot(ticks, min_dists, '-', color=type_colors.get(rtype, 'gray'),
                alpha=0.6, linewidth=0.5, label=f'Nearest {rtype}')
    
    # Draw radius line
    if log['resources']:
        typical_radius = log['resources'][0]['radius']
        ax.axhline(typical_radius, color='red', linestyle='--', alpha=0.5, 
                   linewidth=0.8, label=f'Touch range ({typical_radius:.0f})')
    
    ax.set_xlabel('Tick')
    ax.set_ylabel('Distance')
    ax.set_title('Distance to Nearest Resource', fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)


def plot_death_analysis(log: dict, axes):
    """Death causes pie chart + life duration histogram."""
    ax1, ax2 = axes
    
    if not log['deaths']:
        ax1.text(0.5, 0.5, 'No deaths', ha='center', va='center', fontsize=12)
        ax2.text(0.5, 0.5, 'No deaths', ha='center', va='center', fontsize=12)
        return
    
    # Causes
    causes = {}
    for d in log['deaths']:
        causes[d['cause']] = causes.get(d['cause'], 0) + 1
    
    cause_colors = {
        'starvation': '#4CAF50',
        'dehydration': '#2196F3',
        'hypothermia': '#9C27B0',
        'hyperthermia': '#FF5722',
        'unknown': '#607D8B',
    }
    
    labels = list(causes.keys())
    sizes = list(causes.values())
    colors = [cause_colors.get(c, 'gray') for c in labels]
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 8})
    ax1.set_title(f'Death Causes (n={sum(sizes)})', fontsize=11)
    
    # Life durations
    durations = [d['life_duration'] for d in log['deaths']]
    ax2.hist(durations, bins=20, color='#F44336', alpha=0.7, edgecolor='darkred')
    ax2.set_xlabel('Life Duration (ticks)')
    ax2.set_ylabel('Count')
    ax2.set_title('Life Duration Distribution', fontsize=11)
    ax2.grid(True, alpha=0.15)


def generate_report(log_path: str, output_dir: str = None):
    """Generate complete analysis report from training log."""
    
    log = load_log(log_path)
    
    if output_dir is None:
        output_dir = str(Path(log_path).parent)
    
    stem = Path(log_path).stem
    
    # ── Figure 1: Trajectory maps ──────────────────────────────
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plot_trajectory_map(log, ax1, title="Full Trajectory")
    plot_trajectory_by_life(log, ax2)
    fig1.suptitle(f"Agent Movement — {log.get('total_ticks', '?'):,} ticks, "
                  f"{log.get('total_deaths', '?')} deaths", fontsize=13)
    fig1.tight_layout()
    p1 = Path(output_dir) / f"{stem}_trajectories.png"
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p1}")
    plt.close(fig1)
    
    # ── Figure 2: Vitals + Resource proximity ──────────────────
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    plot_vitals_timeline(log, ax1)
    plot_resource_proximity(log, ax2)
    fig2.tight_layout()
    p2 = Path(output_dir) / f"{stem}_vitals.png"
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p2}")
    plt.close(fig2)
    
    # ── Figure 3: Learning curves ──────────────────────────────
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_reward_and_surprise(log, [axes[0, 0], axes[0, 1]])
    plot_buffer_and_gating(log, [axes[1, 0], axes[1, 1]])
    fig3.tight_layout()
    p3 = Path(output_dir) / f"{stem}_learning.png"
    fig3.savefig(p3, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p3}")
    plt.close(fig3)
    
    # ── Figure 4: Death analysis ───────────────────────────────
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_death_analysis(log, [ax1, ax2])
    fig4.tight_layout()
    p4 = Path(output_dir) / f"{stem}_deaths.png"
    fig4.savefig(p4, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p4}")
    plt.close(fig4)
    
    print(f"\n  All plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Plot Soliter training logs")
    parser.add_argument('log_file', help='Path to training JSON log')
    parser.add_argument('--output-dir', help='Output directory (default: same as log)')
    args = parser.parse_args()
    
    print(f"Analyzing: {args.log_file}")
    generate_report(args.log_file, args.output_dir)


if __name__ == '__main__':
    main()
