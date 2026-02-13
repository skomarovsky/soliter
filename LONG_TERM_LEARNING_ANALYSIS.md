# Long-Term Learning Analysis Framework

## 🎯 **What We're Testing**

Long-term learning examines:
1. **Memory Consolidation** - Does agent retain old knowledge?
2. **Catastrophic Forgetting** - Does learning Task B destroy Task A?
3. **Continual Improvement** - Does performance increase over time?
4. **Fisher Information Saturation** - When does memory "fill up"?
5. **Synaptic Scaling** - Does sleep prevent weight explosion?

---

## 📊 **Key Metrics to Track**

### 1. **Consumption Rate Over Time**
```
Cycles 1-100:   X consumptions/cycle (baseline)
Cycles 101-200: Y consumptions/cycle (should increase)
Cycles 201-300: Z consumptions/cycle (should plateau)
Cycles 301+:    Stable or slight improvement
```

**What to look for:**
- ✅ Upward trend in first 200 cycles = learning
- ✅ Plateau after 200-400 cycles = memory saturation
- ❌ Decline after plateau = catastrophic forgetting

### 2. **Life Duration Trends**
```python
# Group deaths by quartiles
Q1 = cycles 1-250:   Mean life duration
Q2 = cycles 251-500: Mean life duration  
Q3 = cycles 501-750: Mean life duration
Q4 = cycles 751-1000: Mean life duration

# Expected pattern:
Q1 < Q2 < Q3 ≈ Q4  (improvement then plateau)
```

### 3. **Spatial Coverage**
```python
# Measure exploration over time
Early (cycles 1-100):   % world explored
Middle (cycles 101-500): % world explored
Late (cycles 501-1000):  % world explored

# Expected:
Early:  40-60% (exploring)
Middle: 70-90% (learned world)
Late:   80-95% (efficient routes)
```

### 4. **Buffer Size & Pruning**
```python
# From training log "sleeps" section:
buffer_size over time
pruned count over time

# Expected pattern:
Cycles 1-100:   Buffer grows (4000-5000)
Cycles 101-400: Buffer stable (5000-5500) - equilibrium
Cycles 401+:    Buffer stable or shrinking (consolidation)
```

### 5. **Policy Loss Over Time**
```python
# From "sleeps" section:
policy_loss over cycles

# Expected pattern:
High variance early (0-100 cycles) = active learning
Low variance middle (100-400) = consolidation
Stable late (400+) = mastery
```

### 6. **Action Std Decay**
```python
# From "sleeps" section:
action_std over cycles

# Expected pattern:
Starts: 0.5 (high exploration)
Ends:   0.1 (low exploration, exploitation)

# Formula: std *= 0.995 per cycle
# After 500 cycles: 0.5 * (0.995^500) ≈ 0.041
```

---

## 🔬 **Analysis Scripts**

### Script 1: Quick Metrics Overview

```python
#!/usr/bin/env python3
"""
Quick analysis of long-term learning from training log.
Usage: python analyze_learning.py training_*.json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt

def analyze_long_term_learning(log_path):
    with open(log_path) as f:
        log = json.load(f)
    
    cycles = log['total_cycles']
    deaths = log['deaths']
    sleeps = log['sleeps']
    snapshots = log['snapshots']
    
    print(f"\n{'='*70}")
    print(f"LONG-TERM LEARNING ANALYSIS - {cycles} Cycles")
    print(f"{'='*70}\n")
    
    # 1. CONSUMPTION TREND
    print("1. CONSUMPTION RATE OVER TIME")
    print("-" * 50)
    
    cycle_consumptions = {}
    for sleep in sleeps:
        cycle = sleep['cycle']
        total = sleep['total_consumptions']
        cycle_consumptions[cycle] = total
    
    # Group by quartiles
    quartile_size = cycles // 4
    q1_cycles = range(1, quartile_size + 1)
    q2_cycles = range(quartile_size + 1, quartile_size * 2 + 1)
    q3_cycles = range(quartile_size * 2 + 1, quartile_size * 3 + 1)
    q4_cycles = range(quartile_size * 3 + 1, cycles + 1)
    
    def get_consumption_rate(cycle_range):
        consumptions = []
        for c in cycle_range:
            if c in cycle_consumptions and c-1 in cycle_consumptions:
                rate = cycle_consumptions[c] - cycle_consumptions[c-1]
                consumptions.append(rate)
        return np.mean(consumptions) if consumptions else 0
    
    q1_rate = get_consumption_rate(q1_cycles)
    q2_rate = get_consumption_rate(q2_cycles)
    q3_rate = get_consumption_rate(q3_cycles)
    q4_rate = get_consumption_rate(q4_cycles)
    
    print(f"Q1 (cycles 1-{quartile_size}):          {q1_rate:.1f} consumptions/cycle")
    print(f"Q2 (cycles {quartile_size+1}-{quartile_size*2}):        {q2_rate:.1f} consumptions/cycle")
    print(f"Q3 (cycles {quartile_size*2+1}-{quartile_size*3}):       {q3_rate:.1f} consumptions/cycle")
    print(f"Q4 (cycles {quartile_size*3+1}-{cycles}):      {q4_rate:.1f} consumptions/cycle")
    
    improvement = ((q4_rate - q1_rate) / q1_rate * 100) if q1_rate > 0 else 0
    print(f"\nImprovement: {improvement:+.1f}%")
    
    if improvement > 20:
        print("✅ STRONG LEARNING - Consumption rate improving significantly")
    elif improvement > 5:
        print("✅ MODERATE LEARNING - Steady improvement")
    elif improvement > -5:
        print("⚠️  PLATEAU - Performance stable (expected after ~200 cycles)")
    else:
        print("❌ DECLINE - Possible catastrophic forgetting")
    
    # 2. LIFE DURATION TREND
    print(f"\n2. LIFE DURATION OVER TIME")
    print("-" * 50)
    
    q1_deaths = [d for d in deaths if d['cycle'] <= quartile_size]
    q2_deaths = [d for d in deaths if quartile_size < d['cycle'] <= quartile_size * 2]
    q3_deaths = [d for d in deaths if quartile_size * 2 < d['cycle'] <= quartile_size * 3]
    q4_deaths = [d for d in deaths if d['cycle'] > quartile_size * 3]
    
    q1_life = np.mean([d['life_duration'] for d in q1_deaths]) if q1_deaths else 0
    q2_life = np.mean([d['life_duration'] for d in q2_deaths]) if q2_deaths else 0
    q3_life = np.mean([d['life_duration'] for d in q3_deaths]) if q3_deaths else 0
    q4_life = np.mean([d['life_duration'] for d in q4_deaths]) if q4_deaths else 0
    
    print(f"Q1: {q1_life:.0f} ticks average life")
    print(f"Q2: {q2_life:.0f} ticks average life")
    print(f"Q3: {q3_life:.0f} ticks average life")
    print(f"Q4: {q4_life:.0f} ticks average life")
    
    if q4_life > q1_life * 1.5:
        print("✅ STRONG SURVIVAL IMPROVEMENT - Agent learning to survive longer")
    elif q4_life > q1_life * 1.1:
        print("✅ MODERATE IMPROVEMENT - Agent gradually improving")
    else:
        print("⚠️  STABLE - Survival time not increasing significantly")
    
    # 3. SPATIAL EXPLORATION
    print(f"\n3. SPATIAL EXPLORATION OVER TIME")
    print("-" * 50)
    
    def get_exploration_ratio(cycle_range):
        positions = [(s['x'], s['y']) for s in snapshots if s['cycle'] in cycle_range]
        if not positions:
            return 0
        
        xs, ys = zip(*positions)
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        
        world_size = log['world_config']['width']
        coverage = (x_range * y_range) / (world_size ** 2)
        return coverage * 100
    
    early_coverage = get_exploration_ratio(range(1, min(100, cycles + 1)))
    mid_coverage = get_exploration_ratio(range(100, min(500, cycles + 1)))
    late_coverage = get_exploration_ratio(range(max(1, cycles - 100), cycles + 1))
    
    print(f"Early (cycles 1-100):    {early_coverage:.1f}% world explored")
    print(f"Middle (cycles 100-500): {mid_coverage:.1f}% world explored")
    print(f"Late (last 100 cycles):  {late_coverage:.1f}% world explored")
    
    if late_coverage > 60:
        print("✅ EXCELLENT EXPLORATION - Agent knows the world well")
    elif late_coverage > 40:
        print("✅ GOOD EXPLORATION - Agent has covered significant territory")
    else:
        print("⚠️  LIMITED EXPLORATION - Agent may be staying in small area")
    
    # 4. MEMORY CONSOLIDATION
    print(f"\n4. MEMORY SYSTEM HEALTH")
    print("-" * 50)
    
    buffer_sizes = [s['buffer_size'] for s in sleeps]
    pruned_counts = [s['pruned'] for s in sleeps]
    
    early_buffer = np.mean(buffer_sizes[:min(50, len(buffer_sizes))])
    late_buffer = np.mean(buffer_sizes[-50:])
    
    early_pruned = np.mean(pruned_counts[:min(50, len(pruned_counts))])
    late_pruned = np.mean(pruned_counts[-50:])
    
    print(f"Buffer size:")
    print(f"  Early: {early_buffer:.0f} transitions")
    print(f"  Late:  {late_buffer:.0f} transitions")
    
    print(f"Pruning rate:")
    print(f"  Early: {early_pruned:.0f} transitions/cycle")
    print(f"  Late:  {late_pruned:.0f} transitions/cycle")
    
    if late_buffer < early_buffer * 1.2:
        print("✅ MEMORY STABLE - Buffer not growing unbounded")
    else:
        print("⚠️  MEMORY GROWTH - Buffer size increasing (may need adjustment)")
    
    # 5. FISHER INFORMATION SATURATION
    print(f"\n5. FISHER INFORMATION SATURATION")
    print("-" * 50)
    
    # Look for plateau in policy loss (indicates saturation)
    policy_losses = [s['policy_loss'] for s in sleeps if s['cycle'] > 100]
    
    if len(policy_losses) > 100:
        early_loss = np.mean(policy_losses[:50])
        late_loss = np.mean(policy_losses[-50:])
        loss_change = abs(late_loss - early_loss) / (early_loss + 1e-8)
        
        print(f"Policy loss change (cycle 100+ → late): {loss_change:.2%}")
        
        if loss_change < 0.3:
            print("✅ SATURATED - Learning has plateaued (expected ~200-400 cycles)")
            print("   This is GOOD - means memory is consolidated")
        else:
            print("⚠️  STILL LEARNING - System has not reached saturation yet")
    else:
        print("⚠️  TOO EARLY - Need more cycles to assess saturation")
    
    # 6. ACTION STD DECAY
    print(f"\n6. EXPLORATION VS EXPLOITATION")
    print("-" * 50)
    
    action_stds = [s['action_std'] for s in sleeps]
    
    if action_stds:
        start_std = action_stds[0]
        end_std = action_stds[-1]
        
        print(f"Action std start: {start_std:.4f}")
        print(f"Action std end:   {end_std:.4f}")
        print(f"Decay rate:       {(1 - end_std/start_std)*100:.1f}%")
        
        if end_std < 0.15:
            print("✅ HIGH EXPLOITATION - Agent using learned behaviors")
        elif end_std < 0.3:
            print("⚠️  BALANCED - Still some exploration")
        else:
            print("⚠️  HIGH EXPLORATION - Agent still searching")
    
    print(f"\n{'='*70}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*70}\n")
    
    # Overall score
    score = 0
    if improvement > 5: score += 1
    if q4_life > q1_life * 1.1: score += 1
    if late_coverage > 40: score += 1
    if late_buffer < early_buffer * 1.2: score += 1
    
    if score >= 3:
        print("✅ STRONG LONG-TERM LEARNING")
        print("   Agent shows clear improvement and memory consolidation")
    elif score >= 2:
        print("✅ MODERATE LONG-TERM LEARNING")
        print("   Agent is learning but has room for improvement")
    else:
        print("⚠️  LIMITED LONG-TERM LEARNING")
        print("   Agent may need more cycles or parameter tuning")
    
    print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_learning.py training_*.json")
        sys.exit(1)
    
    analyze_long_term_learning(sys.argv[1])
```

### Script 2: Visualization Dashboard

```python
#!/usr/bin/env python3
"""
Create comprehensive visualization dashboard for long-term learning.
Usage: python visualize_learning.py training_*.json
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def visualize_long_term_learning(log_path):
    with open(log_path) as f:
        log = json.load(f)
    
    deaths = log['deaths']
    sleeps = log['sleeps']
    snapshots = log['snapshots']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. CONSUMPTION RATE OVER TIME
    ax1 = fig.add_subplot(gs[0, :2])
    
    cycles = [s['cycle'] for s in sleeps]
    consumptions = [s['total_consumptions'] for s in sleeps]
    
    # Calculate rate (delta)
    rates = [consumptions[i] - consumptions[i-1] if i > 0 else 0 
             for i in range(len(consumptions))]
    
    ax1.plot(cycles, rates, alpha=0.3, color='blue')
    # Rolling average
    window = 10
    if len(rates) > window:
        rates_smooth = np.convolve(rates, np.ones(window)/window, mode='valid')
        ax1.plot(cycles[window-1:], rates_smooth, linewidth=2, color='blue', label='Consumption Rate')
    
    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('Consumptions per Cycle', fontsize=12)
    ax1.set_title('Consumption Rate Over Time (Learning Progress)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. LIFE DURATION OVER TIME
    ax2 = fig.add_subplot(gs[0, 2])
    
    death_cycles = [d['cycle'] for d in deaths]
    life_durations = [d['life_duration'] for d in deaths]
    
    ax2.scatter(death_cycles, life_durations, alpha=0.5, s=20)
    
    # Trend line
    if len(death_cycles) > 10:
        z = np.polyfit(death_cycles, life_durations, 1)
        p = np.poly1d(z)
        ax2.plot(death_cycles, p(death_cycles), "r--", linewidth=2, label=f'Trend (slope={z[0]:.1f})')
    
    ax2.set_xlabel('Cycle', fontsize=12)
    ax2.set_ylabel('Life Duration (ticks)', fontsize=12)
    ax2.set_title('Survival Time Improvement', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. SPATIAL EXPLORATION (MOVEMENT HEATMAP)
    ax3 = fig.add_subplot(gs[1, :2])
    
    xs = [s['x'] for s in snapshots[::10]]  # Sample for performance
    ys = [s['y'] for s in snapshots[::10]]
    
    h = ax3.hist2d(xs, ys, bins=30, cmap='YlOrRd')
    plt.colorbar(h[3], ax=ax3, label='Visit Frequency')
    
    ax3.set_xlabel('X Position', fontsize=12)
    ax3.set_ylabel('Y Position', fontsize=12)
    ax3.set_title('Spatial Exploration Heatmap', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    
    # 4. BUFFER SIZE OVER TIME
    ax4 = fig.add_subplot(gs[1, 2])
    
    buffer_sizes = [s['buffer_size'] for s in sleeps]
    
    ax4.plot(cycles, buffer_sizes, linewidth=2, color='green')
    ax4.axhline(y=np.mean(buffer_sizes[-50:]), color='red', linestyle='--', 
                label=f'Late Mean: {np.mean(buffer_sizes[-50:]):.0f}')
    
    ax4.set_xlabel('Cycle', fontsize=12)
    ax4.set_ylabel('Buffer Size', fontsize=12)
    ax4.set_title('Memory Buffer Growth', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. POLICY LOSS (LEARNING CURVE)
    ax5 = fig.add_subplot(gs[2, 0])
    
    policy_losses = [s['policy_loss'] for s in sleeps]
    
    ax5.semilogy(cycles, policy_losses, alpha=0.5, color='purple')
    
    # Rolling average
    if len(policy_losses) > window:
        loss_smooth = np.convolve(policy_losses, np.ones(window)/window, mode='valid')
        ax5.semilogy(cycles[window-1:], loss_smooth, linewidth=2, 
                     color='purple', label='Policy Loss (smoothed)')
    
    ax5.set_xlabel('Cycle', fontsize=12)
    ax5.set_ylabel('Policy Loss (log scale)', fontsize=12)
    ax5.set_title('Policy Loss Over Time', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend()
    
    # 6. ACTION STD DECAY
    ax6 = fig.add_subplot(gs[2, 1])
    
    action_stds = [s['action_std'] for s in sleeps]
    
    ax6.plot(cycles, action_stds, linewidth=2, color='orange')
    ax6.axhline(y=0.1, color='red', linestyle='--', label='Target Min (0.1)')
    
    ax6.set_xlabel('Cycle', fontsize=12)
    ax6.set_ylabel('Action Std', fontsize=12)
    ax6.set_title('Exploration Decay', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. DEATH CAUSES OVER TIME
    ax7 = fig.add_subplot(gs[2, 2])
    
    # Group deaths by quartiles
    n_cycles = log['total_cycles']
    quartile_size = n_cycles // 4
    
    causes = {'starvation': 0, 'dehydration': 0, 'hypothermia': 0, 'hyperthermia': 0}
    quartile_causes = [{k: 0 for k in causes} for _ in range(4)]
    
    for d in deaths:
        cycle = d['cycle']
        cause = d['cause']
        
        q_idx = min(3, (cycle - 1) // quartile_size)
        if cause in quartile_causes[q_idx]:
            quartile_causes[q_idx][cause] += 1
    
    # Stack bar chart
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    starvation = [q['starvation'] for q in quartile_causes]
    dehydration = [q['dehydration'] for q in quartile_causes]
    hypothermia = [q['hypothermia'] for q in quartile_causes]
    hyperthermia = [q['hyperthermia'] for q in quartile_causes]
    
    x = np.arange(len(labels))
    width = 0.6
    
    ax7.bar(x, starvation, width, label='Starvation', color='brown')
    ax7.bar(x, dehydration, width, bottom=starvation, label='Dehydration', color='blue')
    ax7.bar(x, hypothermia, width, bottom=np.array(starvation)+np.array(dehydration), 
            label='Hypothermia', color='cyan')
    
    ax7.set_xlabel('Time Period', fontsize=12)
    ax7.set_ylabel('Death Count', fontsize=12)
    ax7.set_title('Death Causes Over Time', fontsize=14, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(labels)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Long-Term Learning Analysis - {n_cycles} Cycles', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('long_term_learning_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: long_term_learning_analysis.png")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_learning.py training_*.json")
        sys.exit(1)
    
    visualize_long_term_learning(sys.argv[1])
```

---

## 🧪 **Testing Protocol**

### Phase 1: Short Run (100 cycles, ~30 min)
```bash
python scripts/train_soliter.py --cycles 100 --output-dir experiments/learning_test
python analyze_learning.py experiments/learning_test/training_*.json
```

**Validate**:
- ✅ Consumption rate increasing
- ✅ Buffer size stable
- ✅ No immediate catastrophic forgetting

### Phase 2: Medium Run (500 cycles, ~2-3 hours)
```bash
python scripts/train_soliter.py --cycles 500 --output-dir experiments/learning_medium
python visualize_learning.py experiments/learning_medium/training_*.json
```

**Look for**:
- ✅ Fisher Information saturation (plateau around cycle 200-400)
- ✅ Life duration improvement
- ✅ Spatial exploration completion

### Phase 3: Long Run (1000 cycles, ~5-6 hours)
```bash
nohup python scripts/train_soliter.py --cycles 1000 --output-dir experiments/learning_long > training.log 2>&1 &

# Check progress:
tail -f training.log
```

**Analyze**:
- ✅ Sustained performance (no catastrophic forgetting)
- ✅ Memory consolidation working
- ✅ Synaptic scaling preventing weight explosion

---

## 📈 **Expected Patterns**

### Successful Long-Term Learning:

```
Consumption Rate:
  Cycles 1-100:   ↗↗↗ (rapid improvement)
  Cycles 101-300: ↗   (steady improvement)
  Cycles 301+:    →   (plateau - saturation)

Life Duration:
  Early:  Short, variable
  Middle: Increasing
  Late:   Stable, long lives

Buffer Size:
  Early:  Growing (learning new things)
  Middle: Stable (equilibrium reached)
  Late:   Stable or shrinking (consolidation)

Policy Loss:
  Early:  High variance (active learning)
  Middle: Decreasing (consolidation)
  Late:   Low, stable (mastery)
```

---

## 🎯 **Success Criteria**

After 500+ cycles, verify:

1. ✅ **Consumption rate improved** by >20% (Q1 → Q4)
2. ✅ **Life duration improved** by >10% (Q1 → Q4)
3. ✅ **Spatial exploration** >60% world coverage
4. ✅ **Buffer size stable** (not growing unbounded)
5. ✅ **Fisher Information saturated** (plateau visible)
6. ✅ **No catastrophic forgetting** (Q4 performance ≥ Q3)

---

## 🔬 **Research Questions Answered**

### 1. Does the agent actually learn?
**Metric**: Consumption rate over time
**Expected**: Upward trend, then plateau

### 2. Does memory consolidation work?
**Metric**: Buffer size stability, Fisher Information plateau
**Expected**: Buffer stabilizes, policy loss plateaus

### 3. Does synaptic scaling prevent weight explosion?
**Metric**: Policy loss remains bounded, no divergence
**Expected**: Losses stay in reasonable range (0.1-100)

### 4. When does saturation occur?
**Metric**: Plateau in learning curves
**Expected**: Around cycle 200-400 (as predicted)

### 5. Does the agent retain old knowledge?
**Metric**: Late performance ≥ middle performance
**Expected**: Q4 metrics ≥ Q3 (no decline)

---

## 💾 **Save These Scripts**

I'll create these analysis scripts for you:

```bash
cd soliter-develop/scripts
# analyze_learning.py - Quick metrics
# visualize_learning.py - Comprehensive dashboard
```

---

**Status**: ✅ Analysis framework complete
**Usage**: Run after any training session with 50+ cycles
**Output**: Quantitative metrics + visualization dashboard
**Purpose**: Validate consciousness prerequisites through learning curves
