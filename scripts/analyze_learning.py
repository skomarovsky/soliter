#!/usr/bin/env python3
"""
Quick analysis of long-term learning from training log.
Usage: python analyze_learning.py training_*.json
"""

import json
import sys
import numpy as np

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
    quartile_size = max(1, cycles // 4)
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
    mid_coverage = get_exploration_ratio(range(100, min(500, cycles + 1))) if cycles >= 100 else 0
    late_coverage = get_exploration_ratio(range(max(1, cycles - 100), cycles + 1))
    
    print(f"Early (cycles 1-100):    {early_coverage:.1f}% world explored")
    if mid_coverage > 0:
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
    print(f"\n5. LEARNING SATURATION")
    print("-" * 50)
    
    if cycles > 100:
        policy_losses = [s['policy_loss'] for s in sleeps if s['cycle'] > 100]
        
        if len(policy_losses) > 100:
            early_loss = np.mean(policy_losses[:50])
            late_loss = np.mean(policy_losses[-50:])
            loss_change = abs(late_loss - early_loss) / (early_loss + 1e-8)
            
            print(f"Policy loss change (mid → late): {loss_change:.2%}")
            
            if loss_change < 0.3:
                print("✅ SATURATED - Learning has plateaued (expected ~200-400 cycles)")
                print("   This is GOOD - means memory is consolidated")
            else:
                print("⚠️  STILL LEARNING - System has not reached saturation yet")
        else:
            print("⚠️  TOO EARLY - Need more cycles to assess saturation")
    else:
        print("⚠️  TOO EARLY - Need 100+ cycles to assess saturation")
    
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
