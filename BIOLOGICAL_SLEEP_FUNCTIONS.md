# BIOLOGICAL SLEEP FUNCTIONS: What Soliter Needs

## Current Status: Sleep is TOO MINIMAL!

Current sleep_cycle():
- ✅ Restore wakefulness
- ✅ Decay exploration
- ❌ NO memory consolidation
- ❌ NO synaptic pruning
- ❌ NO homeostatic scaling
- ❌ NO circadian rhythms

## What Real Sleep Does:

1. **Memory Consolidation** - Transfer short→long term memory
2. **Synaptic Pruning** - Remove weak connections
3. **Homeostatic Scaling** - Balance neuron activities
4. **Circadian Regulation** - Day/night synchronization

## Priority for Soliter:

### HIGH PRIORITY:
1. **Memory Consolidation** (Hebbian strengthening)
   - Remember resource locations
   - Consolidate successful patterns
   
2. **Homeostatic Scaling**
   - Prevent network instability
   - Balance neuron activities

### MEDIUM PRIORITY:
3. **Synaptic Pruning**
   - Remove noise
   - Improve efficiency

### LOW PRIORITY:
4. **Circadian Rhythms**
   - Sleep at night
   - Performance varies by time

## Why Old Code Failed:

Old: Homeostatic scaling + PPO batch updates = CONFLICT
New: Homeostatic scaling ALONE (no PPO) = SAFE

## Recommended Implementation:

```python
def sleep_cycle(self):
    # 1. Consolidate memories (Hebbian - no gradients!)
    self.consolidate_recent_successes()
    
    # 2. Homeostatic scaling (balance activities)
    self.balance_neuron_activities()
    
    # 3. Synaptic pruning (remove noise)
    self.prune_weak_connections()
    
    # 4. Decay exploration
    self.decay_exploration()
    
    # 5. Restore wakefulness
    self.agent.exit_sleep()
```

This gives biological realism WITHOUT PPO conflicts!
