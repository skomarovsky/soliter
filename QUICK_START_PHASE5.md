# Phase 5 Fixes - Quick Start Guide

## What Was Fixed

Your Phase 5 training revealed that agents couldn't learn because the world (300×300) was too large for them to discover resources during their exploration phase. I've implemented three critical fixes:

1. **World size reduced**: 300×300 → 150×150
2. **Resource counts adjusted**: Scaled to maintain density
3. **Spawn location optimized**: Agents now spawn near resource clusters

## Quick Test (5 minutes)

```bash
cd soliter-develop
python scripts/train_soliter.py --cycles 50 --output-dir experiments/phase5_quick_test
```

**What to look for in the output:**
- First consumption should appear before cycle 20
- "Consumed" column should show numbers (not just dots)
- By cycle 50, you should see 10+ consumption events

## Expected Results

### Before (Your Training Log)
- 400,000 ticks, 200 cycles
- **7 total consumptions** (0.00175% rate)
- Agent explored only 23% of world
- No learning occurred

### After (With Fixes)
- 100,000 ticks, 50 cycles
- **50-150 consumptions** (>10% rate)
- Agent explores >50% of world
- **Actual learning visible**

## Files Modified

1. `configs/default.yaml` - World size and resource counts
2. `scripts/train_soliter.py` - Spawn and respawn logic

**No other changes needed** - all your Phase 4 work (memory consolidation, EWC, drive system) is intact.

## Validation Checklist

After running the 50-cycle test, check:

- [ ] Console shows consumption events (food/water/heat)
- [ ] Satisfaction values increase over time
- [ ] Agent position varies widely (not stuck in one spot)
- [ ] Fewer deaths from the same cause repeatedly
- [ ] Buffer size grows steadily (experiences being collected)

## Next Steps

### If Test Passes ✅
Run longer experiment:
```bash
python scripts/train_soliter.py --cycles 200 --output-dir experiments/phase5_validation
```

Then analyze with:
```bash
python scripts/plot_training.py experiments/phase5_validation/training_*.json
```

### If Test Fails ⚠️

**Low consumption (<5 events):**
- Try even smaller world (100×100)
- Increase initial action_std to 1.0

**Resource discovery but no consumption:**
- Check drive_system activation in logs
- Verify gradient sensor responses

**Consumption but no learning trend:**
- Check if rewards are being collected
- Verify PPO updates are occurring

## Understanding the Fix

The problem wasn't your code - it was a **scale mismatch**:

```
Old Configuration:
  World: 300×300 (90,000 units²)
  Agent exploration: ~60 unit radius (11,300 units²)
  Coverage: 12.5%
  Result: Most resources unreachable → no learning signal

New Configuration:
  World: 150×150 (22,500 units²)  
  Agent exploration: ~60 unit radius (11,300 units²)
  Coverage: 50%
  Result: Most resources discoverable → learning possible
```

## Why This Matters for Consciousness Research

Your project aims to demonstrate consciousness prerequisites through continual learning. But **you can't test consciousness emergence without functional learning first**. These fixes ensure:

1. **Agents can learn** - Resources are discoverable
2. **Drive system functions** - Internal rewards get satisfied
3. **Memory consolidates** - There's something to remember
4. **Context integrates** - Experiences across cycles connect

Without these fixes, you were testing memory systems in a "sensory deprivation" environment. Now the agent can actually engage with its world.

## Contact

If you encounter issues, the key diagnostic is **consumption rate**:
- <1% = Still broken (try smaller world or higher action_std)
- 1-10% = Marginal (might work with more cycles)
- >10% = Working! (proceed to long-horizon testing)

Good luck! The mechanical systems are solid - they just needed a world they could actually explore.
