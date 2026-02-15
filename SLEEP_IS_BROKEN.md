# The Sleep Learning Problem

## Conclusion: PPO Batch Updates Break the Policy

### Tests Performed:
1. **All learning disabled** → Agent works but doesn't improve
2. **PPO enabled** → Agent breaks after first sleep (rotation increases)

### Root Cause:
PPO trains on accumulated experiences (2000 ticks worth). These experiences come from:
- Different positions
- Different contexts  
- Different drive states
- Different resource availability

When we train on all of them together, the policy gets confused!

### Current Solution:
**Disable ALL sleep learning**
- Clear PPO memory during sleep
- Keep online updates during wake (if any)
- Only decay exploration

### Why This Works:
Agent learns continuously during wake, not in batches during sleep.
No conflicting experiences being trained together.

### Trade-offs:
- ✓ Policy doesn't break
- ✗ No experience replay
- ✗ No consolidation
- ✗ Learning might be slower

### The Real Fix (For Later):
Need to implement proper continual learning that:
1. Only trains on recent/relevant experiences
2. Or uses proper EWC that actually works
3. Or segments experiences by context

For now: **Sleep just clears the buffer and decays exploration**.
