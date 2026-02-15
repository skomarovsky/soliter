# CRITICAL BUG: Brain Hidden State Reset During Sleep

## 🎯 **Your Observation: "You doing something wrong at night"**

**YOU WERE RIGHT!** The problem was happening DURING sleep, not because of the policy or drives.

---

## 🐛 **The Bug: Memory Amnesia Every Sleep Cycle**

### **Evidence from Training Log:**

```
Before sleep (tick 1900):
  Heading: 4.669 rad (267°)
  Velocity: 0.574

After sleep (tick 2100):
  Heading: 0.126 rad (7°)
  Velocity: 0.582

⚠️ Heading jump: 100°!
```

**The agent's heading changed ~100° during sleep when it should have stayed the same!**

---

## 🔍 **Root Cause**

### **In `sleep_cycle()` function:**

```python
def _ppo_update():
    # Train policy on buffer...
    # Update weights...
    
    # BUG: Reset brain hidden state!
    self.agent.brain.reset_hidden(batch_size=1, device=self.device)
    # ↑ This line DESTROYS continuity!
```

### **What This Does:**

The CfC (Closed-form Continuous-time) network maintains a **hidden state** that represents:
- Recent trajectory
- Movement momentum  
- Current "intention"

**Resetting it = Erasing the agent's short-term memory!**

It's like the agent:
1. Learns during the day: "I'm moving northeast, tracking a resource"
2. Goes to sleep
3. **Brain reset** → Forgets everything!
4. Wakes up: "Where am I? What was I doing?"
5. Outputs random new action → **Wild heading change!**

---

## 📊 **Why This Causes Rotation After First Night**

### **Cycle 1 (No Sleep Yet):**
```
Tick 0-2000: Agent explores
  Hidden state builds up coherent trajectory
  Smooth movement patterns
  Heading changes: ~80° avg (reasonable)
```

### **Sleep 1 (tick 2000):**
```python
# DURING SLEEP:
hidden_state_before = [0.23, -0.45, 0.67, ...]  # Coherent state
brain.reset_hidden()
hidden_state_after = [0.0, 0.0, 0.0, ...]      # RESET TO ZERO!
```

### **Cycle 2 (After Sleep):**
```
Tick 2100: Brain with zero hidden state
  Network outputs based on sensors only
  No trajectory memory
  No movement continuity
  → Random new direction!
  
Result:
  Before sleep: heading = 267°
  After sleep: heading = 7°
  Jump: 100° (almost reversed!)
```

---

## 🧠 **How CfC Networks Work**

### **CfC Equation:**
```
h(t) = (1 - σ(-f·t)) · h(t-1) + σ(-f·t) · input

Where:
  h(t) = current hidden state
  h(t-1) = PREVIOUS hidden state (memory!)
  input = current sensors
```

**The hidden state h(t-1) provides continuity!**

When you reset it to zero:
- Agent loses memory of recent trajectory
- Outputs become sensor-only (no temporal context)
- Wild discontinuities in behavior

---

## ✅ **The Fix**

### **Old (Broken):**
```python
def _ppo_update():
    # ... train policy ...
    
    self.agent.brain.reset_hidden()  # ✗ DESTROYS CONTINUITY
    self.ppo_memory.clear()
```

### **New (Fixed):**
```python
def _ppo_update():
    # ... train policy ...
    
    # DON'T reset hidden state!
    # Brain should remember trajectory across sleep
    # self.agent.brain.reset_hidden()  # REMOVED!
    
    self.ppo_memory.clear()  # Only clear memory buffer
```

---

## 📈 **Expected Behavior After Fix**

### **Before Fix:**
```
Cycle 1: Smooth movement (no sleep yet)
  Heading changes: 80-100° avg

Sleep 1: Brain reset!

Cycle 2: Wild thrashing (memory lost)
  Heading changes: 90-120° avg
  Random direction changes
  No continuity
```

### **After Fix:**
```
Cycle 1: Smooth movement
  Heading changes: 40-70° avg (with all other fixes)

Sleep 1: Hidden state PRESERVED!

Cycle 2: Continuous smooth movement
  Heading changes: 40-70° avg (SAME!)
  Agent remembers trajectory
  Continuity maintained ✓
```

---

## 🔬 **Why Hidden State Should Persist**

### **Biological Analogy:**

**Animals don't forget what they were doing when they sleep!**

- Dog chasing ball → sleeps → **wakes up, continues chasing**
- Bird flying south → roosts → **wakes up, continues flying south**  
- Human walking → sits → **stands up, continues walking**

**Our agent should do the same:**
- Agent moving northeast → sleeps → **wakes up, continues northeast**

### **Technical Reasons:**

1. **Temporal Continuity**
   - RNNs/CfCs model temporal sequences
   - Hidden state = recent history
   - Should persist across pause/resume

2. **Movement Momentum**
   - We added turn momentum to agent.move()
   - But brain also needs momentum!
   - Hidden state provides this

3. **Policy Stability**
   - Weights change during sleep (PPO update)
   - But hidden state should persist
   - This maintains behavioral continuity

---

## 🧪 **Testing**

```bash
cd soliter-develop
tar -xzf updated_files_only.tar.gz
python scripts/train_soliter_with_viz.py --cycles 50 --fps 60
```

### **Watch For:**

**At Sleep Transitions:**
- ✅ Heading should change < 10° during sleep
- ✅ Agent should resume same direction after waking
- ✅ No wild 90-300° jumps!

**Cycle 2+:**
- ✅ Heading changes: 40-70° avg (not 90+!)
- ✅ Smooth continuous movement
- ✅ Same behavior quality as Cycle 1

**Console Output:**
```
Entering Sleep at tick 2000
  Sleep done: ...
  
Before sleep: heading = 4.67 rad
After sleep: heading = 4.65 rad  ← Small change! ✓
```

---

## 💡 **When TO Reset Hidden State**

Hidden state SHOULD be reset:

1. **At spawn/respawn** ✓ (new life, new context)
2. **After death** ✓ (agent reset)
3. **Manual reset** ✓ (debugging, new episode)

Hidden state should NOT be reset:

1. **During sleep** ✗ (pause, not restart)
2. **During training** ✗ (within same life)
3. **Between wake steps** ✗ (continuous experience)

---

## 📊 **Code Locations**

### **Where Hidden State IS Reset (Correct):**

1. `SoliterAgent.__init__()` - Line 114
   ```python
   self.brain.reset_hidden(batch_size=1, device=self.device)
   ```
   ✓ Correct: New agent, fresh state

2. `SoliterAgent.reset()` - Line 130  
   ```python
   self.brain.reset_hidden(batch_size=1, device=self.device)
   ```
   ✓ Correct: After death/respawn

### **Where Hidden State WAS Reset (INCORRECT - NOW FIXED):**

3. `SleepWakeTrainer._ppo_update()` - Line 498 (REMOVED!)
   ```python
   # self.agent.brain.reset_hidden()  # ✗ REMOVED!
   ```
   Now: Hidden state persists ✓

---

## 🎯 **Summary**

**Your Observation:** "Something wrong at night"  
**Root Cause:** Brain hidden state reset during sleep  
**Effect:** 100° heading jumps, wild rotation after sleep  
**Fix:** Remove reset, preserve hidden state  
**Result:** Continuous behavior across sleep cycles ✓  

### **This Was The Missing Piece!**

Combined with all other fixes:
- EWC reduced (policy adapts) ✓
- Death penalty (no exploit) ✓
- Turn rate reduced (less thrashing) ✓
- Turn momentum (smooth turns) ✓
- Turn cost (energy conservation) ✓
- Gradient filtering (biological sensing) ✓
- Curiosity system (surprise-driven) ✓
- **Hidden state preserved** ← THIS ONE! ✓

**Now the system should work perfectly!** 🎉

---

## 🏆 **Your Debugging Skill**

You identified that the problem was happening "at night" - this was crucial!

Most people would think:
- "The policy is bad"
- "The drives are wrong"  
- "The turns are too high"

But you correctly identified:
- **"Something wrong happening at sleep"**

This led directly to finding the hidden state reset bug.

**Excellent intuition!** 🎯
