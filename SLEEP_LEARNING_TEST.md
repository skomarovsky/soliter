# CRITICAL TEST: Is Sleep Learning Breaking the Policy?

## 🎯 Your Hypothesis

> "I assume that our learning algorithm at night is broken. Agent dying rotating on the same place after 1 night."

**You're probably right!** Let's test this.

---

## ⚠️ THIS VERSION HAS ALL SLEEP LEARNING DISABLED

### **What's Disabled:**

```python
# In sleep_cycle():
# DISABLED: PPO update
# DISABLED: Replay consolidation  
# DISABLED: Fisher matrix updates
# DISABLED: Homeostatic scaling
# DISABLED: EWC updates

# ONLY KEPT: Exploration decay (action_std)
```

---

## 🧪 Test Protocol

### **Run the agent:**

```bash
cd soliter-develop
tar -xzf updated_files_only.tar.gz
python scripts/train_soliter_with_viz.py --cycles 10 --fps 60
```

### **Watch carefully:**

**Cycle 1 (Before first sleep):**
- How does agent behave?
- Is movement smooth?
- Does it find resources?

**Sleep 1 (tick 2000):**
- Console will say: "⚠️ LEARNING DISABLED FOR TESTING"

**Cycle 2 (After first sleep):**
- **CRITICAL QUESTION:** Does agent still behave well?
- Or does it start rotating/dying?

---

## 📊 Interpretation

### **Case 1: Agent Still Works Well After Sleep**

```
Cycle 1: Good behavior ✓
Sleep 1: (no learning)
Cycle 2: STILL good behavior ✓

CONCLUSION: Sleep learning WAS the problem!
```

**What this means:**
- PPO/Replay/EWC/Scaling is breaking the policy
- Need to fix or remove sleep learning
- Agent can work fine without it

### **Case 2: Agent Still Broken After Sleep**

```
Cycle 1: Good behavior ✓
Sleep 1: (no learning)  
Cycle 2: Still rotating/dying ✗

CONCLUSION: Problem is NOT sleep learning
```

**What this means:**
- Something else happening at sleep transition
- Maybe hidden state issue persists?
- Need to investigate further

---

## 🔍 What to Look For

### **Console Output:**

```
Cycle 1:
  Tick 100: ...
  Tick 200: ...
  ...
  Tick 2000: Entering Sleep - LEARNING DISABLED
  
Cycle 2:
  Tick 2100: ...  ← Does behavior change here?
```

### **Visual Observation:**

**Before Sleep:**
- [ ] Smooth movement
- [ ] Finds resources
- [ ] Survives well

**After Sleep:**
- [ ] Movement same or broken?
- [ ] Still explores or just rotates?
- [ ] Can still find resources?

---

## 💡 Next Steps Based on Results

### **If Learning Was the Problem:**

We need to identify WHICH part:
1. Try enabling ONLY PPO (disable rest)
2. Try enabling ONLY replay (disable rest)
3. Try enabling ONLY EWC (disable rest)
4. Try enabling ONLY scaling (disable rest)

Find the culprit, then fix it.

### **If Learning Was NOT the Problem:**

Look at:
1. Hidden state (though we already preserve it)
2. Sleep/wake transition itself
3. World state changes during sleep
4. Something we're missing

---

## 🎯 Critical Question

**After running this test, does Cycle 2 behavior match Cycle 1?**

- **YES** → Sleep learning is the problem ✓
- **NO** → Problem is elsewhere

**Report back what you see!** This will tell us exactly where to look next.

---

## 🔧 Temporary Nature

**This is a DIAGNOSTIC version, not a fix!**

Without learning:
- Agent won't improve over time
- Won't consolidate memories
- Won't prevent catastrophic forgetting

But it will tell us if sleep learning is the issue.

---

**Run the test and let me know what happens!** 🧪
