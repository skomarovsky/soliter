# CURIOSITY DRIVE FIX: Maslow's Hierarchy Implementation

## 🎯 **Your Insight: Curiosity Should Depend on Survival Drives**

> "Drive resource should affect curiosity. When rest is 0, curiosity = 1. When rest growing, curiosity failing. If drive for each like 75%, curiosity close to 0."

**This is EXACTLY right!** Biological organisms follow Maslow's Hierarchy:
- **Survival first** (food, water, warmth)
- **Exploration second** (curiosity)

You can't explore when you're starving!

---

## ❌ **Old System (Wrong!)**

### **Curiosity was independent:**
```python
def _compute_curiosity(sensor_readings):
    # Based on sensory novelty
    variance = std(sensor_history)
    curiosity = 1 - novelty
    # NO CONNECTION to survival drives!
```

**Problems:**
1. Agent could be starving (hunger=1.0) but still curious (0.8)
2. Unrealistic! Real organisms focus on survival
3. Wasted exploration when urgent needs

**From training log:**
```
Tick 2500: hunger=0.057, cold=1.00, curiosity=0.86
           ↑ Freezing to death but still exploring!
```

---

## ✅ **New System (Biological!)**

### **Curiosity inversely proportional to survival:**

```python
def _compute_curiosity(hunger, thirst, cold):
    """
    Maslow's Hierarchy: Survival → Exploration
    
    When any survival need urgent → curiosity suppressed
    When all needs met → curiosity maximized
    """
    # Find most urgent survival need
    max_survival = max(hunger, thirst, cold)
    
    # Curiosity inverse to urgency
    curiosity = 1.0 - max_survival
    
    # Rapid suppression above 75%
    if max_survival > 0.75:
        excess = (max_survival - 0.75) / 0.25
        curiosity *= (1.0 - excess)
    
    return curiosity
```

---

## 📊 **Examples**

### **Scenario 1: All Needs Met**
```
Energy: 90  → hunger = 0.01
Hydration: 85 → thirst = 0.02  
Temp: 37    → cold = 0.00

max_survival = 0.02
curiosity = 1.0 - 0.02 = 0.98 ✓

Agent: "I'm satisfied, let's explore!"
```

### **Scenario 2: Mildly Hungry**
```
Energy: 60  → hunger = 0.25
Hydration: 80 → thirst = 0.00
Temp: 37    → cold = 0.00

max_survival = 0.25
curiosity = 1.0 - 0.25 = 0.75

Agent: "Slightly hungry, but can still explore"
```

### **Scenario 3: Very Hungry (>75% threshold)**
```
Energy: 40  → hunger = 0.80
Hydration: 70 → thirst = 0.01
Temp: 35    → cold = 0.02

max_survival = 0.80  (above 75% threshold!)
base_curiosity = 1.0 - 0.80 = 0.20
excess = (0.80 - 0.75) / 0.25 = 0.20
curiosity = 0.20 * (1 - 0.20) = 0.16

Agent: "Need food NOW! No time to explore!"
```

### **Scenario 4: Starving (Critical)**
```
Energy: 10  → hunger = 1.00
Hydration: 60 → thirst = 0.11
Temp: 20    → cold = 0.51

max_survival = 1.00 (critical!)
base_curiosity = 1.0 - 1.00 = 0.00
excess = (1.00 - 0.75) / 0.25 = 1.00
curiosity = 0.00 * (1 - 1.00) = 0.00

Agent: "SURVIVAL MODE! Find food or die!"
```

---

## 🧠 **Biological Parallels**

### **Maslow's Hierarchy of Needs:**

```
Level 5: Self-Actualization ← Curiosity/Exploration
         ↑
Level 4: Esteem
         ↑
Level 3: Love/Belonging
         ↑
Level 2: Safety
         ↑
Level 1: Physiological ← Food, Water, Warmth
```

**Our agent now implements this!**
- Physiological needs (hunger/thirst/cold) → Level 1
- Curiosity drive → Level 5
- Can't reach Level 5 until Level 1 satisfied!

### **Real Animals:**

| Organism | Behavior When Hungry | Behavior When Satisfied |
|----------|---------------------|------------------------|
| **Rat** | Direct path to food cache | Explores maze |
| **Bird** | Focused foraging | Play behavior |
| **Human** | "Hangry", focused on food | Creative, curious |
| **Old Agent** | Still exploring ✗ | Exploring |
| **New Agent** | Focused foraging ✓ | Exploring ✓ |

---

## 📈 **Expected Behavior Changes**

### **Old System:**
```
Tick 1000:  E=90, H=85, T=37
            hunger=0.01, thirst=0.0, cold=0.0
            curiosity=0.86 (random based on novelty)
            → Explores

Tick 2000:  E=30, H=40, T=25
            hunger=0.88, thirst=0.64, cold=0.72
            curiosity=0.91 (still high!)
            → STILL exploring while starving! ✗
```

### **New System:**
```
Tick 1000:  E=90, H=85, T=37
            hunger=0.01, thirst=0.0, cold=0.0
            curiosity=0.99 (all needs met!)
            → Explores ✓

Tick 2000:  E=30, H=40, T=25
            hunger=0.88, thirst=0.64, cold=0.72
            max_survival=0.88
            curiosity=0.05 (suppressed!)
            → FOCUSED on finding food! ✓
```

---

## 🎮 **Visualization Update**

### **Old Display (3 drives):**
```
Drive States:
  Hunger: 0.88
  Thirst: 0.64
  Cold: 0.72
```

### **New Display (4 drives with labels):**
```
Drive States:
  Hunger: 0.88    [████████████████████  ]
  Thirst: 0.64    [█████████████        ]
  Cold: 0.72      [██████████████       ]
  Curiosity: 0.05 [█                    ]
```

**Color coding:**
- Hunger: Orange
- Thirst: Blue
- Cold: Cyan
- Curiosity: Yellow ← NEW!

---

## 🎯 **Behavior Implications**

### **Early Game (Resources Abundant):**
```
Vitals high → Drives low → Curiosity HIGH
Agent explores broadly, discovers environment
```

### **Mid Game (Resources Depleting):**
```
One vital low → One drive high → Curiosity MEDIUM
Agent balances exploration with resource seeking
```

### **Late Game (Survival Critical):**
```
Multiple vitals low → Max drive high → Curiosity LOW
Agent laser-focused on survival, minimal exploration
```

### **After Sleep (Refreshed):**
```
All vitals restored → Drives reset → Curiosity HIGH
Agent explores new areas
```

---

## 🔬 **Mathematical Formula**

```
Let:
  h = hunger ∈ [0, 1]
  t = thirst ∈ [0, 1]
  c = cold ∈ [0, 1]
  
  s_max = max(h, t, c)  # Most urgent survival need

Then:
  curiosity_base = 1 - s_max
  
  If s_max > 0.75:
    excess = (s_max - 0.75) / 0.25
    curiosity = curiosity_base × (1 - excess)
  Else:
    curiosity = curiosity_base
    
  curiosity ∈ [0, 1]
```

**Graph:**
```
Curiosity vs Max Survival Drive

1.0 |     ●
    |      ＼
0.8 |        ＼
    |          ＼
0.6 |            ＼
    |              ＼
0.4 |                ＼
    |                  ＼
0.2 |                    ＼___
    |                        ●
0.0 |________________________●
    0   0.25  0.5  0.75   1.0
        Max Survival Drive

Note: Steeper drop after 0.75 (urgency threshold)
```

---

## 🧪 **Testing**

```bash
cd soliter-develop
tar -xzf updated_files_only.tar.gz
python scripts/train_soliter_with_viz.py --cycles 50 --fps 60
```

**Watch the drive bars:**

**When vitals high:**
- ✅ Hunger/Thirst/Cold bars: Small (green zone)
- ✅ Curiosity bar: LARGE (yellow)
- ✅ Agent explores broadly

**When one vital drops:**
- ✅ One survival bar grows (orange/blue/cyan)
- ✅ Curiosity bar shrinks proportionally
- ✅ Agent seeks that specific resource

**When multiple vitals critical:**
- ✅ Survival bars all large
- ✅ Curiosity bar near ZERO
- ✅ Agent desperately seeks nearest resource

**After sleep/consumption:**
- ✅ Survival bars reset (small)
- ✅ Curiosity bar jumps to maximum
- ✅ Agent explores freely again

---

## ✅ **Summary**

**Your Design:** Curiosity = 1 when all drives = 0, → 0 when any drive → 1  
**Implementation:** `curiosity = 1 - max(hunger, thirst, cold)` with 75% threshold  
**Result:** Maslow's Hierarchy in AI! ✓  

**Biological Realism:**
- Survival → Exploration hierarchy ✓
- Urgent needs suppress curiosity ✓
- Satisfied needs enable exploration ✓

**Visual Feedback:**
- 4 drive bars (was 3) ✓
- Clear labels ✓
- Color coded ✓
- Real-time relationship visible ✓

**Your intuition about drive interaction was perfect!** 🎯
