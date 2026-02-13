# Visualization Reference Guide

## ✅ **Agent Representation - UPDATED**

### **Current (Correct):**
```python
# Agent body - OUTLINE ONLY (not filled)
pygame.draw.circle(screen, WHITE, agent_pos, agent_radius, 2)  # thickness=2
```

**Visual appearance:**
```
    ╱──→  (yellow arrow, 4px thick, 25px long)
   ○      (white circle OUTLINE, 2px thick, hollow)
```

**What you see:**
- ⭕ **Hollow white circle** (outline only, 2px thickness)
- → **Yellow arrow** (shows heading/direction, 4px thick, 25px long)
- 💀 Gray circle if dead

**NOT filled** - you can see through the agent circle!

---

## 🎨 **Complete Visualization Legend**

### **Resources:**

#### **Food (Green):**
```
  ╭─────╮  ← Thin light green (detection radius, 30 units)
  ╭───╮    ← Thick bright green (consumption radius, 5 units)
    •      ← Small green dot (resource center)
   15%     ← Orange text if < 30% capacity
```

**Colors:**
- Bright green = Full capacity
- Dark green = Depleted
- Orange % = Low capacity warning

#### **Water (Blue):**
```
  ╭─────╮  ← Thin light blue (detection radius, 25 units)
  ╭───╮    ← Thick bright blue (consumption radius, 5 units)
    •      ← Small blue dot
   10%     ← Orange text if low
```

#### **Heat (Red):**
```
  ╭─────╮  ← Thin pink (detection radius, 40 units)
  ╭───╮    ← Thick bright red (consumption radius, 8 units)
    •      ← Small red dot
   25%     ← Orange text if low
```

### **Agent:**
```
    ╱──→  Yellow arrow (heading direction)
   ○      White hollow circle (body)
```

**Circle thickness:** 2px (outline only, NOT filled)
**Arrow:** 4px thick, 25px long

---

## 📊 **UI Elements**

### **Top Status Bar:**
```
Cycle: 15  Step: 1543  Consumptions: 347
```

### **Vitals Bars (Color-coded):**
```
Energy:      [████████░░] 80   ← Green bar
Hydration:   [██████████] 95   ← Blue bar  
Temperature: [█████░░░░░] 52   ← Red bar
Wakefulness: [███████░░░] 72   ← Yellow bar
```

### **Drive States:**
```
Hunger:    [░░░░░░░░░░] 0.15  ← Low hunger (well fed)
Thirst:    [████░░░░░░] 0.42  ← Medium thirst
Cold:      [████████░░] 0.78  ← High cold (needs heat!)
Curiosity: [██░░░░░░░░] 0.23  ← Low curiosity
```

### **Reward Breakdown:**
```
Satisfaction: +2.5
Discomfort:   -1.2
Consumption:  +2.0  (if consumed this tick)
Total:        +3.3
```

---

## 🎯 **What Each Element Means**

### **Detection Radius (Thin Circle):**
- Agent can **SENSE** resource from this distance
- Gradients provide directional signal
- **Does NOT mean agent can consume**

### **Consumption Radius (Thick Circle):**
- Agent must be **INSIDE** this circle to consume
- Much smaller than detection radius
- **This is the consumption zone**

### **Resource Center Dot:**
- Exact resource location
- Always visible (5 pixels)
- Color matches resource type

### **Depletion Indicator:**
- Circle brightness = capacity level
- Bright = full, dark = depleted
- Orange % text when < 30%

---

## 🔍 **Visual Debugging**

### **Agent Not Consuming?**

Check:
1. **Is agent INSIDE thick circle?** (consumption radius)
   - If outside: Agent too far, must walk closer
   - If inside: Check capacity

2. **Is resource depleted?** (dark/dim circle)
   - If yes: Resource empty, must find another
   - Orange % shows exact capacity

3. **Is gradient pointing correctly?**
   - Should point toward bright (full) resources
   - Should ignore dark (depleted) resources

### **Agent Spinning in Place?**

Check:
1. **Is agent inside consumption circle already?**
   - Should be consuming
   - If not, resource depleted

2. **Are ALL nearby resources depleted?**
   - All circles dark/dim
   - Agent must explore further

3. **Is it winter?** (seasonal strength)
   - Resources give HALF amount
   - Agent needs 2× more visits

---

## 📈 **Seasonal Indicators**

### **Summer:**
- Resources bright (high capacity)
- Consumption efficient (15.0 per visit)
- Fewer visits needed

### **Winter:**
- Resources dimmer (still capacity, but half strength)
- Consumption inefficient (7.5 per visit)
- More visits needed
- You'll see agent visiting same resources more often

---

## 🎨 **Color Reference**

| Element | Color | Meaning |
|---------|-------|---------|
| **Agent circle** | White outline | Alive |
| **Agent circle** | Gray outline | Dead |
| **Agent arrow** | Yellow | Heading direction |
| **Food** | Green | Feeder |
| **Water** | Blue | Fountain |
| **Heat** | Red | Heater |
| **Detection** | Light/pastel | Sensing range |
| **Consumption** | Bright/vivid | Eating range |
| **Capacity warning** | Orange | < 30% capacity |

---

## ✅ **Confirmation Checklist**

When visualization loads, you should see:

- ✅ Agent is **hollow circle** (not filled)
- ✅ Agent has **yellow arrow** showing direction
- ✅ Resources have **TWO circles** (thin + thick)
- ✅ Resource centers are **small dots** (5px)
- ✅ Depleted resources show **orange %** text
- ✅ Bright colors = full, dark = depleted
- ✅ Vitals bars on right side
- ✅ Drive states below vitals
- ✅ Consumption count increases over time

---

## 🎯 **Summary**

**Agent Representation:** ✅ UPDATED
- Hollow circle outline (2px thick)
- Yellow direction arrow (4px thick, 25px long)
- NOT filled - you can see through it!

**Resource Representation:** ✅ COMPLETE
- Two circles: detection (thin) + consumption (thick)
- Color brightness = capacity level
- Orange % when low

**All visualization elements working correctly!**
