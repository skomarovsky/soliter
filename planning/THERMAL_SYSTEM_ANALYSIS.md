# THERMAL SYSTEM ANALYSIS & DESIGN PROPOSAL
## Project Soliter: Temperature Mechanics & Overheating

**Date:** 2026-02-14  
**Status:** Analysis & Proposal (NOT IMPLEMENTED)

---

## 🔍 CURRENT IMPLEMENTATION ANALYSIS

### **Temperature Update Formula (Line 246-248):**

```python
temp_diff = self.temperature - ambient_temperature
self.temperature -= 0.05 * temp_diff * dt
```

**This is exponential decay toward ambient:**
- If agent at 37°C, ambient at 50°C (summer) → temp_diff = -13°C
- Change = -0.05 × (-13) = +0.65°C per tick
- Agent heats up toward 50°C

**Problem Identified:** Movement does NOT generate heat!

---

## ❌ CURRENT PROBLEMS

### **Problem 1: No Movement Heat Generation**

**Code shows:**
```python
# Energy decay includes movement cost
movement_cost = velocity ** 2
turning_cost = abs(turn) * 0.5
energy_decay = base * (1 + movement_cost + turning_cost)

# Temperature ONLY decays toward ambient
temp_diff = self.temperature - ambient_temperature
self.temperature -= 0.05 * temp_diff  # No movement factor!
```

**Result:** Movement doesn't heat agent up!

### **Problem 2: Summer Death Mechanism Missing**

**Current death checks:**
```python
if self.temperature <= 0:
    self.cause_of_death = 'hypothermia'
```

**No check for:** `if self.temperature >= 100: death from hyperthermia`

### **Problem 3: No Cooling Mechanism**

**Only way to change temperature:**
1. Consume heater (+20°C) - but only at night
2. Passive decay toward ambient

**No active cooling** like:
- Resting (lower metabolic heat)
- Seeking shade (not in environment)
- Drinking water (could reduce temp)

### **Problem 4: Winter Too Fast, Summer No Recovery**

**Your observation is correct:**

**Winter (ambient 17°C):**
```
Agent at 37°C, ambient 17°C
temp_diff = 37 - 17 = 20°C
Change = -0.05 × 20 = -1.0°C per tick
Loses 1°C every tick! (TOO FAST)
In 20 ticks: 37 → 17°C (hypothermia risk)
```

**Summer (ambient 50°C):**
```
Agent at 37°C, ambient 50°C
temp_diff = 37 - 50 = -13°C
Change = -0.05 × (-13) = +0.65°C per tick
Gains 0.65°C per tick
In 20 ticks: 37 → 50°C (overheats)
```

**Issue:** Same decay rate (0.05) for both warming and cooling is unrealistic!

---

## 🎯 PROPOSED THERMAL SYSTEM REDESIGN

### **Design Goals:**

1. **Movement generates heat** (realistic metabolism)
2. **Summer can cause overheating** (new survival challenge)
3. **Agent can cool down** through behavior (rest, hydration)
4. **Winter hypothermia** slower (more time to find heat)
5. **Asymmetric thermal dynamics** (easier to heat than cool)

---

## 📐 PROPOSED FORMULAS

### **1. Heat Generation from Movement:**

```python
def update_temperature(self, velocity, turn, ambient_temp, dt):
    # === HEAT GENERATION ===
    # Metabolic heat from movement (biological reality)
    metabolic_base = 0.1  # Base metabolism generates heat
    movement_heat = velocity ** 2 * 0.3  # Fast movement = more heat
    rotation_heat = abs(turn) * 0.2  # Turning also generates heat
    
    total_metabolic_heat = metabolic_base + movement_heat + rotation_heat
    
    # === PASSIVE THERMAL EXCHANGE ===
    temp_diff = self.temperature - ambient_temp
    
    # Asymmetric rates (cooling harder than warming)
    if temp_diff > 0:  # Agent warmer than ambient (needs to cool)
        thermal_decay = 0.02 * temp_diff  # SLOW cooling
    else:  # Agent colder than ambient (needs to warm)
        thermal_decay = 0.08 * temp_diff  # FAST warming (dangerous in winter)
    
    # === COMBINED UPDATE ===
    temp_change = total_metabolic_heat - thermal_decay
    self.temperature += temp_change * dt
    
    # Clamp
    self.temperature = max(0.0, min(150.0, self.temperature))
```

**Result:**
- **Still agent in winter:** Cools slowly (0.02 rate), has time to find heater
- **Moving agent in winter:** Generates heat, slows cooling
- **Moving agent in summer:** Generates heat, can overheat!
- **Still agent in summer:** Warms up slowly toward ambient

---

### **2. Death from Overheating:**

```python
def _check_death(self):
    """Check for death conditions."""
    if self.energy <= 0:
        self.is_alive = False
        self.cause_of_death = 'starvation'
    
    if self.hydration <= 0:
        self.is_alive = False
        self.cause_of_death = 'dehydration'
    
    # NEW: Hypothermia (too cold)
    if self.temperature <= 15:  # Below 15°C
        self.is_alive = False
        self.cause_of_death = 'hypothermia'
    
    # NEW: Hyperthermia (too hot)
    if self.temperature >= 45:  # Above 45°C (human: ~42°C fatal)
        self.is_alive = False
        self.cause_of_death = 'hyperthermia'
```

**Temperature zones:**
```
  0°C  ────── 15°C ─────── 37°C ─────── 45°C ────── 150°C
  │           │    SAFE    │             │           │
  │      HYPOTHERMIA      IDEAL     HYPERTHERMIA    │
  Death                                            Death
```

---

### **3. Cooling Mechanisms:**

#### **Option A: Water Provides Cooling**

```python
def consume_resource(self, resource_type, amount):
    if resource_type == 'food':
        self.energy = min(100.0, self.energy + amount)
    
    elif resource_type == 'water':
        self.hydration = min(100.0, self.hydration + amount)
        # NEW: Water also provides cooling!
        cooling_effect = amount * 0.3  # 30% of hydration amount
        self.temperature = max(15.0, self.temperature - cooling_effect)
    
    elif resource_type == 'heat':
        self.temperature = min(150.0, self.temperature + amount)
```

**Biological justification:**
- Drinking water lowers body temperature (sweating, evaporative cooling)
- Gives water dual purpose: hydration + temperature regulation

#### **Option B: Resting Reduces Heat**

```python
def update_temperature(self, velocity, turn, ambient_temp, dt):
    # ...metabolic heat calculation...
    
    # Resting bonus (lower metabolism)
    if velocity < 0.1 and abs(turn) < 0.05:  # Nearly still
        resting_cooling = 0.15  # Bonus cooling when resting
        total_metabolic_heat -= resting_cooling
    
    # ...rest of update...
```

**Agent behavior:**
- In summer: Must rest periodically to avoid overheating
- Creates interesting trade-off: Move fast (overheat) vs move slow (survive)

#### **Option C: Shade Zones (Environment)**

```python
class ShadeZone:
    """Cool zones in the environment."""
    position: np.ndarray
    radius: float = 20.0
    cooling_rate: float = 0.5  # Reduces ambient temp by 10°C
    
    def get_effective_ambient(self, agent_pos, base_ambient):
        if distance(agent_pos, self.position) < self.radius:
            return base_ambient - 10.0  # Cooler in shade
        return base_ambient
```

**Agent must learn:**
- Summer → Seek shade zones
- Winter → Avoid shade zones
- Adds spatial strategy

---

## 🎭 PROPOSED AGENT BEHAVIORS

### **Summer Behavior (Overheating Risk):**

**Scenario: Summer day, ambient 50°C**

```
Naive agent:
  - Moves fast (velocity=1.0)
  - Searches for food
  - Movement heat: 1.0² × 0.3 = 0.3°C/tick
  - Ambient warming: (50-37) × 0.02 = 0.26°C/tick
  - Total: +0.56°C per tick
  - Time to death (37→45°C): 14 ticks
  → DIES FROM HYPERTHERMIA!

Smart agent (learned):
  - Moves slowly (velocity=0.3)
  - Rests frequently
  - Drinks water for cooling
  - Movement heat: 0.3² × 0.3 = 0.027°C/tick
  - Ambient warming: (50-37) × 0.02 = 0.26°C/tick
  - Resting cooling: -0.15°C/tick (when still)
  - Water cooling: -4.5°C per drink (15 units × 0.3)
  - Balances heat: Temp stays 38-42°C
  → SURVIVES!
```

**Key behaviors:**
1. **Slow movement** in summer (lower metabolic heat)
2. **Frequent resting** (cooling bonus)
3. **Prioritize water** (dual purpose: hydration + cooling)
4. **Seek shade** (if implemented)

---

### **Winter Behavior (Hypothermia Risk):**

**Scenario: Winter night, ambient 17°C**

```
Naive agent:
  - Stays still (velocity=0)
  - No movement heat: 0°C/tick
  - Ambient cooling: (37-17) × 0.08 = 1.6°C/tick
  - Time to danger (37→20°C): 10 ticks
  → HYPOTHERMIA RISK!

Smart agent (learned):
  - Seeks heaters actively
  - Gentle movement (generates small heat)
  - Movement heat: 0.2² × 0.3 = 0.012°C/tick
  - Ambient cooling: (37-17) × 0.08 = 1.6°C/tick
  - Net: -1.59°C/tick (still cooling)
  - Finds heater: +20°C
  - Temp: 37 → 35 → 33 → 55°C (heater) → safe
  → SURVIVES!
```

**Key behaviors:**
1. **Active seeking** of heaters (can't rely on resting)
2. **Gentle movement** (some heat generation)
3. **Heaters critical** (must find before temp drops below 20°C)

---

### **Seasonal Adaptation:**

```
SPRING (moderate):
  - Ambient: 30-40°C
  - Strategy: Balanced foraging
  - Moderate movement acceptable
  - Occasional water for cooling

SUMMER (hot, 40-50°C):
  - HIGH OVERHEATING RISK
  - Strategy: Conservative movement
  - Frequent water consumption
  - Rest during peak heat
  - Slow, deliberate movement

AUTUMN (cooling, 25-35°C):
  - Moderate risk
  - Strategy: Build spatial memory
  - Learn resource locations for winter
  - Stock up on resources

WINTER (cold, 15-25°C):
  - HIGH HYPOTHERMIA RISK
  - Strategy: Active heater seeking
  - Cannot rely on resting
  - Must find heat sources
  - Movement generates small heat
```

---

## ⚖️ THERMAL BALANCE EQUATIONS

### **Summer Equilibrium:**

**Agent at thermal equilibrium (temp stable):**
```
Heat generation = Heat loss
metabolic_heat = passive_cooling

(base + movement² × 0.3 + turn × 0.2) = 0.02 × (T_agent - T_ambient)

Example (ambient 50°C):
(0.1 + v² × 0.3) = 0.02 × (T - 50)

For stable T = 40°C:
0.1 + v² × 0.3 = 0.02 × (40 - 50)
0.1 + v² × 0.3 = -0.2
v² × 0.3 = -0.3
v² = -1.0  → IMPOSSIBLE!

Conclusion: Agent CANNOT maintain equilibrium below ambient!
Must use active cooling (water, resting, shade)
```

### **Critical Movement Speed (Summer):**

**Maximum safe velocity without overheating:**
```
Assume: Resting cooling available (-0.15), water every 100 ticks (-0.045/tick avg)
Ambient: 50°C, target temp: 40°C

Heat in = Heat out
(0.1 + v² × 0.3) = 0.02×(50-40) + 0.15 + 0.045
0.1 + v² × 0.3 = 0.2 + 0.195
v² × 0.3 = 0.295
v² = 0.983
v = 0.99

Maximum velocity ≈ 1.0 (full speed!)
But only if: resting frequently + drinking water regularly
```

---

## 🧮 PARAMETER RECOMMENDATIONS

### **Temperature Constants:**

```python
# Heat generation
metabolic_base = 0.1       # Always generating some heat
movement_heat_coeff = 0.3  # velocity² coefficient
rotation_heat_coeff = 0.2  # abs(turn) coefficient

# Thermal exchange
cooling_rate = 0.02  # When agent warmer than ambient (slow)
warming_rate = 0.08  # When agent colder than ambient (fast)

# Death thresholds
hypothermia_threshold = 15.0  # Too cold
hyperthermia_threshold = 45.0  # Too hot
optimal_temperature = 37.0     # Body temperature

# Cooling mechanisms
water_cooling_factor = 0.3     # Each unit of water reduces temp by 0.3°C
resting_cooling_bonus = 0.15   # Bonus cooling when velocity < 0.1
```

### **Ambient Temperature Ranges:**

```python
# Current in environment/world.py
base_temperature = 35.0
seasonal_amplitude = 10.0  # ±10°C seasonal variation
diurnal_amplitude = 5.0    # ±5°C day/night variation

Result:
Summer day:   35 + 10 + 5 = 50°C (danger!)
Summer night: 35 + 10 - 5 = 40°C (warm)
Winter day:   35 - 10 + 5 = 30°C (cool)
Winter night: 35 - 10 - 5 = 20°C (cold)
```

---

## 📊 EXPECTED LEARNING OUTCOMES

### **Early Cycles (No Learning):**
- Summer: Dies from hyperthermia (doesn't slow down)
- Winter: Dies from hypothermia (doesn't seek heat)
- Death rate: 60-80%

### **Mid Cycles (Basic Learning):**
- Summer: Learns water provides cooling, seeks water more
- Winter: Learns heaters are critical, prioritizes them
- Death rate: 30-40%

### **Late Cycles (Optimized):**
- Summer: Rests during hottest periods, drinks frequently, moves slowly
- Winter: Active heater seeking, maintains gentle movement
- Seasonal adaptation: Different strategies per season
- Death rate: <10%

---

## 🎯 IMPLEMENTATION PRIORITY

### **Phase 1: Critical Fixes**
1. ✅ Add movement heat generation
2. ✅ Add hyperthermia death (temp ≥ 45°C)
3. ✅ Asymmetric cooling/warming rates

### **Phase 2: Cooling Mechanisms**
4. ✅ Water provides cooling
5. ✅ Resting reduces metabolic heat

### **Phase 3: Advanced (Optional)**
6. ⚠️ Shade zones in environment
7. ⚠️ Hydration affects cooling rate
8. ⚠️ Energy affects metabolic heat

---

## ✅ SUMMARY

**Your observations are CORRECT:**

1. ✅ **Winter temperature depletes too fast**
   - Solution: Asymmetric rates (0.08 warming, 0.02 cooling)

2. ✅ **Summer temperature doesn't recover**
   - Solution: Add cooling mechanisms (water, resting)

3. ✅ **Movement should increase heat**
   - Solution: Add metabolic heat from movement/rotation

4. ✅ **Summer can cause overheating**
   - Solution: Add hyperthermia death, require behavioral adaptation

**Agent must learn:**
- **Summer:** Move slowly, rest often, drink water, avoid overheating
- **Winter:** Seek heaters actively, gentle movement helps
- **Seasonal adaptation:** Different strategies for different seasons

**This creates rich emergent behavior and survival challenge!**

---

**Status:** Analysis complete, awaiting approval to implement
