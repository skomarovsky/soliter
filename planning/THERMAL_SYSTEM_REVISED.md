# THERMAL SYSTEM - REVISED PROPOSAL
## Slow Temperature Dynamics with Day/Night Integration

**Date:** 2026-02-14  
**Version:** 2.0 (REVISED - Much Slower)  
**Status:** Awaiting Approval

---

## 🎯 KEY REVISIONS

1. **Temperature changes over ~100-200 ticks** (not 10-14)
2. **Day/night cycle affects ambient temperature** significantly
3. **Environment temperature varies** by season AND time of day
4. **Gradual thermal dynamics** give agent time to adapt

---

## 🌡️ AMBIENT TEMPERATURE SYSTEM

### **Formula:**

```python
def get_ambient_temperature(self) -> float:
    """Temperature varies by BOTH season AND time of day."""
    
    base_temp = 30.0  # Moderate baseline
    
    # SEASONAL component (±15°C)
    season_offset = {
        'SUMMER': +15.0,   # Hot season
        'SPRING': +5.0,    # Warm season
        'AUTUMN': -5.0,    # Cool season
        'WINTER': -15.0,   # Cold season
    }[current_season]
    
    # DIURNAL component (±10°C) - varies through day
    hour = current_hour  # 0-23
    
    # Hottest at 2pm (hour 14), coldest at 2am (hour 2)
    hour_radians = (hour - 14) / 12.0 * π
    diurnal_offset = -10.0 * cos(hour_radians)
    # At 2pm: cos(0) = 1 → offset = -10 × 1 = -10... wait, that's wrong
    
    # Better formula:
    # Peak heat at hour 14, min at hour 2
    if hour >= 2:
        hour_normalized = (hour - 2) / 24.0  # 0 at 2am, increases
    else:
        hour_normalized = (hour + 22) / 24.0
    
    diurnal_offset = 10.0 * sin(hour_normalized * 2 * π - π/2)
    # At 2am (normalized=0): sin(-π/2) = -1 → -10°C
    # At 2pm (normalized=0.5): sin(π/2) = +1 → +10°C
    
    return base_temp + season_offset + diurnal_offset
```

### **Temperature Table:**

```
         SUMMER      SPRING      AUTUMN      WINTER
Time   (base+15)   (base+5)    (base-5)   (base-15)
─────────────────────────────────────────────────────
2am    30+15-10=35  30+5-10=25  30-5-10=15  30-15-10=5   ← COLDEST
6am    30+15-5 =40  30+5-5 =30  30-5-5 =20  30-15-5 =10
10am   30+15+5 =50  30+5+5 =40  30-5+5 =30  30-15+5 =20
2pm    30+15+10=55  30+5+10=45  30-5+10=35  30-15+10=25  ← HOTTEST
6pm    30+15-5 =40  30+5-5 =30  30-5-5 =20  30-15-5 =10
10pm   30+15-8 =37  30+5-8 =27  30-5-8 =17  30-15-8 =7
─────────────────────────────────────────────────────
        ⚠️  DANGER                        DANGER  ⚠️
       Overheat                          Freezing
```

**Critical Periods:**
- **Summer 12pm-4pm:** 50-55°C (EXTREME overheat risk)
- **Winter 12am-6am:** 5-10°C (EXTREME freeze risk)
- **Spring/Autumn:** Moderate challenges

---

## 🔥 AGENT THERMAL DYNAMICS (SLOW)

### **Heat Generation from Activity:**

```python
def update_temperature(self, velocity, turn, ambient_temp, dt):
    """
    MUCH SLOWER thermal changes - hundreds of ticks to critical temp.
    """
    
    # === METABOLIC HEAT GENERATION ===
    # Base metabolism (always generating some heat)
    metabolic_base = 0.005  # °C per tick (was 0.1 - TOO FAST!)
    
    # Movement heat (quadratic with velocity)
    movement_heat = (velocity ** 2) * 0.01  # °C per tick
    
    # Rotation heat (linear with turn rate)
    rotation_heat = abs(turn) * 0.005  # °C per tick
    
    # RESTING BONUS: Lower metabolism when still
    if velocity < 0.1 and abs(turn) < 0.05:
        metabolic_base *= 0.5  # Half heat production when resting
    
    total_heat_production = metabolic_base + movement_heat + rotation_heat
    
    # === THERMAL EXCHANGE WITH ENVIRONMENT ===
    temp_diff = self.temperature - ambient_temp
    
    # ASYMMETRIC rates (cooling harder than warming)
    if temp_diff > 0:  # Agent warmer than ambient
        # Cooling rate: VERY SLOW (hard to cool down)
        thermal_exchange = 0.003 * temp_diff
    else:  # Agent colder than ambient
        # Warming rate: SLOW (but faster than cooling)
        thermal_exchange = 0.008 * temp_diff
    
    # === NET TEMPERATURE CHANGE ===
    delta_temp = total_heat_production - thermal_exchange
    self.temperature += delta_temp * dt
    
    # Clamp to physical limits
    self.temperature = max(0.0, min(150.0, self.temperature))
```

---

## ⏱️ TIME TO CRITICAL TEMPERATURE

### **Winter Night (Ambient 5°C, Agent at 37°C):**

**Scenario 1: Still Agent (velocity=0, turn=0)**
```
Heat production:
  metabolic_base = 0.005 × 0.5 = 0.0025°C/tick (resting bonus)
  movement = 0
  rotation = 0
  Total = 0.0025°C/tick

Heat loss:
  temp_diff = 37 - 5 = 32°C
  thermal_exchange = 0.008 × 32 = 0.256°C/tick (warming from cold ambient)
  
Net change = 0.0025 - 0.256 = -0.2535°C/tick

Time to danger (37°C → 20°C):  17°C / 0.2535 = 67 ticks
Time to death (37°C → 15°C):   22°C / 0.2535 = 87 ticks

CRITICAL: Agent has ~80-90 ticks to find heater!
```

**Scenario 2: Moving Agent (velocity=0.3)**
```
Heat production:
  metabolic_base = 0.005°C/tick (active)
  movement = 0.09 × 0.01 = 0.0009°C/tick
  Total = 0.0059°C/tick

Heat loss: 0.256°C/tick (same)

Net change = 0.0059 - 0.256 = -0.25°C/tick

Time to death: 22 / 0.25 = 88 ticks

Movement helps marginally but not enough!
Must find heater.
```

**Scenario 3: Found Heater (+20°C boost)**
```
37 → 36.7 → 36.5 → ... → 35 (60 ticks)
Finds heater: 35 + 20 = 55°C
Now warmer than ambient!

New scenario: 55°C agent, 5°C ambient
Heat loss = 0.003 × (55-5) = 0.15°C/tick (slow cooling)
Heat production = 0.0025°C/tick

Net = 0.0025 - 0.15 = -0.1475°C/tick
Time to reach safe 37°C: (55-37) / 0.1475 = 122 ticks

Agent stays warm for ~120 ticks before needing another heater!
```

### **Summer Day (Ambient 55°C, Agent at 37°C):**

**Scenario 1: Moving Agent (velocity=0.5)**
```
Heat production:
  metabolic_base = 0.005°C/tick
  movement = 0.25 × 0.01 = 0.0025°C/tick
  Total = 0.0075°C/tick

Heat gain from ambient:
  temp_diff = 37 - 55 = -18°C
  thermal_exchange = 0.003 × (-18) = -0.054°C/tick (warming)
  
Net change = 0.0075 - (-0.054) = +0.0615°C/tick

Time to danger (37°C → 42°C):  5°C / 0.0615 = 81 ticks
Time to death (37°C → 45°C):   8°C / 0.0615 = 130 ticks

Agent has ~120-130 ticks before overheating!
```

**Scenario 2: Resting Agent (velocity=0)**
```
Heat production:
  metabolic_base = 0.005 × 0.5 = 0.0025°C/tick (resting)
  movement = 0
  Total = 0.0025°C/tick

Heat gain: 0.054°C/tick (same)

Net change = 0.0025 + 0.054 = +0.0565°C/tick

Time to death: 8 / 0.0565 = 142 ticks

Resting extends survival by ~12 ticks!
```

**Scenario 3: Drinking Water (-2.25°C per drink)**
```
37 → 38 → 39 → 40 → 41 (65 ticks)
Drinks water: 41 - 2.25 = 38.75°C

Gained 65 ticks of cooling!

Agent can survive by drinking every ~60 ticks
Or: Rest + drink every ~70 ticks

STRATEGY: Alternate movement/resting + regular water
```

---

## 💧 COOLING MECHANISMS

### **1. Water Provides Cooling**

```python
def consume_resource(self, resource_type, amount):
    if resource_type == 'water':
        # Hydration
        self.hydration = min(100.0, self.hydration + amount)
        
        # COOLING effect (gradual, not instant)
        cooling = amount * 0.15  # 15% of water amount
        self.temperature = max(0.0, self.temperature - cooling)

# Example: 15 units of water
# Cooling: 15 × 0.15 = 2.25°C reduction
```

### **2. Resting Reduces Metabolic Heat**

```python
# Already in formula above
if velocity < 0.1 and abs(turn) < 0.05:
    metabolic_base *= 0.5  # Half heat when resting
    # Saves: 0.005 × 0.5 = 0.0025°C/tick
```

### **3. Heaters Provide Warmth**

```python
# Heater consumption (unchanged)
self.temperature = min(150.0, self.temperature + 20.0)

# Provides ~120 ticks of warmth in winter
```

---

## 💀 DEATH THRESHOLDS

```python
def _check_death(self):
    # Hypothermia (too cold)
    if self.temperature <= 15.0:
        self.is_alive = False
        self.cause_of_death = 'hypothermia'
    
    # Hyperthermia (too hot)
    if self.temperature >= 45.0:
        self.is_alive = False
        self.cause_of_death = 'hyperthermia'
    
    # Temperature zones:
    # 0-15°C:   DEATH (hypothermia)
    # 15-20°C:  DANGER (very cold)
    # 20-42°C:  SAFE (normal range)
    # 42-45°C:  DANGER (very hot)
    # 45-150°C: DEATH (hyperthermia)
```

---

## 🎭 EXPECTED BEHAVIORS

### **Summer Afternoon (55°C ambient, 2pm):**

**Naive Agent:**
```
Moves at full speed (v=1.0)
Heat production: 0.015°C/tick
Ambient warming: 0.054°C/tick
Net: +0.069°C/tick
Dies in: 116 ticks (2 minutes of movement)
```

**Smart Agent:**
```
10am-12pm: Moves slowly (v=0.3), seeks resources
12pm-3pm:  RESTS (v=0), drinks water every 70 ticks
3pm-6pm:   Resumes movement as temperature drops
Strategy: Adapts to time of day
Survives indefinitely!
```

### **Winter Night (5°C ambient, 2am):**

**Naive Agent:**
```
Doesn't seek heaters
Temp drops: -0.25°C/tick
Dies in: 87 ticks
```

**Smart Agent:**
```
Feels cold (temp dropping)
Actively seeks heaters (gentle movement helps slightly)
Finds heater every ~100 ticks
Maintains temp: 35-55°C oscillation
Survives indefinitely!
```

### **Seasonal Strategies:**

```
SPRING (moderate 25-45°C):
  - Balanced foraging
  - Normal movement speeds
  - Occasional water for safety
  
SUMMER (hot 35-55°C):
  - SLOW movement
  - REST during peak heat (12pm-3pm)
  - DRINK water every ~70 ticks
  - Activity morning/evening only
  
AUTUMN (cool 15-35°C):
  - Moderate challenge
  - Build resource memory for winter
  - Normal foraging
  
WINTER (cold 5-25°C):
  - CRITICAL heater dependence
  - Gentle movement (generates small heat)
  - Cannot rest too long
  - Must find heater every ~80 ticks
```

---

## 📊 PARAMETER SUMMARY

```python
# === THERMAL RATES (SLOW) ===
metabolic_base = 0.005        # Base heat production
movement_heat_coeff = 0.01    # velocity² coefficient
rotation_heat_coeff = 0.005   # abs(turn) coefficient
resting_factor = 0.5          # Half heat when still

cooling_rate = 0.003          # Agent warmer than ambient
warming_rate = 0.008          # Agent colder than ambient

# === COOLING MECHANISMS ===
water_cooling_factor = 0.15   # °C per water unit
# resting already in metabolic_base

# === DEATH THRESHOLDS ===
hypothermia_death = 15.0      # Too cold
hyperthermia_death = 45.0     # Too hot
optimal_temp = 37.0           # Body temperature

# === AMBIENT TEMPERATURE ===
base_temperature = 30.0
seasonal_range = 15.0         # ±15°C
diurnal_range = 10.0          # ±10°C

# Results:
# Hottest: Summer 2pm = 55°C
# Coldest: Winter 2am = 5°C
```

---

## ⚖️ THERMAL BALANCE

### **Can Agent Reach Equilibrium?**

**Winter at 5°C (still, resting):**
```
Heat in: 0.0025°C/tick
Heat out: 0.008 × (T-5)°C/tick

Equilibrium when: 0.0025 = 0.008 × (T-5)
0.3125 = T - 5
T = 5.3°C

NO! Equilibrium at 5.3°C (fatal!)
Agent CANNOT survive winter without heaters!
```

**Summer at 55°C (still, resting):**
```
Heat in: 0.0025°C/tick
Heat out: 0.003 × (T-55)°C/tick (when T>55, this is negative = warming)

When T<55:
0.0025 = 0.003 × (T-55)
0.833 = T - 55
T = 55.8°C

NO! Equilibrium at 55.8°C (fatal!)
Agent CANNOT survive summer without water!
```

**Conclusion: Agent MUST actively manage temperature!**
- Winter: Needs heaters
- Summer: Needs water + resting strategy
- Cannot passively survive - requires learning!

---

## ✅ SUMMARY

**Your requirements implemented:**

1. ✅ **Slow changes:** 80-140 ticks to critical temp (not 10-14)
2. ✅ **Day/night integration:** Temperature varies ±10°C through day
3. ✅ **Season+time ambient:** Combined effects create 5-55°C range
4. ✅ **Movement generates heat:** Gradual accumulation
5. ✅ **Cooling mechanisms:** Water + resting
6. ✅ **Death thresholds:** Hypothermia (<15°C) and Hyperthermia (>45°C)

**Agent must learn:**
- Summer: Rest during day, drink water, slow movement
- Winter: Seek heaters actively, gentle movement helps
- Time of day matters: Adapt activity to temperature
- Seasonal strategies: Different approaches per season

**Ready for approval and implementation!**

---

**Status:** Awaiting your approval to proceed with implementation
**Next Step:** Update ENVIRONMENT_DESIGN.md after approval
