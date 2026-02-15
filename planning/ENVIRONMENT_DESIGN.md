# ENVIRONMENT DESIGN SPECIFICATION
## Project Soliter: Biological Agent Simulation Environment

**Version:** 1.0  
**Date:** 2026-02-14  
**Status:** Design Document

---

## 1. WORLD STRUCTURE

### 1.1 Physical Dimensions

**World Size:**
```
Width:  200 units
Height: 200 units
Total Area: 40,000 square units
```

**Rationale:**
- Large enough for spatial memory to matter (agent must remember locations)
- Small enough to traverse in reasonable time (~100-200 ticks at max speed)
- Allows for distinct territories and resource clustering
- Enables resource scarcity without making foraging impossible

**Coordinate System:**
```
Origin: (0, 0) at top-left
X-axis: 0 → 200 (left to right)
Y-axis: 0 → 200 (top to bottom)

   0                     200
 0 ┌───────────────────────┐
   │                       │
   │      WORLD            │
   │                       │
200└───────────────────────┘
```

### 1.2 Boundaries

**Boundary Type: Hard Walls (Clipping)**

**Current Implementation:**
```python
def move(self, velocity, turn, world_bounds, dt):
    # Move agent
    new_position = position + delta
    
    # Clip to boundaries (agent cannot leave world)
    if world_bounds:
        new_position[0] = np.clip(new_position[0], 0, world_bounds[0] - 1)
        new_position[1] = np.clip(new_position[1], 0, world_bounds[1] - 1)
    
    self.position = new_position
```

**Behavior:**
- Agent cannot cross boundaries (walls are solid)
- No wrapping (world is not a torus)
- Agent feels boundary through proximity sensors
- Can get "stuck" in corners (realistic challenge)

**Alternative Considered (Not Implemented):**
- Wrapping boundaries (torus): Rejected (unrealistic, confuses spatial memory)
- Soft boundaries (penalty): Rejected (adds complexity)
- Infinite world: Rejected (removes scarcity)

### 1.3 Obstacles

**Current Status: NO INTERNAL OBSTACLES**

**Future Consideration:**
```python
class Obstacle:
    position: np.ndarray
    radius: float
    type: str  # 'wall', 'rock', 'barrier'
```

**Rationale for Not Having Obstacles:**
- Focus on resource foraging first
- Adds complexity to pathfinding
- Can be added in Phase 6 (Research Extensions)

---

## 2. TEMPORAL CYCLES

### 2.1 Day/Night Cycle

**Cycle Length:**
```
Full Cycle: 2000 ticks
Daytime:    1000 ticks (ticks 0-999)
Nighttime:  1000 ticks (ticks 1000-1999)
```

**Current State Determination:**
```python
def is_night(self, tick):
    """Check if current time is night."""
    cycle_position = tick % 2000
    return 1000 <= cycle_position < 2000
```

**Visual Representation:**
```
Tick:  0    500   1000  1500  2000  2500
       │─────│─────│─────│─────│─────│
       └─Day─┘ Night└─Day─┘ Night└─...
```

**Effects of Day/Night:**

1. **Ambient Temperature:**
```python
def get_ambient_temperature(self, tick):
    cycle_pos = (tick % 2000) / 2000.0
    phase = cycle_pos * 2 * π
    
    # Day: 35-45°C, Night: 25-35°C
    base_temp = 35.0
    variation = 5.0 * sin(phase)
    
    return base_temp + variation
```

Temperature curve:
```
45°C ┐     ╱╲         ╱╲
40°C │    ╱  ╲       ╱  ╲
35°C │───╱────╲─────╱────╲───
30°C │        ╲   ╱      ╲
25°C └─────────╲─╱────────╲─╱
     0   500  1000 1500 2000
     Day      Night    Day
```

2. **Heater Availability:**
```python
# Heaters ONLY work at night (biological: campfire)
if heater.is_night:
    can_consume = True
else:
    can_consume = False  # Heaters inactive during day
```

**Biological Rationale:**
- **Day:** Warm, no need for external heat
- **Night:** Cold, agent must find heaters to avoid hypothermia
- Mimics circadian rhythms and thermoregulation challenges

### 2.2 Seasonal Cycle

**Cycle Length:**
```
Full Year: 8000 ticks (4 day/night cycles)
Each Season: 2000 ticks (1 day/night cycle)

Spring: ticks 0-1999
Summer: ticks 2000-3999
Autumn: ticks 4000-5999
Winter: ticks 6000-7999
```

**Season Determination:**
```python
def get_current_season(self, tick):
    season_tick = tick % 8000
    
    if season_tick < 2000:
        return 'SPRING'
    elif season_tick < 4000:
        return 'SUMMER'
    elif season_tick < 6000:
        return 'AUTUMN'
    else:
        return 'WINTER'
```

**Seasonal Effects:**

1. **Resource Availability Strength:**
```python
SEASONAL_STRENGTH = {
    'SPRING': 1.0,   # 100% availability
    'SUMMER': 1.0,   # 100% availability
    'AUTUMN': 0.75,  # 75% availability
    'WINTER': 0.5,   # 50% availability
}

def get_availability_strength(self, season):
    return SEASONAL_STRENGTH[season]
```

**Resource Yield by Season:**
```
Food/Water yield:
Spring: 15 × 1.0 = 15 units
Summer: 15 × 1.0 = 15 units
Autumn: 15 × 0.75 = 11.25 units
Winter: 15 × 0.5 = 7.5 units  (SCARCITY!)
```

2. **Ambient Temperature Modulation:**
```python
SEASONAL_TEMP_OFFSET = {
    'SPRING': 0,      # Normal
    'SUMMER': +5,     # Hotter
    'AUTUMN': -3,     # Cooler
    'WINTER': -8,     # Much colder
}
```

**Combined Day/Night + Seasonal Temperature:**
```
Summer Day:   45 + 5 = 50°C (hot!)
Summer Night: 25 + 5 = 30°C
Winter Day:   45 - 8 = 37°C (body temp)
Winter Night: 25 - 8 = 17°C (hypothermia risk!)
```

**Biological Rationale:**
- **Spring/Summer:** Abundant resources, easier survival
- **Autumn:** Preparation period (resources declining)
- **Winter:** Scarcity forces efficient foraging, risk of starvation
- Mimics natural seasonal challenges faced by organisms

---

## 3. RESOURCES

### 3.1 Resource Types

**Three Essential Resources:**

1. **Food (Feeders)**
   - Restores: Energy
   - Biological: Sustenance, prevents starvation
   - Color: Green

2. **Water (Waterers)**
   - Restores: Hydration
   - Biological: Prevents dehydration
   - Color: Blue

3. **Heat (Heaters)**
   - Restores: Temperature
   - Biological: Thermoregulation, prevents hypothermia
   - Color: Red/Orange

### 3.2 Resource Placement

**Spatial Distribution:**

**Design Philosophy:**
Resources should be placed to:
1. Create distinct "patches" or "territories"
2. Require spatial memory (remembering locations matters)
3. Force decision-making (which patch to visit?)
4. Enable emergent behavior (territory formation, circuits)
5. Provide both clustered and dispersed options

**Placement Strategies:**

#### **Strategy 1: Triangular Cluster Layout (RECOMMENDED)**

```python
# Three main resource clusters forming a triangle
# Each cluster contains all three resource types

CLUSTER_1 = {
    'center': (50, 50),
    'feeders': [(50, 50), (55, 48)],
    'waterers': [(48, 52), (52, 54)],
    'heaters': [(50, 56), (46, 50)],
}

CLUSTER_2 = {
    'center': (150, 50),
    'feeders': [(150, 50), (145, 52)],
    'waterers': [(152, 48), (148, 54)],
    'heaters': [(150, 46), (154, 51)],
}

CLUSTER_3 = {
    'center': (100, 150),
    'feeders': [(100, 150), (102, 148)],
    'waterers': [(98, 152), (104, 151)],
    'heaters': [(100, 154), (96, 149)],
}
```

**Visual Layout:**
```
  0                          200
0 ┌────────────────────────────┐
  │                            │
  │   CLUSTER_1    CLUSTER_2   │
  │     ▲              ▲       │
50│    F W H          F W H    │
  │                            │
  │                            │
  │         CLUSTER_3          │
  │            ▲               │
150          F W H             │
  │                            │
200└────────────────────────────┘

Distance between clusters: ~100 units
Travel time: ~50-70 ticks at normal speed
```

**Advantages:**
- Clear spatial structure (triangle is easy to learn)
- Each cluster is self-sufficient (all resource types)
- Balanced distribution (no cluster dominates)
- Forces circuit behavior (1→2→3→1)
- Large enough to require memory

#### **Strategy 2: Grid Layout**

```python
# Resources in a 2×2 grid pattern
# More structured, easier to learn

GRID_POSITIONS = {
    'NW': (60, 60),   # Northwest
    'NE': (140, 60),  # Northeast  
    'SW': (60, 140),  # Southwest
    'SE': (140, 140), # Southeast
}

# Each grid point has mixed resources
feeders = [(60, 60), (140, 60), (60, 140), (140, 140)]
waterers = [(65, 65), (135, 65), (65, 135), (135, 135)]
heaters = [(55, 55), (145, 55), (55, 145), (145, 145)]
```

**Visual:**
```
     60        140
60   ■────────■
     │        │
     │        │
140  ■────────■
```

**Advantages:**
- Very structured (easy spatial memory)
- Symmetric (no bias)
- Good for testing learning

**Disadvantages:**
- Too predictable (less challenge)
- Less naturalistic

#### **Strategy 3: Random Clustered (Current Default)**

```python
def generate_random_clustered_resources(num_clusters=3, cluster_radius=20):
    """Generate resources with random cluster centers."""
    
    # Generate cluster centers (avoid edges)
    clusters = []
    for i in range(num_clusters):
        center_x = random.randint(50, 150)
        center_y = random.randint(50, 150)
        
        # Ensure minimum distance from other clusters
        while any(distance((center_x, center_y), c) < 80 for c in clusters):
            center_x = random.randint(50, 150)
            center_y = random.randint(50, 150)
        
        clusters.append((center_x, center_y))
    
    # Place resources around each cluster
    resources = {'feeders': [], 'waterers': [], 'heaters': []}
    for cx, cy in clusters:
        # 1-2 of each type per cluster
        for _ in range(random.randint(1, 2)):
            offset_x = random.randint(-cluster_radius, cluster_radius)
            offset_y = random.randint(-cluster_radius, cluster_radius)
            
            resources['feeders'].append((cx + offset_x, cy + offset_y))
            # Similar for waterers and heaters...
    
    return resources
```

**Advantages:**
- Different every run (tests generalization)
- More naturalistic (like real ecosystems)
- Prevents overfitting to specific layout

**Disadvantages:**
- Harder to debug (can't reproduce exact layout)
- Might generate poor layouts by chance

#### **Strategy 4: Ecologically-Inspired**

```python
# Mimic real ecology: 
# - Food near water (riparian zones)
# - Heat sources isolated (camp locations)

WATER_SOURCES = [(50, 100), (150, 100), (100, 50)]  # "Rivers"

FOOD_NEAR_WATER = [
    (55, 105),  # Near water source 1
    (145, 95),  # Near water source 2
    (105, 55),  # Near water source 3
]

HEAT_SOURCES = [
    (70, 70),   # Isolated camp 1
    (130, 130), # Isolated camp 2
]
```

**Rationale:**
- Food grows near water (realistic)
- Heat sources are deliberate (campsites)
- Agent must visit different zones for different needs
- More complex decision-making

### **Recommended Configuration:**

**For Phase 0-3 (Learning & Testing):**
```python
# Use Strategy 1: Triangular Cluster
# Consistent, learnable, good for testing

resources = create_triangular_clusters(
    cluster_1_center=(50, 50),
    cluster_2_center=(150, 50),
    cluster_3_center=(100, 150),
    resources_per_cluster={'feeders': 2, 'waterers': 2, 'heaters': 2}
)

Total: 6 feeders, 6 waterers, 6 heaters (18 resources)
```

**For Phase 4+ (Challenge & Realism):**
```python
# Use Strategy 3 or 4: Random or Ecological
# Tests generalization, more realistic

resources = generate_random_clustered_resources(
    num_clusters=3,
    cluster_radius=20,
    min_cluster_distance=80
)
```

### **Resource Spacing Rules:**

**Within Cluster:**
```python
# Resources of same type: 10-20 units apart
# (Close enough to be "same patch" but not overlapping)
min_same_type_distance = 10
max_same_type_distance = 20

# Resources of different types: 5-15 units apart  
# (Mixed patch - can get multiple needs met)
min_mixed_distance = 5
max_mixed_distance = 15
```

**Between Clusters:**
```python
# Cluster centers: >80 units apart
# (Forces travel, spatial memory valuable)
min_cluster_distance = 80
max_cluster_distance = 120

# Edge clearance: >30 units from boundaries
# (Prevents resources in corners/edges)
min_edge_distance = 30
```

**Validation Rules:**
```python
def validate_resource_placement(resources, world_size=(200, 200)):
    """Ensure resource placement meets design criteria."""
    
    # Rule 1: No resources too close to edges
    for r in all_resources:
        assert 30 < r.x < 170
        assert 30 < r.y < 170
    
    # Rule 2: Resources not too dense
    for r1 in all_resources:
        nearby = [r2 for r2 in all_resources 
                  if r1 != r2 and distance(r1, r2) < 10]
        assert len(nearby) < 3  # Max 2 other resources very close
    
    # Rule 3: World coverage adequate
    coverage = calculate_coverage(resources, detection_radius=40)
    assert 0.4 < coverage < 0.7  # 40-70% of world reachable
    
    # Rule 4: Balanced distribution
    for resource_type in ['feeders', 'waterers', 'heaters']:
        positions = [r.position for r in resources[resource_type]]
        center_of_mass = np.mean(positions, axis=0)
        # Center of mass should be near world center (100, 100)
        assert distance(center_of_mass, (100, 100)) < 50
```

### **Actual Placement Example (Strategy 1):**

```python
# RECOMMENDED INITIAL CONFIGURATION
# Used for Phase 0-3 testing

def create_default_resource_layout():
    """Create the standard test layout."""
    
    resources = {
        'feeders': [
            Feeder(position=np.array([50.0, 50.0])),
            Feeder(position=np.array([55.0, 48.0])),
            Feeder(position=np.array([150.0, 50.0])),
            Feeder(position=np.array([145.0, 52.0])),
            Feeder(position=np.array([100.0, 150.0])),
            Feeder(position=np.array([102.0, 148.0])),
        ],
        'waterers': [
            Waterer(position=np.array([48.0, 52.0])),
            Waterer(position=np.array([52.0, 54.0])),
            Waterer(position=np.array([152.0, 48.0])),
            Waterer(position=np.array([148.0, 54.0])),
            Waterer(position=np.array([98.0, 152.0])),
            Waterer(position=np.array([104.0, 151.0])),
        ],
        'heaters': [
            Heater(position=np.array([50.0, 56.0])),
            Heater(position=np.array([46.0, 50.0])),
            Heater(position=np.array([150.0, 46.0])),
            Heater(position=np.array([154.0, 51.0])),
            Heater(position=np.array([100.0, 154.0])),
            Heater(position=np.array([96.0, 149.0])),
        ],
    }
    
    return resources

# This creates 3 clusters in triangle formation:
# - Cluster 1 (NW): ~(50, 50)  - 2 of each type
# - Cluster 2 (NE): ~(150, 50) - 2 of each type
# - Cluster 3 (S):  ~(100, 150) - 2 of each type
# Total: 18 resources
```

**Expected Agent Behavior with This Layout:**

```
Early exploration (Cycles 1-10):
- Random wandering
- Discovers Cluster 1 first (if spawns near NW)
- Gradually finds other clusters
- Death if doesn't explore enough

Learning phase (Cycles 10-50):
- Builds spatial memory of 3 clusters
- Begins circuit: 1→2→3→1
- Learns to return when depleted
- Reduced deaths

Optimized phase (Cycles 50+):
- Efficient circuit traversal
- Anticipates depletion
- Minimal wasted movement
- Survives indefinitely
```

### 3.3 Resource Detection Radii

**Two Radii Per Resource:**

1. **Detection Radius (Gradient Sensing):**
```python
detection_radius = 40 units  # Can sense from this distance
```

2. **Consumption Radius (Physical Access):**
```python
consumption_radius = 5 units  # Must be this close to consume
```

**Sensing Behavior:**
```python
def compute_gradient(agent_pos, resource_pos, detection_radius):
    distance = norm(agent_pos - resource_pos)
    
    if distance > detection_radius:
        return [0, 0]  # No signal (too far)
    
    # Within range: gradient points toward resource
    direction = (resource_pos - agent_pos) / distance
    strength = 1.0 - (distance / detection_radius)
    
    return direction * strength
```

**Gradient Strength Curve:**
```
Strength
1.0 ┐
    │ ████████
0.8 │         ████
    │             ████
0.6 │                 ████
    │                     ████
0.4 │                         ████
    │                             ████
0.2 │                                 ████
    │                                     ████
0.0 └─────────────────────────────────────────
    0    5   10   15   20   25   30   35   40
    │←consumption→│←─────detection─────→│
         radius              radius
```

**Biological Rationale:**
- **Detection radius:** Like smell/sight range (long-distance sensing)
- **Consumption radius:** Must be physically close to eat/drink
- **Gap between radii:** Forces navigation (can't consume from afar)

### 3.4 Resource Depletion & Recovery

**Depletion Model:**

**Current Implementation:**
```python
class Resource:
    max_uses: int = 15           # Total uses before empty
    current_uses: int = 0        # Uses so far
    yield_amount: float = 15.0   # Amount given per use
    
    cooldown_period: int = 300   # Ticks before recovery starts
    recovery_rate: float = 0.05  # Recovery per tick (5%)
    recovery_duration: int = 200 # Ticks to fully recover
```

**Consumption Sequence:**
```
Tick 0: Resource full (uses=0/15)
        Agent consumes → +15 energy
        uses=1/15

Tick 50: Agent returns, consumes again → +15 energy
         uses=2/15

... (agent consumes 13 more times)

Tick 800: Agent consumes 15th time → +15 energy
          uses=15/15 → DEPLETED!
          
Tick 801-1100: Cooldown (300 ticks)
               Resource unavailable
               
Tick 1101-1300: Recovery (200 ticks)
                Uses decrease: 15→14→13...→0
                Rate: 0.05 per tick
                
Tick 1301: Resource full again (uses=0/15)
```

**Depletion Ratio:**
```python
def get_depletion_ratio(self):
    """0.0 = empty, 1.0 = full"""
    return 1.0 - (self.current_uses / self.max_uses)
```

**Visual Representation:**
```
Full    ████████████████  uses=0  (100%)
        ████████████      uses=3  (80%)
        ████████          uses=6  (60%)
        ████              uses=9  (40%)
Empty   ░░░░              uses=15 (0%)
        
Cooldown period: 300 ticks (no recovery yet)
Recovery period: 200 ticks (gradual refill)
```

**Seasonal Modification:**
```python
def consume(self, tick, season):
    if not self.can_consume():
        return 0
    
    # Base yield
    amount = self.yield_amount
    
    # Apply seasonal modifier
    seasonal_strength = SEASONAL_STRENGTH[season]
    amount *= seasonal_strength
    
    self.current_uses += 1
    return amount
```

**Example:**
```
Spring: 15 × 1.0 = 15 units
Winter: 15 × 0.5 = 7.5 units (half yield!)
```

**Recovery Timing:**
```python
def update(self, tick):
    if self.is_depleted():
        # Cooldown phase
        if tick - self.last_consumption_tick < self.cooldown_period:
            return  # Still cooling down
        
        # Recovery phase
        recovery_progress = (tick - self.last_consumption_tick - self.cooldown_period)
        if recovery_progress < self.recovery_duration:
            # Gradual recovery
            recovery_fraction = recovery_progress / self.recovery_duration
            self.current_uses = self.max_uses * (1.0 - recovery_fraction)
        else:
            # Fully recovered
            self.current_uses = 0
```

**Biological Rationale:**
- **Depletion:** Resources aren't infinite (realism)
- **Cooldown:** Prevents immediate re-harvesting (like plant growth delay)
- **Slow recovery:** Forces agent to find other resources (spatial distribution matters)
- **Seasonal variation:** Winter scarcity creates survival pressure

---

## 4. SCARCITY MECHANISMS

**Seven Sources of Scarcity:**

### 4.1 Resource Depletion
```
After 15 uses → Empty for 500 ticks (cooldown + recovery)
Forces agent to find alternative resources
```

### 4.2 Slow Recovery
```
Recovery rate: 0.05 per tick
Full recovery: 200 ticks
Agent must manage multiple resource locations
```

### 4.3 Long Cooldown
```
Cooldown: 300 ticks before recovery even starts
Resource unavailable for extended period
```

### 4.4 Small Consumption Radius
```
Must be within 5 units to consume
Can detect from 40 units but must navigate precisely
```

### 4.5 Multiple Drives
```
Must balance: Energy, Hydration, Temperature
Can't focus on just one resource type
```

### 4.6 Gradient Filtering
```
Only sense AVAILABLE resources
Depleted resources produce no gradient
Must find new sources when current depletes
```

### 4.7 Seasonal Strength
```
Winter: 50% yield (7.5 instead of 15)
Must consume twice as often to maintain vitals
```

**Combined Effect:**
```
Total resource "pressure":
- 15 uses × 15 yield = 225 energy per resource
- Recovery time: 500 ticks
- Number of feeders: 3
- Effective supply rate: 225 × 3 / 500 = 1.35 energy/tick

Agent energy decay: ~0.15 energy/tick (base + movement)
Ratio: Supply/Demand ≈ 1.35/0.15 ≈ 9:1

This means: Sufficient resources but requires active foraging!
```

---

## 5. WORLD BEHAVIOR PATTERNS

### 5.1 Component Interactions

**How Components Work Together:**

#### **Temperature System:**

```python
# Agent temperature affected by multiple factors

def update_temperature(agent, world, dt):
    # 1. Ambient temperature (environment)
    ambient = world.get_ambient_temperature()
    
    # 2. Metabolism (agent's internal heat)
    metabolic_heat = agent.energy * 0.01  # Higher energy = warmer
    
    # 3. Movement (generates heat)
    movement_heat = agent.velocity ** 2 * 0.5
    
    # 4. Temperature gradient toward ambient
    temp_diff = ambient - agent.temperature
    temp_change = temp_diff * 0.1  # 10% adjustment per tick
    
    # 5. Apply changes
    agent.temperature += (temp_change + movement_heat + metabolic_heat) * dt
    
    # 6. Decay (heat loss)
    decay_rate = 0.05
    agent.temperature -= decay_rate * dt
    
    # Result: Temperature oscillates around ambient
    # Unless agent seeks heaters (night) or stays still (day)
```

**Temperature Behavior Example:**
```
Scenario: Agent active during cold winter night

Tick 0:
  Ambient: 17°C (winter night)
  Agent temp: 37°C (body temp)
  Movement: 0.6 velocity → +0.18°C heat
  Gradient: (17-37) * 0.1 = -2.0°C
  Decay: -0.05°C
  Result: 37 - 2.0 + 0.18 - 0.05 = 35.13°C

Tick 100 (without heater):
  Agent temp: ~25°C (hypothermia approaching!)
  
Tick 100 (with heater):
  Heater adds: +20°C
  Agent temp: 35°C + 20°C = 55°C (safe!)
```

**Critical Points:**
- **Body temp:** 37°C (ideal)
- **Hypothermia threshold:** <20°C (cold drive activates)
- **Death threshold:** 0°C
- **Heater boost:** +20°C per consumption

#### **Energy System:**

```python
def update_energy(agent, velocity, dt):
    # Base metabolic rate
    base_decay = 0.008  # Passive energy loss
    
    # Movement cost (quadratic - faster = much more expensive)
    movement_cost = velocity ** 2 * 0.5
    
    # Turning cost (NEW - prevents thrashing)
    turning_cost = abs(turn) * 0.5
    
    # Total decay
    total_decay = base_decay * (1 + movement_cost + turning_cost)
    
    agent.energy -= total_decay * dt
```

**Energy Budget Example:**
```
Still agent:
  Decay: 0.008 per tick
  Lifetime: 100 / 0.008 = 12,500 ticks (can survive long time)

Moving agent (v=0.5):
  Movement: 0.5² = 0.25
  Decay: 0.008 * (1 + 0.25) = 0.01 per tick
  Lifetime: 100 / 0.01 = 10,000 ticks

Fast moving (v=1.0):
  Movement: 1.0² = 1.0
  Decay: 0.008 * (1 + 1.0) = 0.016 per tick
  Lifetime: 100 / 0.016 = 6,250 ticks (50% reduction!)
  
Constantly turning (turn=0.5):
  Turning: 0.5 * 0.5 = 0.25
  Total: 0.008 * (1 + 0 + 0.25) = 0.01 per tick
```

**Energy Conservation Implications:**
- **Resting** is beneficial (minimal decay)
- **Fast movement** expensive (quadratic cost)
- **Thrashing/turning** wasteful (linear penalty)
- Must balance: Speed vs efficiency

#### **Hydration System:**

```python
def update_hydration(agent, dt):
    # Base decay
    base_decay = 0.008
    
    # Temperature modifier (hotter = more thirst)
    if agent.temperature > 40:  # Above normal
        temp_factor = 1.5  # 50% faster dehydration
    else:
        temp_factor = 1.0
    
    # Movement modifier (exertion = more thirst)
    movement_factor = 1.0 + (agent.velocity * 0.3)
    
    total_decay = base_decay * temp_factor * movement_factor
    agent.hydration -= total_decay * dt
```

**Hydration Interactions:**
```
Cool & still:   0.008 × 1.0 × 1.0 = 0.008/tick
Hot & still:    0.008 × 1.5 × 1.0 = 0.012/tick (50% faster!)
Cool & moving:  0.008 × 1.0 × 1.15 = 0.0092/tick
Hot & moving:   0.008 × 1.5 × 1.15 = 0.0138/tick (72% faster!)
```

**Implications:**
- Summer days (hot) → More water needed
- Active foraging → More water needed
- Combined effects multiply → Critical in summer

### 5.2 Resource Dependency Chains

**Sequential Needs:**

```
Morning (Day start):
  Energy low → Seeks food
  Finds feeder → Eats → Energy restored
  Movement to food → Hydration decreased
  Now thirsty → Seeks water
  Finds waterer → Drinks → Hydration restored
  Temperature fine (daytime warmth)
  
Evening (Day end):
  All vitals moderate
  Gets sleepy (wakefulness decay)
  Enters sleep → Wakefulness restores
  
Night:
  Temperature dropping (cold ambient)
  Seeks heater → Warms up
  Energy/hydration still okay
  
Next morning:
  Cycle repeats
```

**Failed Chain Example:**
```
Agent finds food → Eats
Energy good, but dehydrated from traveling
No water nearby (all waterers depleted)
Must travel far to next waterer
Uses more energy traveling → Energy drops again
Finally finds water → Drinks
But now needs food again!
Stuck in depletion chase → Eventually dies
```

**Optimal Strategy:**
```
Visit resource clusters (have all types)
Consume all three: Food + Water + Heat
All needs met at once
Minimizes travel
Sustainable foraging
```

### 5.3 Emergent Spatial Patterns

**Expected Behaviors:**

#### **Pattern 1: Home Range Formation**

```
Cycles 1-20:
  Agent explores randomly
  Discovers Cluster 1 (e.g., NW patch)
  Survives well by staying near Cluster 1
  
Cycles 20-50:
  Cluster 1 resources deplete
  Forced to expand range
  Discovers Cluster 2 (NE patch)
  Now uses both clusters
  
Cycles 50+:
  All clusters known
  Has "home range" covering all patches
  Rotates between patches as they deplete/recover
  
Result: Territory emerges naturally!
```

**Home range size:**
```
Minimal: Single cluster (~40×40 area = 1,600 sq units)
Optimal: 2-3 clusters (~120×100 area = 12,000 sq units)
Full world: All resources (200×200 = 40,000 sq units)
```

#### **Pattern 2: Circuit Behavior**

```
With 3 clusters in triangle:

Stage 1: Discovery
  Random walk → Find Cluster 1
  
Stage 2: Depletion Response  
  Cluster 1 depletes → Wander → Find Cluster 2
  
Stage 3: Two-Cluster Circuit
  Alternate: 1 → 2 → 1 → 2
  (Cluster 3 still unknown)
  
Stage 4: Full Circuit
  Eventually discovers Cluster 3
  Circuit: 1 → 2 → 3 → 1 → 2 → 3...
  
Optimal: Agent learns recovery timing
  Visit order: 1 → 2 → 3 → (1 recovered) → 1
  Perfect synchronization with recovery!
```

**Circuit metrics:**
```
Distance per circuit: ~300 units (100 between each cluster)
Time per circuit: ~150 ticks (at v=0.5)
Resources per circuit: 3 clusters × 3 types × 2 resources = 18 consumptions
Energy cost: 150 ticks × 0.01 = 1.5 energy
Energy gained: 18 × 15 = 270 energy
Net gain: 270 - 1.5 = 268.5 (highly efficient!)
```

#### **Pattern 3: Seasonal Adaptation**

```
Spring/Summer (abundant):
  Can afford to explore
  Visits all clusters
  Tries new areas
  Builds complete spatial map
  
Autumn (declining):
  Focuses on reliable clusters
  Less exploration
  More conservative behavior
  
Winter (scarcity):
  Minimal movement
  Stays near best cluster
  Only travels when forced
  Survival mode
  
Result: Behavioral flexibility emerges!
```

### 5.4 Death Scenarios & Failure Modes

**Common Death Patterns:**

#### **Death Type 1: Starvation in Empty World**

```
Cause: Never finds any resources
Pattern:
  - Random wandering
  - No gradient following
  - Circles in empty spaces
  - Energy depletes to zero
  
Death location: Anywhere (no pattern)
Death time: ~1000-2000 ticks

Diagnosis: Gradient sensing broken OR policy not learning
```

#### **Death Type 2: Depletion Chase**

```
Cause: Depletes resources faster than they recover
Pattern:
  - Finds Cluster 1 → Depletes it
  - Finds Cluster 2 → Depletes it
  - Finds Cluster 3 → Depletes it
  - Returns to Cluster 1 → Still empty!
  - Wanders → Dies of starvation
  
Death location: Near depleted clusters
Death time: ~3000-5000 ticks

Diagnosis: Circuit too fast OR too few resources
```

#### **Death Type 3: Corner Trap**

```
Cause: Gets stuck in corner due to boundary + poor navigation
Pattern:
  - Moves toward corner (random or gradient)
  - Hits boundary
  - Tries to continue → Blocked
  - Rotates randomly → Still in corner
  - Resources nearby but can't path to them
  
Death location: (0-30, 0-30) or similar corner
Death time: Variable

Diagnosis: Boundary handling broken OR directional stability too rigid
```

#### **Death Type 4: Hypothermia (Winter Night)**

```
Cause: Doesn't use heaters during cold periods
Pattern:
  - Winter arrives → Temperature drops
  - Night comes → Gets colder (17°C ambient)
  - Agent doesn't seek heaters
  - Temperature drops: 37 → 30 → 20 → 10 → 0
  - Dies of hypothermia
  
Death location: Anywhere away from heaters
Death time: ~100-200 ticks into winter night

Diagnosis: Not following heat gradients OR heater detection broken
```

#### **Death Type 5: Dehydration During Summer**

```
Cause: Hot weather + movement = extreme thirst
Pattern:
  - Summer day → 50°C ambient
  - Agent moving actively
  - Hydration decay: 0.008 × 1.5 × 1.15 = 0.0138/tick
  - Doesn't prioritize water
  - Dies of dehydration while energy/temp fine
  
Death location: Between resources
Death time: Summer cycles

Diagnosis: Drive system not prioritizing thirst OR water gradients weak
```

### 5.5 Optimal Foraging Theory Applied

**Marginal Value Theorem:**

```
Biological theory: Leave patch when:
  Benefit rate < Travel cost + Alternative patch benefit

For Soliter:
  Stay at Cluster 1 if:
    (Resource yield / Time) > (Travel to Cluster 2 + Cluster 2 yield)
  
  Leave Cluster 1 when:
    Resources depleted OR
    Better opportunity elsewhere
```

**Application:**

```python
# Optimal decision-making

def should_leave_patch(current_cluster, other_clusters):
    # Current patch depleted?
    if current_cluster.all_depleted():
        return True  # Must leave
    
    # Current patch yield rate
    current_yield = current_cluster.available_resources * 15
    
    # Best alternative
    best_alternative = max(other_clusters, 
                          key=lambda c: c.available_resources)
    
    travel_cost = distance(current_cluster, best_alternative) * 0.01
    alternative_yield = best_alternative.available_resources * 15
    
    # Leave if alternative is better
    return alternative_yield - travel_cost > current_yield
```

**Expected emergence:**
```
Agent learns to:
1. Exploit current patch fully
2. Leave before complete depletion (travel while resources remain)
3. Time visits to coincide with recovery
4. Minimize travel between patches
5. Adjust strategy by season (winter: stay longer)
```

### 5.6 Environmental Pressure Gradient

**Survival Difficulty by Configuration:**

```
EASY MODE:
  Resources: 20 (many)
  Recovery: Fast (100 ticks)
  Winter strength: 0.75 (mild)
  World size: 150×150 (small)
  Result: Agent survives easily, learns slowly
  
NORMAL MODE (Current):
  Resources: 18 (adequate)
  Recovery: Slow (500 ticks)
  Winter strength: 0.50 (harsh)
  World size: 200×200 (moderate)
  Result: Survival requires learning
  
HARD MODE:
  Resources: 12 (scarce)
  Recovery: Very slow (800 ticks)
  Winter strength: 0.25 (severe)
  World size: 250×250 (large)
  Result: Only efficient agents survive
  
EXTREME MODE:
  Resources: 6 (minimal)
  Recovery: Extremely slow (1200 ticks)
  Winter strength: 0.10 (extreme)
  World size: 300×300 (huge)
  + Obstacles (rocks blocking paths)
  Result: Research-level challenge
```

---

## 6. CONFIGURATION PARAMETERS

### 6.1 World Configuration

```python
@dataclass
class WorldConfig:
    # Dimensions
    width: int = 200
    height: int = 200
    
    # Temporal cycles
    day_night_cycle_length: int = 2000
    seasonal_cycle_length: int = 8000
    
    # Temperature
    base_temperature: float = 35.0
    day_night_variation: float = 10.0
    seasonal_variation: Dict[str, float] = field(default_factory=lambda: {
        'SPRING': 0.0,
        'SUMMER': 5.0,
        'AUTUMN': -3.0,
        'WINTER': -8.0,
    })
    
    # Boundaries
    boundary_type: str = 'hard'  # 'hard', 'soft', 'wrap'
```

### 6.2 Resource Configuration

```python
@dataclass
class ResourceConfig:
    # Detection
    detection_radius: float = 40.0
    consumption_radius: float = 5.0
    
    # Depletion
    max_uses: int = 15
    yield_amount: float = 15.0
    
    # Recovery
    cooldown_period: int = 300
    recovery_rate: float = 0.05
    recovery_duration: int = 200
    
    # Seasonal
    seasonal_strength: Dict[str, float] = field(default_factory=lambda: {
        'SPRING': 1.0,
        'SUMMER': 1.0,
        'AUTUMN': 0.75,
        'WINTER': 0.5,
    })
```

### 6.3 Tunable Parameters

**For Difficulty Adjustment:**

**Make Easier:**
```python
# More resources
num_feeders = 5  # instead of 3

# Faster recovery
recovery_duration = 100  # instead of 200

# Larger consumption radius
consumption_radius = 10  # instead of 5

# Less seasonal impact
winter_strength = 0.75  # instead of 0.5
```

**Make Harder:**
```python
# Fewer resources
num_feeders = 2

# Slower recovery
recovery_duration = 400

# Smaller detection radius
detection_radius = 30  # instead of 40

# Harsher winter
winter_strength = 0.25  # only 25% yield!
```

---

## 7. IMPLEMENTATION STATUS

### 7.1 Currently Implemented ✅

**World Structure:**
- [x] World size (200×200) - `world.py`
- [x] Hard boundary walls (clipping) - `soliter_agent.py:move()`
- [x] Coordinate system (0,0 = top-left)

**Temporal Cycles:**
- [x] Day/night cycle (2000 ticks) - `world.py:is_night()`
- [x] Seasonal cycle (8000 ticks) - `world.py:get_current_season()`
- [x] Temperature variation (day/night) - `world.py:get_ambient_temperature()`
- [x] Seasonal temperature offsets - `SEASONAL_TEMP_OFFSET`

**Resources:**
- [x] Three resource types (food, water, heat) - `resources.py`
- [x] Dual detection radii (40 units detection, 5 units consumption)
- [x] Resource depletion (15 uses max) - `Resource.consume()`
- [x] Slow recovery (300 cooldown + 200 recovery) - `Resource.update()`
- [x] Seasonal strength modifiers - `SEASONAL_STRENGTH`
- [x] Heaters night-only availability - `Heater.is_agent_in_consumption_range()`

**Agent Vitals:**
- [x] Energy decay with movement cost - `soliter_agent.py:update_vitals()`
- [x] Hydration decay - `soliter_agent.py:update_vitals()`
- [x] Temperature regulation - `soliter_agent.py:update_vitals()`
- [x] Wakefulness decay - `soliter_agent.py:update_vitals()`

**Sensing:**
- [x] Proximity sensors (8 angles, 41 channels) - `sensors.py`
- [x] Gradient sensors (6 channels: food_x, food_y, water_x, water_y, heat_x, heat_y) - `gradient_sensors.py`
- [x] Gradient filtering (only available resources) - `GradientSensors.compute_gradients()`
- [x] Drive state inputs (4 channels) - `drive_system.py`

### 7.2 Implementation Files

**Core Environment:**
```
soliter/environment/
├── world.py              # World, temporal cycles, temperature
├── resources.py          # Resource classes (Feeder, Waterer, Heater)
├── sensors.py            # Proximity sensing
└── gradient_sensors.py   # Gradient sensing (directional)
```

**Key Methods:**

**World (`world.py`):**
```python
class World:
    def is_night(self, tick: int) -> bool:
        """Returns True if tick is during night (1000-1999 of 2000)"""
        
    def get_current_season(self, tick: int) -> str:
        """Returns 'SPRING', 'SUMMER', 'AUTUMN', or 'WINTER'"""
        
    def get_ambient_temperature(self) -> float:
        """Returns current ambient temp (day/night + seasonal)"""
        
    def get_seasonal_period(self) -> str:
        """Returns current season for resource calculations"""
```

**Resources (`resources.py`):**
```python
class Resource:
    def can_consume(self) -> bool:
        """Returns True if resource available (not depleted)"""
        
    def consume(self, tick: int, season: str) -> float:
        """Consume resource, returns yield (seasonal adjusted)"""
        
    def update(self, tick: int):
        """Update recovery state (called every tick)"""
        
    def get_depletion_ratio(self) -> float:
        """Returns 0.0-1.0 (0=empty, 1=full)"""
        
    def is_agent_in_consumption_range(self, agent_pos) -> bool:
        """Returns True if agent close enough (5 units)"""

class Feeder(Resource):
    resource_type = 'food'
    yield_amount = 15.0
    
class Waterer(Resource):
    resource_type = 'water'
    yield_amount = 15.0
    
class Heater(Resource):
    resource_type = 'heat'
    yield_amount = 20.0  # Higher boost for temperature
    
    def is_agent_in_consumption_range(self, agent_pos, is_night) -> bool:
        """Only available at night!"""
        if not is_night:
            return False
        return super().is_agent_in_consumption_range(agent_pos)
```

**Gradient Sensors (`gradient_sensors.py`):**
```python
class GradientSensors:
    def compute_gradients(
        self,
        agent_position: np.ndarray,
        feeders: List[Feeder],
        waterers: List[Waterer],
        heaters: List[Heater],
        is_night: bool,
        seasonal_period: str,
    ) -> np.ndarray:
        """
        Compute gradient vectors for all resource types.
        
        Returns: [food_x, food_y, water_x, water_y, heat_x, heat_y]
        Shape: (6,)
        
        Only senses AVAILABLE resources (not depleted)!
        """
```

**Agent Vitals (`soliter_agent.py`):**
```python
class SoliterAgent:
    def update_vitals(
        self,
        velocity: float,
        ambient_temp: float,
        turn: float = 0.0,
        dt: float = 1.0
    ):
        """Update all vital stats."""
        
        # Energy
        base_decay = 0.008
        movement_cost = velocity ** 2
        turning_cost = abs(turn) * 0.5
        energy_decay = base_decay * (1 + movement_cost + turning_cost)
        self.energy -= energy_decay * dt
        
        # Hydration
        hydration_decay = 0.008
        self.hydration -= hydration_decay * dt
        
        # Temperature
        temp_diff = ambient_temp - self.temperature
        temp_change = temp_diff * 0.1
        self.temperature += temp_change * dt
        self.temperature -= 0.05 * dt  # Heat loss
        
        # Wakefulness
        self.wakefulness -= 0.0001 * dt
        
        # Check death
        if self.energy <= 0:
            self.cause_of_death = 'starvation'
            self.is_alive = False
        if self.hydration <= 0:
            self.cause_of_death = 'dehydration'
            self.is_alive = False
        if self.temperature <= 0:
            self.cause_of_death = 'hypothermia'
            self.is_alive = False
    
    def consume_resource(self, resource_type: str, amount: float):
        """Consume resource and restore vitals."""
        if resource_type == 'food':
            self.energy = min(100, self.energy + amount)
        elif resource_type == 'water':
            self.hydration = min(100, self.hydration + amount)
        elif resource_type == 'heat':
            self.temperature = min(50, self.temperature + amount)
```

### 7.3 Configuration Example

**Complete World Setup:**
```python
from soliter.environment import World, WorldConfig
from soliter.environment.resources import Feeder, Waterer, Heater
import numpy as np

# Create world
world_config = WorldConfig(
    width=200,
    height=200,
    day_night_cycle_length=2000,
    seasonal_cycle_length=8000,
)
world = World(world_config)

# Create resources (Strategy 1: Triangular Clusters)
resources = {
    'feeders': [
        # Cluster 1 (NW)
        Feeder(position=np.array([50.0, 50.0])),
        Feeder(position=np.array([55.0, 48.0])),
        # Cluster 2 (NE)
        Feeder(position=np.array([150.0, 50.0])),
        Feeder(position=np.array([145.0, 52.0])),
        # Cluster 3 (S)
        Feeder(position=np.array([100.0, 150.0])),
        Feeder(position=np.array([102.0, 148.0])),
    ],
    'waterers': [
        # Cluster 1
        Waterer(position=np.array([48.0, 52.0])),
        Waterer(position=np.array([52.0, 54.0])),
        # Cluster 2
        Waterer(position=np.array([152.0, 48.0])),
        Waterer(position=np.array([148.0, 54.0])),
        # Cluster 3
        Waterer(position=np.array([98.0, 152.0])),
        Waterer(position=np.array([104.0, 151.0])),
    ],
    'heaters': [
        # Cluster 1
        Heater(position=np.array([50.0, 56.0])),
        Heater(position=np.array([46.0, 50.0])),
        # Cluster 2
        Heater(position=np.array([150.0, 46.0])),
        Heater(position=np.array([154.0, 51.0])),
        # Cluster 3
        Heater(position=np.array([100.0, 154.0])),
        Heater(position=np.array([96.0, 149.0])),
    ],
}

# Simulation loop
for tick in range(10000):
    # Update world
    world.tick = tick
    is_night = world.is_night()
    season = world.get_current_season()
    ambient_temp = world.get_ambient_temperature()
    
    # Update resources
    for resource_list in resources.values():
        for resource in resource_list:
            resource.update(tick)
    
    # Agent acts (simplified)
    # ... agent.move(), agent.update_vitals(), etc.
```

### 7.4 Not Implemented ❌

**Missing Features:**

- [ ] Internal obstacles/walls in world
- [ ] Dynamic resource spawning (resources appear/disappear)
- [ ] Resource migration (resources that move)
- [ ] Predators/threats
- [ ] Multi-agent interactions
- [ ] Weather events (storms, droughts)
- [ ] Hazardous zones (toxic areas)
- [ ] Different terrain types (slowing movement)
- [ ] Visual line-of-sight blocking
- [ ] Day/night affecting visibility
- [ ] Resource growth (biomass accumulation)
- [ ] Seasonal migration of resources

### 7.5 Future Enhancements 🔮

**Phase 6+ Research Extensions:**

#### **Enhancement 1: Dynamic Obstacles**
```python
class Obstacle:
    position: np.ndarray
    radius: float
    type: str  # 'wall', 'rock', 'barrier'
    
    def blocks_path(self, start, end) -> bool:
        """Check if obstacle blocks path between two points."""
        
# Add to world:
obstacles = [
    Obstacle(position=np.array([100, 100]), radius=10, type='rock'),
    # ... more obstacles
]

# Agent must path around obstacles
```

#### **Enhancement 2: Moving Resources (Prey)**
```python
class MovingFeeder(Feeder):
    velocity: float = 0.3
    direction: float  # radians
    
    def update(self, tick):
        # Resources move!
        self.position += self.velocity * np.array([
            np.cos(self.direction),
            np.sin(self.direction)
        ])
        
        # Random direction changes
        if random.random() < 0.01:
            self.direction += random.uniform(-π/4, π/4)
        
        super().update(tick)

# Agent must track moving resources (harder!)
```

#### **Enhancement 3: Multi-Agent Competition**
```python
# Multiple agents in same world
agents = [SoliterAgent() for _ in range(5)]

# Resources depleted by ALL agents
for agent in agents:
    if agent_near_resource:
        resource.consume()  # First come, first served!

# Emergent behaviors:
# - Competition for resources
# - Territory formation
# - Avoidance or cooperation
```

#### **Enhancement 4: Weather Events**
```python
class WeatherSystem:
    def get_current_weather(self, tick) -> str:
        # Random weather events
        if random.random() < 0.01:
            return 'STORM'  # Reduces visibility, increases costs
        return 'CLEAR'
    
    def apply_weather_effects(self, agent, weather):
        if weather == 'STORM':
            # Movement harder during storm
            agent.movement_cost_multiplier = 2.0
            # Gradient sensing reduced
            agent.detection_radius *= 0.5
```

---

## 8. BIOLOGICAL REALISM CHECKLIST

**Environment mimics real ecology:**

- [x] Finite resources (depletion)
- [x] Resource regeneration (recovery)
- [x] Spatial distribution (patchy)
- [x] Temporal variation (day/night, seasons)
- [x] Thermoregulation challenges (temperature)
- [x] Multiple needs (food, water, warmth)
- [x] Scarcity pressure (winter, depletion)
- [x] Territory formation opportunity (patch locations)
- [x] Energy conservation incentive (movement cost)

**Not realistic (acceptable simplifications):**

- [ ] No weather variation within day/season
- [ ] Resources don't migrate or reproduce
- [ ] No predators or threats
- [ ] Instant resource consumption
- [ ] No tool use or nesting

---

## 9. TESTING & VALIDATION

### 9.1 Environment Tests

**Unit Tests:**
```python
def test_day_night_cycle():
    assert world.is_night(500) == False   # Day
    assert world.is_night(1500) == True   # Night
    assert world.is_night(2500) == False  # Day (next cycle)

def test_seasonal_cycle():
    assert world.get_season(500) == 'SPRING'
    assert world.get_season(2500) == 'SUMMER'
    assert world.get_season(6500) == 'WINTER'

def test_resource_depletion():
    resource = Feeder()
    for i in range(15):
        amount = resource.consume(i, 'SPRING')
        assert amount == 15.0
    
    # 16th consumption should fail
    assert resource.can_consume() == False
```

### 9.2 Integration Tests

**Resource Distribution:**
```python
def test_resource_coverage():
    """Ensure resources cover the world adequately."""
    resources = create_default_resources()
    
    # Calculate coverage
    covered_points = 0
    for x in range(0, 200, 10):
        for y in range(0, 200, 10):
            if any(distance((x,y), r.pos) < 40 for r in resources):
                covered_points += 1
    
    coverage = covered_points / (20 * 20)
    assert coverage > 0.5  # At least 50% coverage
```

**Scarcity Validation:**
```python
def test_scarcity_balance():
    """Ensure scarcity is present but not impossible."""
    # Calculate total resource capacity
    total_feeders = 3
    uses_per_feeder = 15
    yield_per_use = 15
    total_energy = total_feeders * uses_per_feeder * yield_per_use
    # = 3 × 15 × 15 = 675 energy total
    
    # Agent needs
    agent_decay = 0.15  # per tick
    ticks_until_recovery = 500
    total_needed = agent_decay * ticks_until_recovery
    # = 0.15 × 500 = 75 energy
    
    # Safety margin
    assert total_energy / total_needed > 5  # 5× buffer
```

---

## 10. SUMMARY

**Environment Design Goals:**

1. **Biological Realism:** Mimics natural scarcity and resource dynamics
2. **Learning Pressure:** Forces agent to develop spatial memory
3. **Temporal Dynamics:** Day/night and seasonal variation
4. **Difficulty Balance:** Challenging but not impossible
5. **Emergent Behavior:** Territory, foraging patterns emerge naturally

**Key Design Decisions:**

- World: 200×200 (large enough for memory, small enough to traverse)
- Resources: Clustered patches (like real ecosystems)
- Scarcity: 7 mechanisms (depletion, recovery, seasons, etc.)
- Boundaries: Hard walls (no wrapping, realistic)
- Cycles: 2000 tick day/night, 8000 tick seasons

**Next Steps:**

1. Validate current implementation matches this spec
2. Run baseline tests (Phase 0)
3. Tune parameters based on agent performance
4. Consider enhancements (Phase 6)

---

**Document Status:** DRAFT v1.0  
**Next Review:** After Phase 0 baseline testing  
**Maintainer:** Project Soliter Team

---

## 11. QUICK REFERENCE

### Environment Parameters at a Glance

```yaml
WORLD:
  Size: 200 × 200 units
  Boundaries: Hard walls (clipping)
  Area: 40,000 square units

TEMPORAL:
  Day/Night Cycle: 2000 ticks (1000 day, 1000 night)
  Seasonal Cycle: 8000 ticks (2000 per season)
  Seasons: Spring (100%), Summer (100%), Autumn (75%), Winter (50%)

TEMPERATURE:
  Day Range: 35-45°C (base + variation)
  Night Range: 25-35°C
  Seasonal Offsets: Spring (0), Summer (+5), Autumn (-3), Winter (-8)
  Critical Points:
    - Body temp: 37°C (ideal)
    - Cold threshold: <20°C (drive activates)
    - Death: 0°C

RESOURCES:
  Types: Food, Water, Heat
  Count: 18 total (6 of each type)
  Layout: 3 clusters (triangular formation)
  Detection Radius: 40 units
  Consumption Radius: 5 units
  Max Uses: 15 per resource
  Yield: 15 units (food/water), 20 units (heat)
  Cooldown: 300 ticks
  Recovery: 200 ticks (rate: 0.05/tick)
  Total Recovery Time: 500 ticks

AGENT VITALS:
  Energy Decay: 0.008 base + movement² + turning×0.5
  Hydration Decay: 0.008 base
  Temperature: Gradient toward ambient (10% per tick)
  Wakefulness Decay: 0.0001 per tick

SCARCITY MECHANISMS:
  1. Depletion (15 uses → empty)
  2. Slow recovery (500 ticks total)
  3. Seasonal reduction (winter 50%)
  4. Small consumption radius (5 units)
  5. Multiple needs (3 vitals)
  6. Gradient filtering (only available)
  7. Movement costs (quadratic energy)
```

### Resource Cluster Positions

```
Cluster 1 (Northwest):
  Center: (50, 50)
  Food: (50, 50), (55, 48)
  Water: (48, 52), (52, 54)
  Heat: (50, 56), (46, 50)

Cluster 2 (Northeast):
  Center: (150, 50)
  Food: (150, 50), (145, 52)
  Water: (152, 48), (148, 54)
  Heat: (150, 46), (154, 51)

Cluster 3 (South):
  Center: (100, 150)
  Food: (100, 150), (102, 148)
  Water: (98, 152), (104, 151)
  Heat: (100, 154), (96, 149)
```

### Key Behavioral Expectations

```
EARLY CYCLES (1-20):
  - Random exploration
  - Discovery of first cluster
  - High death rate (50%+)
  - Sporadic resource consumption

MID CYCLES (20-100):
  - Circuit formation (2-3 clusters)
  - Spatial memory evident
  - Lower death rate (10-20%)
  - Regular consumption patterns

LATE CYCLES (100+):
  - Optimized foraging circuit
  - Anticipates depletion/recovery
  - Minimal deaths (<5%)
  - Efficient movement (minimal waste)
  - Seasonal adaptation (winter conservation)
```

### Success Metrics

```
BASELINE (Phase 0):
  - Consumptions/cycle: >0 (finds resources)
  - Deaths/1000 ticks: <1 (survives)
  - Heading stability: <100° avg change
  - Behavior consistency: Cycle 1 ≈ Cycle 50

LEARNING (Phase 1-3):
  - Consumptions/cycle: >10 (regular foraging)
  - Returns to locations: Yes (spatial memory)
  - Circuit formation: Visits 2-3 clusters
  - Deaths/1000 ticks: <0.1 (rare deaths)

OPTIMIZED (Phase 4-5):
  - Consumptions/cycle: >20 (efficient)
  - Path efficiency: <400 units/cycle
  - Resource utilization: All clusters used
  - Seasonal adaptation: Different winter strategy
  - Deaths: 0 (indefinite survival)
```

---

## 12. VALIDATION CHECKLIST

Before considering environment design complete:

**Functional Requirements:**
- [ ] World bounds enforced (agent cannot leave 200×200)
- [ ] Day/night cycles correctly (1000 ticks each)
- [ ] Seasons cycle correctly (2000 ticks each)
- [ ] Temperature varies with time and season
- [ ] Resources deplete after 15 uses
- [ ] Resources recover after 500 ticks
- [ ] Seasonal strength applied (winter 50%)
- [ ] Heaters only work at night
- [ ] Gradients only from available resources
- [ ] Agent vitals decay correctly
- [ ] Resource consumption restores vitals
- [ ] Death occurs at 0 energy/hydration/temperature

**Balance Requirements:**
- [ ] Agent can survive >10 cycles with reasonable policy
- [ ] Agent cannot survive indefinitely without learning
- [ ] Resource scarcity present (must actively forage)
- [ ] Winter noticeably harder than summer
- [ ] Spatial memory provides advantage
- [ ] Death causes are distributed (not all one type)

**Technical Requirements:**
- [ ] No runtime errors in 100 cycle run
- [ ] Performance: >30 FPS in visualization
- [ ] Memory usage stable (no leaks)
- [ ] Deterministic with fixed seed
- [ ] Save/load state works correctly

**Design Requirements:**
- [ ] Environment promotes learning
- [ ] Multiple valid strategies exist
- [ ] Emergent behaviors possible
- [ ] Biologically plausible
- [ ] Well-documented
- [ ] Parameters tunable

---

## DOCUMENT STATUS

**Version:** 1.0 COMPLETE  
**Date:** 2026-02-14  
**Status:** ✅ Ready for Review  
**Next Steps:**
1. Review with team
2. Validate against implementation
3. Run Phase 0 tests
4. Adjust parameters based on results
5. Update document with findings

**Changelog:**
- v1.0 (2026-02-14): Initial complete specification
  - Added detailed resource placement strategies
  - Added component interaction details
  - Added behavioral patterns and failure modes
  - Added implementation guide with code
  - Added quick reference and validation checklist

**Maintainer:** Project Soliter Team  
**Location:** `/planning/ENVIRONMENT_DESIGN.md`

---

**END OF DOCUMENT**
