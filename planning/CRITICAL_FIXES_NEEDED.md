# CRITICAL FIXES NEEDED - Visualization & Environment

## Issues Identified by User:

### 1. ✅ Day/Night Visual Feedback
**Problem:** No visual indication of day vs night
**Solution:** Change background color based on time and season

### 2. ❌ Resource Recovery Not Working  
**Problem:** Resources should recover during day+night cycle but they don't
**Solution:** Fix resource recovery logic

### 3. ❌ Resource Depletion Not Visualized
**Problem:** Can't see how depleted resources are
**Solution:** Show depletion ratio visually

### 4. ❌ Heater Day Behavior Wrong
**Problem:** Heater shows detection/consumption circles during day when inactive
**Solution:** Hide circles during day, change color to brown (obstacle only)

### 5. ❌ Analysis Tools Missing
**Problem:** Can't analyze training results
**Solution:** Create analysis script

---

## IMPLEMENTATION PLAN:

### Fix 1: Background Color (Day/Night/Season)

**Location:** `scripts/train_soliter_with_viz.py`

```python
def get_background_color(world):
    """Get background color based on time and season."""
    is_night = world.is_night()
    season = world.get_season()
    
    # Summer colors
    if season == Season.SUMMER:
        if is_night:
            return (20, 20, 60)  # Dark blue (summer night)
        else:
            return (135, 206, 235)  # Sky blue (summer day)
    
    # Winter colors
    elif season == Season.WINTER:
        if is_night:
            return (10, 10, 10)  # Black (winter night)
        else:
            return (200, 220, 240)  # Pale blue (winter day)
    
    # Spring colors
    elif season == Season.SPRING:
        if is_night:
            return (30, 30, 50)  # Dark grey-blue
        else:
            return (180, 220, 240)  # Light blue
    
    # Autumn colors
    else:  # AUTUMN
        if is_night:
            return (40, 30, 30)  # Dark brownish
        else:
            return (200, 200, 180)  # Pale yellow

# In draw() method:
background_color = get_background_color(world)
self.screen.fill(background_color)
```

### Fix 2: Resource Recovery

**Problem Location:** `soliter/environment/resources.py`

**Check current recovery logic:**
```python
def update(self, world_tick: int):
    """Update resource state (recovery)."""
    # Is recovery actually being called?
    # Is recovery rate correct?
    # Is depletion being tracked?
```

**Expected behavior:**
- After 15 uses → depleted
- Wait 300 ticks (cooldown)
- Recover over 200 ticks (gradually: 15→14→13...→0)
- Total recovery time: 500 ticks

**Fix needed:**
1. Ensure `resource.update(tick)` is called every tick
2. Verify recovery logic works
3. Add debug logging to see recovery happening

### Fix 3: Resource Depletion Visualization

**Location:** `scripts/train_soliter_with_viz.py` in `draw()` method

**Current:** Resources drawn as solid green/blue/red circles

**Needed:** Visual depletion indicator

```python
def draw_resource(resource, screen, scale):
    """Draw resource with depletion visualization."""
    pos = (int(resource.position[0] * scale), 
           int(resource.position[1] * scale))
    
    # Get depletion ratio (1.0 = full, 0.0 = empty)
    depletion = resource.get_depletion_ratio()
    
    # Color intensity based on depletion
    if resource.resource_type == 'food':
        base_color = (0, 255, 0)  # Green
    elif resource.resource_type == 'water':
        base_color = (0, 0, 255)  # Blue
    else:  # heat
        base_color = (255, 100, 0)  # Orange
    
    # Fade color when depleted
    color = tuple(int(c * depletion) for c in base_color)
    if depletion == 0:
        color = (50, 50, 50)  # Dark grey when empty
    
    # Draw detection radius (thin)
    detection_radius = int(40 * scale)
    pygame.draw.circle(screen, (200, 200, 200), pos, detection_radius, 1)
    
    # Draw consumption radius (thick, color shows depletion)
    consumption_radius = int(5 * scale)
    pygame.draw.circle(screen, color, pos, consumption_radius)
    
    # Draw depletion bar above resource
    bar_width = 20
    bar_height = 4
    bar_x = pos[0] - bar_width // 2
    bar_y = pos[1] - consumption_radius - 10
    
    # Background (grey)
    pygame.draw.rect(screen, (100, 100, 100), 
                    (bar_x, bar_y, bar_width, bar_height))
    
    # Filled portion (green = full, red = empty)
    fill_width = int(bar_width * depletion)
    fill_color = (255, 0, 0) if depletion < 0.3 else (0, 255, 0)
    pygame.draw.rect(screen, fill_color, 
                    (bar_x, bar_y, fill_width, bar_height))
```

### Fix 4: Heater Day/Night Behavior

**Location:** `scripts/train_soliter_with_viz.py` in `draw()` method

```python
def draw_heaters(heaters, world, screen, scale):
    """Draw heaters with day/night behavior."""
    is_night = world.is_night()
    
    for heater in heaters:
        pos = (int(heater.position[0] * scale), 
               int(heater.position[1] * scale))
        
        if is_night:
            # Night: Active heater (red/orange)
            depletion = heater.get_depletion_ratio()
            color_intensity = int(255 * depletion)
            color = (255, color_intensity // 2, 0)  # Orange
            
            # Draw detection radius
            detection_radius = int(40 * scale)
            pygame.draw.circle(screen, (255, 200, 200), pos, 
                             detection_radius, 1)
            
            # Draw consumption radius
            consumption_radius = int(5 * scale)
            pygame.draw.circle(screen, color, pos, consumption_radius)
        else:
            # Day: Inactive (brown obstacle, no circles)
            color = (139, 90, 43)  # Brown
            obstacle_radius = int(5 * scale)
            pygame.draw.circle(screen, color, pos, obstacle_radius)
            
            # Optional: Draw small "X" to show inactive
            pygame.draw.line(screen, (100, 60, 30), 
                           (pos[0]-3, pos[1]-3), 
                           (pos[0]+3, pos[1]+3), 2)
            pygame.draw.line(screen, (100, 60, 30), 
                           (pos[0]+3, pos[1]-3), 
                           (pos[0]-3, pos[1]+3), 2)
```

### Fix 5: Resource Recovery Check

**Location:** Main training loop

**Add this to verify recovery is working:**

```python
# In main loop, after resource updates
if world.tick % 100 == 0:  # Every 100 ticks
    for feeder in resources['feeders']:
        if feeder.current_uses > 0:
            print(f"Tick {world.tick}: Feeder uses={feeder.current_uses}/15, "
                  f"depletion={feeder.get_depletion_ratio():.2f}")
```

### Fix 6: Analysis Script

**Create:** `scripts/analyze_training.py`

```python
#!/usr/bin/env python3
"""Analyze training results from JSON log."""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_training(log_file):
    """Analyze training log and generate report."""
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    print("=" * 70)
    print("TRAINING ANALYSIS REPORT")
    print("=" * 70)
    
    # Overall stats
    print(f"\nTotal ticks: {data['total_ticks']:,}")
    print(f"Total cycles: {max(s['cycle'] for s in data['snapshots'])}")
    print(f"Total consumptions: {sum(1 for s in data['snapshots'] if s.get('consumed'))}")
    print(f"Total deaths: {len(data['deaths'])}")
    
    # Consumption analysis
    food_count = sum(1 for s in data['snapshots'] if s.get('consumed') == 'food')
    water_count = sum(1 for s in data['snapshots'] if s.get('consumed') == 'water')
    heat_count = sum(1 for s in data['snapshots'] if s.get('consumed') == 'heat')
    
    print(f"\nConsumptions by type:")
    print(f"  Food:  {food_count:4d}")
    print(f"  Water: {water_count:4d}")
    print(f"  Heat:  {heat_count:4d}")
    
    # Behavior metrics by cycle
    print(f"\nBehavior by cycle:")
    print("Cycle | Heading Δ | Distance | Consumptions")
    print("-" * 50)
    
    for cycle in range(1, 6):  # First 5 cycles
        cycle_snaps = [s for s in data['snapshots'] if s['cycle'] == cycle]
        if not cycle_snaps:
            continue
        
        # Heading changes
        headings = [s['heading'] for s in cycle_snaps]
        heading_changes = []
        for i in range(1, len(headings)):
            change = abs(headings[i] - headings[i-1])
            if change > np.pi:
                change = 2*np.pi - change
            heading_changes.append(change)
        avg_heading = np.degrees(np.mean(heading_changes)) if heading_changes else 0
        
        # Distance traveled
        positions = [(s['x'], s['y']) for s in cycle_snaps]
        total_dist = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_dist += np.sqrt(dx**2 + dy**2)
        
        # Consumptions
        consumptions = sum(1 for s in cycle_snaps if s.get('consumed'))
        
        print(f"  {cycle:2d}  | {avg_heading:7.1f}° | {total_dist:7.0f}  | {consumptions:4d}")
    
    # Death analysis
    if data['deaths']:
        print(f"\nDeath analysis:")
        causes = {}
        for death in data['deaths']:
            cause = death['cause']
            causes[cause] = causes.get(cause, 0) + 1
        
        for cause, count in causes.items():
            print(f"  {cause:15s}: {count:3d}")
    
    # Plot if matplotlib available
    try:
        plot_training_metrics(data)
        print("\n✅ Plots saved to analysis/")
    except Exception as e:
        print(f"\n⚠️  Could not create plots: {e}")

def plot_training_metrics(data):
    """Create visualization plots."""
    Path("analysis").mkdir(exist_ok=True)
    
    # Extract metrics
    ticks = [s['tick'] for s in data['snapshots'][::10]]  # Sample
    energy = [s['energy'] for s in data['snapshots'][::10]]
    hydration = [s['hydration'] for s in data['snapshots'][::10]]
    
    # Plot vitals over time
    plt.figure(figsize=(12, 6))
    plt.plot(ticks, energy, label='Energy', alpha=0.7)
    plt.plot(ticks, hydration, label='Hydration', alpha=0.7)
    plt.xlabel('Tick')
    plt.ylabel('Vital Level')
    plt.title('Agent Vitals Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis/vitals.png', dpi=150)
    plt.close()
    
    # Plot spatial distribution
    plt.figure(figsize=(10, 10))
    x = [s['x'] for s in data['snapshots'][::50]]
    y = [s['y'] for s in data['snapshots'][::50]]
    plt.scatter(x, y, alpha=0.3, s=1)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Agent Movement Pattern')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis/movement.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_training.py <log_file.json>")
        sys.exit(1)
    
    analyze_training(sys.argv[1])
```

---

## PRIORITY ORDER:

1. **FIX RESOURCE RECOVERY** (critical - affects gameplay)
2. **ADD DEPLETION VISUALIZATION** (see what's happening)
3. **ADD BACKGROUND COLORS** (day/night feedback)
4. **FIX HEATER DISPLAY** (correct day behavior)
5. **CREATE ANALYSIS SCRIPT** (understand results)

---

## NEXT STEPS:

1. I'll implement these fixes one at a time
2. Test each fix
3. Package for you to test
4. Iterate based on your feedback

**Which fix should I implement first?**
