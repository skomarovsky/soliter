# PROJECT SOLITER: PHASED DEVELOPMENT PLAN
## NCP/CfC Implementation with Biological Sleep Functions

---

## 🎯 **Overall Goal**

Build a biologically-realistic autonomous agent with:
- NCP/CfC neural architecture (C. elegans-inspired)
- Drive-based behavior (hunger, thirst, cold, curiosity)
- Biological sleep functions (consolidation, pruning, homeostasis)
- Continuous learning (no batch RL)
- Energy conservation and realistic foraging

---

## 📊 **Phase Structure**

Each phase follows this pattern:
1. **Implement** - Add new features
2. **Test** - Run experiments (50+ cycles)
3. **Collect Data** - Log comprehensive metrics
4. **Analyze** - Identify what works/fails
5. **Document** - Record findings
6. **Decision** - Continue or iterate

---

# PHASE 0: BASELINE (Current State)

## 🎯 **Goal:** Establish stable baseline with minimal NCP

### **Implementation Status:**
- ✅ NCP trainer created (sleep_wake_ncp.py)
- ✅ PPO completely removed
- ✅ All motor fixes in place:
  - Death penalty (25% resources)
  - Turn rate reduced (0.03)
  - Turn momentum (0.7)
  - Turn cost (energy penalty)
  - Directional stability
  - Gradient filtering
  - Hidden state preserved
  - Curiosity system (surprise-driven)

### **Current Sleep Functions:**
```python
def sleep_cycle(self):
    decay_exploration()      # ✅ Implemented
    restore_wakefulness()    # ✅ Implemented
    # NO memory consolidation
    # NO synaptic pruning
    # NO homeostatic scaling
```

### **Test Protocol:**
```bash
cd ~/wss/soliter
python scripts/train_soliter_with_viz.py --cycles 50 --fps 60 \
  --output-dir experiments/phase0_baseline
```

### **Data to Collect:**

**Agent Behavior:**
- [ ] Heading changes per cycle (should be <80° avg)
- [ ] Distance traveled per cycle
- [ ] Velocity distribution
- [ ] Death count and causes

**Resource Interaction:**
- [ ] Total consumptions (food/water/heat)
- [ ] Time to first consumption
- [ ] Spatial distribution of consumptions

**Drive System:**
- [ ] Average drive levels (hunger/thirst/cold/curiosity)
- [ ] Drive correlations with behavior
- [ ] Wasteful movement penalty frequency

**Network Stability:**
- [ ] Action std decay curve
- [ ] Hidden state magnitudes over time
- [ ] Policy consistency across cycles

### **Success Criteria:**
- ✅ Agent survives 50 cycles
- ✅ Behavior consistent (Cycle 1 ≈ Cycle 50)
- ✅ No policy breaking after sleep
- ✅ Smooth movement (<80° avg heading change)
- 🔄 Some resource consumption (>0, even if low)

### **Expected Issues:**
- ⚠️ Low resource consumption (no explicit learning)
- ⚠️ Slow improvement (pure CfC dynamics)
- ⚠️ Random exploration dominates

### **Deliverables:**
- [ ] Training log JSON
- [ ] Analysis report (heading, deaths, consumption)
- [ ] Visualization video (first 10 cycles)
- [ ] PHASE0_RESULTS.md document

### **Decision Point:**
- **If baseline stable** → Proceed to Phase 1
- **If unstable/broken** → Debug motor/sensor issues first
- **If zero consumption** → Check gradient sensing before continuing

---

# PHASE 1: MEMORY CONSOLIDATION (Hebbian Learning)

## 🎯 **Goal:** Add memory consolidation so agent learns from experience

### **Hypothesis:**
"Hebbian strengthening during sleep will allow agent to build long-term memory of successful behaviors without batch RL conflicts."

### **Implementation:**

**1. Add Experience Tracking:**
```python
class ExperienceTracker:
    def __init__(self, capacity=2000):
        self.recent_experiences = []
        self.successful_patterns = []
        
    def record(self, state, action, reward, drive_state):
        self.recent_experiences.append({
            'state': state,
            'action': action,
            'reward': reward,
            'drives': drive_state,
            'timestamp': tick
        })
        
        if reward > 0.1:  # Significant success
            self.successful_patterns.append(...)
```

**2. Implement Hebbian Consolidation:**
```python
def consolidate_memories(self):
    """
    Strengthen synapses for successful experiences.
    Hebbian rule: Δw = η * pre * post * reward
    """
    # Get successful experiences from last wake period
    successes = [e for e in recent if e['reward'] > 0.1]
    
    # For each successful experience
    for exp in successes:
        # Get network activations
        activations = self.agent.brain.forward(exp['state'])
        
        # Strengthen connections (Hebbian)
        with torch.no_grad():
            for layer in self.agent.brain.layers:
                # Hebbian: strengthen active connections
                pre = layer.input_activity
                post = layer.output_activity
                delta_w = 0.001 * pre * post * exp['reward']
                layer.weight += delta_w
```

**3. Update Sleep Cycle:**
```python
def sleep_cycle(self):
    print(f"  💤 Sleep - Memory Consolidation Active")
    
    # 1. Consolidate recent successes (NEW!)
    self.consolidate_memories()
    
    # 2. Decay exploration
    self.decay_exploration()
    
    # 3. Restore wakefulness
    self.agent.exit_sleep()
```

### **Test Protocol:**
```bash
python scripts/train_soliter_with_viz.py --cycles 100 --fps 60 \
  --output-dir experiments/phase1_hebbian \
  --log-activations  # Log network activities
```

### **Data to Collect:**

**Learning Metrics:**
- [ ] Weight changes over time (magnitude)
- [ ] Successful pattern repetition rate
- [ ] Time to resource discovery (improving?)
- [ ] Consumption rate per cycle (increasing?)

**Memory Formation:**
- [ ] Spatial memory emergence (return to known locations?)
- [ ] Pattern recognition (similar situations → similar actions?)
- [ ] Consolidation effectiveness (how many patterns strengthened?)

**Network Health:**
- [ ] Weight magnitude distribution
- [ ] Dead neurons (zero activity?)
- [ ] Runaway excitation (weights exploding?)

### **Success Criteria:**
- ✅ Agent finds resources faster in later cycles
- ✅ Returns to known resource locations
- ✅ Consumption rate increases over time
- ✅ Network weights stable (no explosion/vanishing)

### **Expected Issues:**
- ⚠️ Possible weight instability (runaway strengthening)
- ⚠️ Overfitting to early experiences
- ⚠️ Need careful learning rate tuning

### **Deliverables:**
- [ ] Training log with weight statistics
- [ ] Heatmap of spatial memory (positions visited)
- [ ] Weight change analysis
- [ ] PHASE1_RESULTS.md

### **Decision Point:**
- **If learning works** → Proceed to Phase 2
- **If unstable weights** → Add homeostatic scaling first
- **If no improvement** → Adjust Hebbian parameters or algorithm

---

# PHASE 2: HOMEOSTATIC SCALING

## 🎯 **Goal:** Add network stability through activity balancing

### **Hypothesis:**
"Homeostatic scaling during sleep will prevent runaway excitation and keep all neurons contributing, improving stability and learning."

### **Implementation:**

**1. Add Activity Tracking:**
```python
class ActivityTracker:
    def __init__(self):
        self.neuron_activities = {}
        
    def record_activities(self, layer_name, activations):
        if layer_name not in self.neuron_activities:
            self.neuron_activities[layer_name] = []
        self.neuron_activities[layer_name].append(activations.detach())
        
    def get_average_activity(self, layer_name):
        activities = torch.stack(self.neuron_activities[layer_name])
        return activities.mean(dim=0)  # Average per neuron
```

**2. Implement Homeostatic Scaling:**
```python
def homeostatic_scaling(self):
    """
    Balance neuron activities to target level.
    Prevents some neurons dominating, others dying.
    """
    target_activity = 0.5  # Target average firing rate
    scaling_rate = 0.05    # Gentle adjustment
    
    with torch.no_grad():
        for layer_name, layer in self.agent.brain.named_modules():
            if not hasattr(layer, 'weight'):
                continue
                
            # Get average activity for each neuron
            avg_activity = self.activity_tracker.get_average_activity(layer_name)
            
            # Scale weights to bring activity toward target
            for neuron_idx, activity in enumerate(avg_activity):
                if activity > target_activity * 1.5:  # Too active
                    scaling = 1.0 - scaling_rate
                    layer.weight[neuron_idx] *= scaling
                elif activity < target_activity * 0.5:  # Too quiet
                    scaling = 1.0 + scaling_rate
                    layer.weight[neuron_idx] *= scaling
        
        # Clear activity history
        self.activity_tracker.clear()
```

**3. Update Sleep Cycle:**
```python
def sleep_cycle(self):
    print(f"  💤 Sleep - Consolidation + Homeostasis")
    
    # 1. Memory consolidation (Hebbian)
    self.consolidate_memories()
    
    # 2. Homeostatic scaling (NEW!)
    self.homeostatic_scaling()
    
    # 3. Decay exploration
    self.decay_exploration()
    
    # 4. Restore wakefulness
    self.agent.exit_sleep()
```

### **Test Protocol:**
```bash
python scripts/train_soliter_with_viz.py --cycles 100 --fps 60 \
  --output-dir experiments/phase2_homeostasis \
  --log-neuron-activities
```

### **Data to Collect:**

**Network Health:**
- [ ] Neuron activity distribution (should converge to target)
- [ ] Dead neuron count (activity < 0.1)
- [ ] Overactive neuron count (activity > 0.9)
- [ ] Weight magnitude stability

**Scaling Effectiveness:**
- [ ] Activity variance before/after scaling
- [ ] Convergence to target activity
- [ ] Impact on learning (interference?)

**Behavior Quality:**
- [ ] Does homeostasis hurt or help learning?
- [ ] Resource consumption rate
- [ ] Stability over extended training (200+ cycles)

### **Success Criteria:**
- ✅ All neurons contribute (no dead neurons)
- ✅ Activities cluster near target (0.4-0.6)
- ✅ Learning still works (consumption increasing)
- ✅ Stable over 200+ cycles

### **Expected Issues:**
- ⚠️ Scaling might interfere with Hebbian learning
- ⚠️ Target activity might need tuning
- ⚠️ Oscillations (scaling up, Hebbian scales up, scaling down...)

### **Deliverables:**
- [ ] Neuron activity distributions over time
- [ ] Before/after scaling statistics
- [ ] Long-term stability analysis (200 cycles)
- [ ] PHASE2_RESULTS.md

### **Decision Point:**
- **If stable + learning** → Proceed to Phase 3
- **If interference** → Adjust scaling rate or separate timing
- **If oscillations** → Implement damping or different balance algorithm

---

# PHASE 3: SYNAPTIC PRUNING

## 🎯 **Goal:** Add efficiency through weak connection removal

### **Hypothesis:**
"Pruning weak/unused connections during sleep will improve generalization, reduce noise, and maintain network efficiency without hurting learned behaviors."

### **Implementation:**

**1. Add Connection Strength Tracking:**
```python
class SynapticTracker:
    def __init__(self):
        self.connection_usage = {}
        
    def track_usage(self, layer_name, weight_gradients):
        # Track which connections are actively used
        usage = torch.abs(weight_gradients)
        if layer_name not in self.connection_usage:
            self.connection_usage[layer_name] = usage
        else:
            # Exponential moving average
            alpha = 0.9
            self.connection_usage[layer_name] = (
                alpha * self.connection_usage[layer_name] + 
                (1 - alpha) * usage
            )
```

**2. Implement Synaptic Pruning:**
```python
def synaptic_pruning(self):
    """
    Weaken rarely-used connections.
    This removes noise and improves generalization.
    """
    pruning_threshold = 0.1  # Connections used < 10%
    pruning_amount = 0.05    # Reduce by 5%
    
    with torch.no_grad():
        for layer_name, layer in self.agent.brain.named_modules():
            if not hasattr(layer, 'weight'):
                continue
            
            # Get connection usage
            usage = self.synaptic_tracker.get_usage(layer_name)
            
            # Identify weak connections
            weak_mask = usage < pruning_threshold
            
            # Prune (weaken, don't eliminate completely)
            layer.weight[weak_mask] *= (1.0 - pruning_amount)
```

**3. Update Sleep Cycle:**
```python
def sleep_cycle(self):
    print(f"  💤 Sleep - Full Biological Suite")
    
    # 1. Memory consolidation
    self.consolidate_memories()
    
    # 2. Homeostatic scaling
    self.homeostatic_scaling()
    
    # 3. Synaptic pruning (NEW!)
    self.synaptic_pruning()
    
    # 4. Decay exploration
    self.decay_exploration()
    
    # 5. Restore wakefulness
    self.agent.exit_sleep()
```

### **Test Protocol:**
```bash
python scripts/train_soliter_with_viz.py --cycles 150 --fps 60 \
  --output-dir experiments/phase3_pruning \
  --log-connection-strengths
```

### **Data to Collect:**

**Network Efficiency:**
- [ ] Effective connection count (strong connections)
- [ ] Sparsity over time (% of weak connections)
- [ ] Network capacity (information throughput)

**Pruning Impact:**
- [ ] Does pruning hurt learned behaviors?
- [ ] Generalization improvement?
- [ ] Noise reduction (smoother actions?)

**Long-term Stability:**
- [ ] Performance over 300+ cycles
- [ ] Resource consumption stability
- [ ] No catastrophic forgetting?

### **Success Criteria:**
- ✅ Network becomes sparser over time (50%+ weak connections)
- ✅ Learned behaviors preserved
- ✅ Generalization improves (handles new situations)
- ✅ Stable over 300+ cycles

### **Expected Issues:**
- ⚠️ Might prune important connections by accident
- ⚠️ Interaction with Hebbian (Hebbian strengthens, pruning weakens)
- ⚠️ Need careful threshold tuning

### **Deliverables:**
- [ ] Connection strength heatmaps
- [ ] Sparsity evolution graphs
- [ ] Generalization test results
- [ ] PHASE3_RESULTS.md

### **Decision Point:**
- **If improved efficiency** → Proceed to Phase 4
- **If hurts learning** → Adjust pruning parameters or remove
- **If unstable** → Revisit homeostatic scaling interaction

---

# PHASE 4: CIRCADIAN RHYTHMS

## 🎯 **Goal:** Add temporal regulation through day/night cycles

### **Hypothesis:**
"Circadian rhythms will improve realism by modulating sleep timing, drive intensities, and metabolic rates based on time of day."

### **Implementation:**

**1. Create Circadian Clock:**
```python
class CircadianClock:
    def __init__(self, ticks_per_day=2000):
        self.ticks_per_day = ticks_per_day
        self.current_tick = 0
        
    def update(self, tick):
        self.current_tick = tick
        
    def get_hour(self):
        """0-23 hour of day."""
        return (self.current_tick % self.ticks_per_day) / self.ticks_per_day * 24
        
    def get_phase(self):
        """0-2π circadian phase."""
        return 2 * np.pi * (self.current_tick % self.ticks_per_day) / self.ticks_per_day
        
    def is_night(self):
        hour = self.get_hour()
        return 20 <= hour or hour < 6
        
    def get_sleep_pressure(self):
        """Sleep pressure increases during day, peaks at night."""
        phase = self.get_phase()
        return np.sin(phase - np.pi/2) * 0.5 + 0.5
        
    def get_metabolic_rate(self):
        """Metabolism higher during day."""
        phase = self.get_phase()
        return np.cos(phase) * 0.2 + 1.0  # 0.8-1.2x
```

**2. Integrate with Agent:**
```python
def update_vitals(self, velocity, ambient_temp, turn, dt):
    # Get circadian modifiers
    metabolic_rate = self.circadian.get_metabolic_rate()
    
    # Energy decay affected by circadian rhythm
    base_decay = self.config.energy_decay_base
    circadian_decay = base_decay * metabolic_rate
    
    movement_cost = velocity ** 2
    turning_cost = abs(turn) * 0.5
    energy_decay = circadian_decay * (1 + movement_cost + turning_cost)
    self.energy -= energy_decay * dt
    # ...
```

**3. Update Sleep Trigger:**
```python
def should_enter_sleep(self):
    """Agent should sleep when:
    1. Wakefulness low
    2. Circadian sleep pressure high
    3. Preferably at night
    """
    wakefulness_trigger = self.wakefulness < 0.3
    sleep_pressure = self.circadian.get_sleep_pressure()
    night_bonus = 0.2 if self.circadian.is_night() else 0
    
    total_sleep_drive = sleep_pressure + night_bonus
    
    return wakefulness_trigger and total_sleep_drive > 0.6
```

### **Test Protocol:**
```bash
python scripts/train_soliter_with_viz.py --cycles 200 --fps 60 \
  --output-dir experiments/phase4_circadian \
  --log-circadian
```

### **Data to Collect:**

**Circadian Patterns:**
- [ ] Sleep timing distribution (hour of day)
- [ ] Activity levels by time of day
- [ ] Metabolic rate effects on survival

**Behavior Modulation:**
- [ ] Performance differences day vs night
- [ ] Resource consumption patterns
- [ ] Death timing patterns

**Realism:**
- [ ] Does agent sleep at night?
- [ ] Is agent more active during day?
- [ ] Natural synchronization with world cycles?

### **Success Criteria:**
- ✅ Agent predominantly sleeps at night (>80%)
- ✅ Activity higher during day
- ✅ Circadian effects visible in data
- ✅ No negative impact on survival/learning

### **Expected Issues:**
- ⚠️ Might conflict with drive-based sleep
- ⚠️ Metabolic changes might affect balance
- ⚠️ Day/night already handled by world (heater availability)

### **Deliverables:**
- [ ] Sleep timing histogram (by hour)
- [ ] Activity level curves (24-hour)
- [ ] Circadian impact analysis
- [ ] PHASE4_RESULTS.md

### **Decision Point:**
- **If enhances realism** → Keep and proceed to Phase 5
- **If no clear benefit** → Optional feature, proceed anyway
- **If conflicts** → Adjust or make optional

---

# PHASE 5: FULL INTEGRATION & OPTIMIZATION

## 🎯 **Goal:** Optimize all systems working together

### **Implementation:**

**1. Hyperparameter Tuning:**
- Memory consolidation learning rate
- Homeostatic scaling target/rate
- Synaptic pruning threshold/amount
- Circadian effect strengths
- Drive system weights
- Directional stability parameters

**2. Performance Optimization:**
- Batch neuron activity tracking
- Efficient weight updates
- Sparse connection operations
- Vectorized computations

**3. Add Advanced Features:**
- Spatial memory visualization
- Resource preference learning
- Territory emergence detection
- Social behaviors (if multi-agent)

### **Test Protocol:**
```bash
# Long-term stability test
python scripts/train_soliter_with_viz.py --cycles 500 --fps 60 \
  --output-dir experiments/phase5_integrated

# Multi-agent test (if implemented)
python scripts/train_multi_agent.py --agents 5 --cycles 200
```

### **Data to Collect:**

**System Integration:**
- [ ] Interaction effects between sleep functions
- [ ] Emergent behaviors
- [ ] Long-term stability (500+ cycles)

**Performance Metrics:**
- [ ] Resource consumption efficiency
- [ ] Survival time distribution
- [ ] Learning curve (improvement over time)
- [ ] Energy efficiency (actions per resource)

**Scientific Analysis:**
- [ ] Comparison with biological data
- [ ] Novel behaviors discovered
- [ ] Publication-ready results

### **Success Criteria:**
- ✅ Agent survives indefinitely (no deaths after cycle 100)
- ✅ Efficient foraging (>100 consumptions per 100 cycles)
- ✅ Spatial memory evident (returns to resources)
- ✅ Biological realism validated
- ✅ Publishable results

### **Deliverables:**
- [ ] Complete training dataset (500 cycles)
- [ ] Comprehensive analysis report
- [ ] Comparison with biological organisms
- [ ] PHASE5_FINAL_RESULTS.md
- [ ] Research paper draft

---

# PHASE 6: RESEARCH EXTENSIONS (Future)

## 🎯 **Goal:** Push boundaries of bio-realistic AI

### **Possible Directions:**

**1. Spiking NCPs (sCCfC):**
- Convert to spiking neural network
- Deploy on Intel Loihi neuromorphic chip
- Measure energy efficiency vs biological

**2. Multi-Agent Society:**
- Multiple agents interacting
- Communication emergence
- Territory formation
- Cooperative behaviors

**3. Evolution:**
- Population of agents
- Genetic algorithms for architecture
- Natural selection of behaviors
- Speciation?

**4. Advanced Cognition:**
- Working memory implementation
- Planning and mental simulation
- Tool use and problem-solving
- Meta-learning

**5. Neuromorphic Deployment:**
- Port to neuromorphic hardware
- Real-time embodied robot
- Energy measurements
- Biological comparison

---

# 📊 DATA COLLECTION FRAMEWORK

## **Every Phase Must Collect:**

### **Core Metrics (Always):**
```python
metrics = {
    # Behavior
    'heading_changes': [],
    'distance_traveled': [],
    'velocity_distribution': [],
    
    # Survival
    'deaths': [],
    'life_duration': [],
    'death_causes': [],
    
    # Resources
    'consumptions': {'food': 0, 'water': 0, 'heat': 0},
    'consumption_locations': [],
    
    # Drives
    'avg_drives': {'hunger': [], 'thirst': [], 'cold': [], 'curiosity': []},
    
    # Network
    'action_std': [],
    'sleep_cycles': 0,
}
```

### **Phase-Specific Metrics:**

**Phase 1 (Hebbian):**
- Weight change magnitudes
- Successful pattern counts
- Memory formation indicators

**Phase 2 (Homeostasis):**
- Neuron activity distributions
- Dead/overactive neuron counts
- Scaling effectiveness

**Phase 3 (Pruning):**
- Connection sparsity
- Pruned connection counts
- Generalization metrics

**Phase 4 (Circadian):**
- Sleep timing by hour
- Activity by time of day
- Metabolic rate effects

---

# 🎯 SUCCESS METRICS PER PHASE

| Phase | Primary Metric | Target | Fallback |
|-------|---------------|--------|----------|
| **Phase 0** | Stability | No breaking | Debug sensors |
| **Phase 1** | Learning | Consumption > 10 | Tune Hebbian |
| **Phase 2** | Network health | No dead neurons | Adjust scaling |
| **Phase 3** | Efficiency | 50% sparsity | Optional feature |
| **Phase 4** | Realism | Sleep at night | Optional feature |
| **Phase 5** | Integration | 500 cycle stability | Publication |

---

# 📁 FILE ORGANIZATION

```
soliter-develop/
├── experiments/
│   ├── phase0_baseline/
│   │   ├── training_YYYYMMDD_HHMMSS.json
│   │   ├── analysis.ipynb
│   │   ├── video.mp4
│   │   └── PHASE0_RESULTS.md
│   ├── phase1_hebbian/
│   ├── phase2_homeostasis/
│   ├── phase3_pruning/
│   ├── phase4_circadian/
│   └── phase5_integrated/
├── docs/
│   ├── PHASED_DEVELOPMENT_PLAN.md (this file)
│   ├── NCP_TRANSITION.md
│   ├── BIOLOGICAL_SLEEP_FUNCTIONS.md
│   └── phase_reports/
│       ├── PHASE0_RESULTS.md
│       ├── PHASE1_RESULTS.md
│       └── ...
└── scripts/
    ├── train_soliter_with_viz.py
    ├── analyze_phase.py (NEW - analyze each phase)
    └── compare_phases.py (NEW - compare across phases)
```

---

# 🚀 GETTING STARTED

## **Immediate Next Steps:**

1. **Run Phase 0 Baseline:**
```bash
cd ~/wss/soliter
tar -xzf updated_files_only.tar.gz
mkdir -p experiments/phase0_baseline
python scripts/train_soliter_with_viz.py --cycles 50 --fps 60 \
  --output-dir experiments/phase0_baseline
```

2. **Analyze Results:**
```bash
python scripts/analyze_phase.py experiments/phase0_baseline/training_*.json
```

3. **Document Findings:**
Create `experiments/phase0_baseline/PHASE0_RESULTS.md`

4. **Decision:**
Based on Phase 0 results, decide whether to:
- Proceed to Phase 1 (if stable)
- Debug issues (if broken)
- Adjust parameters (if suboptimal)

---

# ✅ SUMMARY

**Phased Approach:**
- Phase 0: Baseline (pure NCP)
- Phase 1: Hebbian learning
- Phase 2: Homeostatic scaling
- Phase 3: Synaptic pruning
- Phase 4: Circadian rhythms
- Phase 5: Integration & optimization
- Phase 6: Research extensions

**Each Phase:**
1. Implement feature
2. Test (50-200 cycles)
3. Collect comprehensive data
4. Analyze results
5. Document findings
6. Make go/no-go decision

**Goal:** Build biologically-realistic agent with full sleep functions while maintaining stability and learning capability.

**This systematic approach ensures we understand each component before adding complexity!** 🧠
