# Live Web Dashboard Guide 🌐

## ✅ **Implementation Complete**

You now have:
1. ✅ **Live monitoring** (separate process)
2. ✅ **Web dashboard** (browser-based)
3. ✅ **Configurable refresh** (1-60 seconds)
4. ✅ **Integrated viz** (kept for debugging)

---

## 🚀 **Quick Start**

### Terminal 1: Start Training
```bash
cd soliter-develop
python scripts/train_soliter.py --cycles 1000 --output-dir experiments/long_run
```

### Terminal 2: Start Web Dashboard
```bash
python scripts/monitor_training_live.py experiments/long_run --refresh 5 --port 8050
```

### Browser: Open Dashboard
```
http://localhost:8050
```

**That's it!** The dashboard auto-refreshes and shows live training progress.

---

## 🎛️ **Dashboard Features**

### Real-Time Metrics (Top Cards)
- **Current Cycle** - Progress indicator
- **Total Consumptions** - Cumulative learning events
- **Recent Consumption Rate** - Performance trend with % change
- **Avg Life Duration** - Survival improvement with % change
- **Buffer Size** - Memory system health
- **Exploration (std)** - Exploration vs exploitation

### Interactive Charts
1. **Consumption Rate Over Time**
   - Raw data (light blue, transparent)
   - Smoothed trend (dark blue, bold)
   - Shows learning progress

2. **Life Duration Trend**
   - Scatter plot of all deaths
   - Trend line with slope
   - Shows survival improvement

3. **Memory Buffer Size**
   - Buffer growth over time
   - Should stabilize (no unbounded growth)

4. **Policy Loss (Learning Curve)**
   - Log scale for better visibility
   - Should decrease then plateau

### Dashboard Controls
- **Refresh Rate Dropdown** - Change from 1s to 60s
- **Refresh Now Button** - Force immediate update
- **Pause/Resume Button** - Stop/start auto-refresh
- **Status Indicator** - RUNNING (green, pulsing) / COMPLETE (blue) / ERROR (red)

### Info Panel (Bottom)
- Training progress (X/Y cycles, %)
- Elapsed time
- Estimated time remaining (ETA)
- Total deaths count
- Total consumptions

---

## 🔧 **Command Line Options**

### Basic Usage
```bash
python scripts/monitor_training_live.py <log_dir> [options]
```

### Options

**`--refresh N`** (default: 5)
- Set default refresh rate in seconds
- Range: 1-60 seconds
- Can be changed in browser dropdown

**`--port N`** (default: 8050)
- Web server port
- Use different port if 8050 busy

**`--host ADDRESS`** (default: 127.0.0.1)
- Server address
- Use `0.0.0.0` for remote access
- Use `127.0.0.1` for local only

### Examples

**Fast monitoring (1 second refresh)**
```bash
python scripts/monitor_training_live.py experiments/run1 --refresh 1
```

**Slow monitoring (30 second refresh, saves CPU)**
```bash
python scripts/monitor_training_live.py experiments/run1 --refresh 30
```

**Remote access (view from another computer)**
```bash
python scripts/monitor_training_live.py experiments/run1 --host 0.0.0.0 --port 8050
# Then access from another computer: http://YOUR_IP:8050
```

**Different port (if 8050 busy)**
```bash
python scripts/monitor_training_live.py experiments/run1 --port 9000
# Access: http://localhost:9000
```

---

## 🌐 **Remote Monitoring (SSH/Server)**

### Scenario: Training on remote server, monitor from laptop

**On Server:**
```bash
# Start training
python scripts/train_soliter.py --cycles 1000 --output-dir experiments/remote_run

# Start monitor with remote access
python scripts/monitor_training_live.py experiments/remote_run --host 0.0.0.0 --port 8050
```

**On Laptop:**
```
http://YOUR_SERVER_IP:8050
```

**Alternative (SSH Tunnel):**
```bash
# On laptop, create SSH tunnel
ssh -L 8050:localhost:8050 user@server

# On server
python scripts/monitor_training_live.py experiments/remote_run

# On laptop browser
http://localhost:8050
```

---

## 🐛 **Debugging Workflow**

### Quick Visual Debugging (20 cycles)
```bash
# Use integrated visualization for behavior debugging
python scripts/train_soliter_with_viz.py --cycles 20 --fps 60
```

**Use this when:**
- Checking if resources deplete correctly
- Verifying agent movement behavior
- Debugging spatial issues
- Testing new features

### Long-Term Monitoring (1000+ cycles)
```bash
# Terminal 1: Training
python scripts/train_soliter.py --cycles 1000 --output-dir experiments/long_test

# Terminal 2: Web dashboard
python scripts/monitor_training_live.py experiments/long_test --refresh 10
```

**Use this when:**
- Running long experiments
- Testing memory consolidation
- Validating Fisher saturation
- Checking for catastrophic forgetting

---

## 📊 **What to Look For**

### Healthy Training Patterns

**Consumption Rate:**
```
Early cycles (1-100):    ↗↗↗ Steep upward trend
Middle cycles (100-400): ↗   Moderate increase
Late cycles (400+):      →   Plateau (saturation)
```

**Life Duration:**
```
Should show upward trend with increasing average
Scatter increases (some long lives, some short)
Trend line slope should be positive
```

**Buffer Size:**
```
Grows initially: 3000 → 5000
Stabilizes:      5000-5500 (equilibrium)
Doesn't explode: Should NOT keep growing unbounded
```

**Policy Loss:**
```
Starts high:     0.5-5.0 (active learning)
Decreases:       Downward trend
Plateaus:        0.1-1.0 (consolidated)
```

### Warning Signs

**⚠️ Consumption rate declining** → Possible catastrophic forgetting
**⚠️ Buffer size exploding** → Memory system failing
**⚠️ Policy loss increasing** → Training instability
**⚠️ All deaths clustered** → Agent not exploring (but depletion should fix this!)

---

## 🎯 **Usage Scenarios**

### Scenario 1: Quick Validation
```bash
# 5-minute test
python scripts/train_soliter.py --cycles 50 --output-dir experiments/quick_test
python scripts/monitor_training_live.py experiments/quick_test --refresh 1

# Verify:
# - System running
# - Metrics updating
# - No errors
```

### Scenario 2: Overnight Run
```bash
# Start before bed
nohup python scripts/train_soliter.py --cycles 2000 \
  --output-dir experiments/overnight > training.log 2>&1 &

# Monitor from another terminal
python scripts/monitor_training_live.py experiments/overnight --refresh 30

# Check in morning - dashboard shows complete status
```

### Scenario 3: Multi-Person Monitoring
```bash
# One person starts training
python scripts/train_soliter.py --cycles 1000 --output-dir experiments/team_run

# Multiple people can monitor
# Person 1:
python scripts/monitor_training_live.py experiments/team_run --port 8050

# Person 2:
python scripts/monitor_training_live.py experiments/team_run --port 8051

# Each gets their own dashboard instance
```

---

## 💡 **Performance Tips**

### For Fast Training (Minimize Overhead)
```bash
# Use slower refresh rate
--refresh 30

# Dashboard checks file every 30s
# Minimal impact on training speed
```

### For Active Development (Quick Feedback)
```bash
# Use fast refresh
--refresh 1

# Dashboard updates every second
# Slight overhead but better visibility
```

### Balance (Recommended)
```bash
# 5-10 second refresh
--refresh 5

# Good compromise: responsive but low overhead
```

---

## 🔍 **Troubleshooting**

### Dashboard shows "ERROR"
**Problem:** Training log not found
**Solution:** 
```bash
# Verify log file exists
ls experiments/your_run/training_*.json

# If not, training hasn't started or crashed
# Check training log
tail training.log
```

### Dashboard not updating
**Problem:** Training stopped or completed
**Solution:**
- Check status indicator
- If COMPLETE (blue) → training finished
- If RUNNING but not updating → training may have crashed
- Check training process: `ps aux | grep train_soliter`

### Port already in use
**Problem:** Port 8050 busy
**Solution:**
```bash
# Use different port
python scripts/monitor_training_live.py experiments/run1 --port 9000
```

### Can't access remotely
**Problem:** Firewall or host setting
**Solution:**
```bash
# Use 0.0.0.0 for remote access
--host 0.0.0.0

# Open firewall port if needed
# Check server firewall settings
```

---

## 📦 **Dependencies**

The web dashboard requires:
```bash
pip install flask numpy
```

Already in your environment if you can run training.

---

## 🎨 **Dashboard Design**

### Visual Features
- **Gradient background** (purple theme)
- **Animated status indicators** (pulsing when running)
- **Color-coded metrics** (green for positive trends, red for negative)
- **Responsive design** (works on mobile/tablet/desktop)
- **Interactive charts** (Plotly - zoom, pan, hover for details)

### Accessibility
- **Pause button** (stop auto-refresh to read)
- **Manual refresh** (force update anytime)
- **Configurable rate** (1-60 seconds)
- **Clear status** (RUNNING/COMPLETE/ERROR)

---

## 🚀 **Full Example Workflow**

```bash
# 1. Start training (Terminal 1)
cd soliter-develop
python scripts/train_soliter.py --cycles 1000 --output-dir experiments/full_test

# 2. Start monitor (Terminal 2)
python scripts/monitor_training_live.py experiments/full_test --refresh 5 --port 8050

# 3. Open browser
# Go to: http://localhost:8050

# 4. Watch dashboard
# - Metrics update every 5 seconds
# - Charts show real-time trends
# - Status shows RUNNING with pulsing indicator

# 5. Adjust if needed
# - Change refresh rate in dropdown (1-60s)
# - Pause if you want to read
# - Refresh manually with button

# 6. When complete
# - Status changes to COMPLETE (blue)
# - Auto-refresh stops
# - Can still browse final results

# 7. Post-analysis (Terminal 3)
python scripts/analyze_learning.py experiments/full_test/training_*.json
```

---

## ✅ **Summary**

You now have:

1. **Live web dashboard** - Monitor from browser
2. **Separate process** - No training impact
3. **Configurable refresh** - 1-60 seconds
4. **Interactive controls** - Pause, refresh, adjust rate
5. **Real-time charts** - 4 key metrics visualized
6. **Remote capable** - SSH tunnel or direct access
7. **Integrated viz** - Still available for debugging

**Training + Monitoring = Two Terminals, Full Visibility** 🎉

---

**Files:**
- `scripts/monitor_training_live.py` - Web dashboard (new)
- `scripts/train_soliter_with_viz.py` - Integrated viz (kept)
- `scripts/analyze_learning.py` - Post-training analysis (existing)

**Status:** ✅ Complete and ready to use!
