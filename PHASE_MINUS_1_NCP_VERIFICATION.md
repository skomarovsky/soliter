# PHASE -1: NCP/CfC ARCHITECTURE VERIFICATION

## 🎯 CRITICAL: Verify NCP/CfC is Actually Implemented!

Before Phase 0, we MUST verify the neural architecture is correct.

---

## ✅ What We Found in Code:

### **1. CfC Import (Line 11):**
```python
from ncps.torch import CfC
from ncps.wirings import NCP
```

This imports from the **ncps** library (Ramin Hasani et al.)
- GitHub: https://github.com/mlech26l/ncps
- Paper: "Liquid Time-Constant Networks" (2020)

### **2. CfC Network Creation (Line 67):**
```python
self.cfc = CfC(
    input_size=sensory_size,
    units=self.wiring.units,
    mixed_memory=True,  # Short + long time constants
    mode="default",
    activation="lecun_tanh",
)
```

This creates a proper CfC network!

### **3. NCP Wiring (Line 56):**
```python
self.wiring = create_soliter_wiring(
    sensory_size=sensory_size,
    inter_size=inter_size,
    command_size=command_size,
    motor_size=motor_size,
)
```

This creates the Neural Circuit Policy topology.

---

## ⚠️ CRITICAL QUESTION: Is ncps Package Installed?

The code ASSUMES this library exists, but we need to verify:

```bash
python -c "import ncps; print(ncps.__version__)"
```

### **If Installed:**
✅ Proceed to Phase 0

### **If NOT Installed:**
❌ Need to install first:

```bash
pip install ncps
# or
pip install ncps-torch
```

---

## 🔍 What is ncps Library?

### **Official Package:**
- **Name:** Neural Circuit Policies (ncps)
- **Authors:** Ramin Hasani, Mathias Lechner (MIT)
- **Paper:** "Closed-form continuous-time neural networks" (Nature Machine Intelligence, 2022)

### **What it Provides:**

**1. CfC (Closed-form Continuous-time) Networks:**
```python
from ncps.torch import CfC

# Creates continuous-time RNN
cfc = CfC(input_size=51, units=256)
```

**2. LTC (Liquid Time-Constant) Networks:**
```python
from ncps.torch import LTC

# Alternative continuous-time architecture
ltc = LTC(input_size=51, units=256)
```

**3. NCP (Neural Circuit Policy) Wiring:**
```python
from ncps.wirings import NCP

# Creates C. elegans-inspired topology
wiring = NCP(
    inter_neurons=64,
    command_neurons=32,
    motor_neurons=3,
    sensory_fanout=4,  # Each sensor connects to 4 interneurons
    inter_fanout=2,
)
```

---

## 🧪 VERIFICATION TEST

### **Step 1: Check Package Installation**

```bash
cd ~/wss/soliter
python -c "import ncps; print('✅ ncps installed:', ncps.__version__)"
```

**Expected Output:**
```
✅ ncps installed: 1.0.0  (or similar)
```

**If Error:**
```
ModuleNotFoundError: No module named 'ncps'
```
→ Need to install!

### **Step 2: Test CfC Creation**

```bash
python -c "
from ncps.torch import CfC
import torch

cfc = CfC(input_size=51, units=256, mixed_memory=True)
x = torch.randn(1, 51)
output, hidden = cfc(x)
print('✅ CfC works! Output shape:', output.shape)
"
```

**Expected Output:**
```
✅ CfC works! Output shape: torch.Size([1, 256])
```

### **Step 3: Test Agent Brain**

```bash
cd ~/wss/soliter
python -c "
import torch
import sys
sys.path.insert(0, '.')

from soliter.core.cfc_network import CfCBrain

brain = CfCBrain(sensory_size=51)
sensors = torch.randn(51)
action, hidden = brain(sensors.unsqueeze(0))
print('✅ Soliter CfC Brain works!')
print('   Input shape:', sensors.shape)
print('   Output shape:', action.shape)
print('   Actions:', action.squeeze().tolist())
"
```

**Expected Output:**
```
✅ Soliter CfC Brain works!
   Input shape: torch.Size([51])
   Output shape: torch.Size([1, 3])
   Actions: [0.234, -0.567, 0.123]
```

---

## 📋 INSTALLATION (If Needed)

### **Method 1: pip (Recommended)**
```bash
pip install ncps
```

### **Method 2: From Source**
```bash
git clone https://github.com/mlech26l/ncps.git
cd ncps
pip install -e .
```

### **Dependencies:**
```
torch >= 1.8.0
numpy >= 1.19.0
```

---

## 🎯 PHASE -1 CHECKLIST

Before proceeding to Phase 0:

- [ ] Verify ncps package installed
- [ ] Test CfC creation works
- [ ] Test Soliter brain creation works
- [ ] Verify NCP wiring topology correct
- [ ] Check sensor input size (51 channels correct?)
- [ ] Confirm motor output size (3: velocity, turn, sleep)

---

## 🚨 CRITICAL ISSUE: Sensor Size Mismatch?

### **Current Code Assumes:**
```python
sensory_size = 41  # In CfCBrain default
```

### **But NCP Trainer Uses:**
```python
# 41 proximity + 6 gradients + 4 drives = 51 total
```

**NEED TO FIX:** Update CfCBrain default to 51!

```python
class CfCBrain(nn.Module):
    def __init__(
        self,
        sensory_size: int = 51,  # ← Change from 41 to 51!
        ...
    ):
```

---

## ✅ Once Verified, Create PHASE_MINUS_1_RESULTS.md:

```markdown
# Phase -1 Results: NCP/CfC Verification

## Installation Status:
- [x] ncps package installed (version: X.X.X)
- [x] CfC creation works
- [x] Soliter brain creation works

## Architecture Confirmed:
- Input: 51 sensors
- Interneurons: 256 (CfC units)
- Command: 64 neurons
- Output: 3 motors

## Issues Found:
- [ ] None (or list issues)

## Ready for Phase 0: YES/NO
```

---

## 🎯 IMMEDIATE ACTION

**Run verification test:**

```bash
cd ~/wss/soliter
tar -xzf updated_files_only.tar.gz

# Test 1: Package installed?
python -c "import ncps; print('ncps version:', ncps.__version__)"

# Test 2: Brain works?
python -c "
from soliter.core.cfc_network import CfCBrain
import torch
brain = CfCBrain(sensory_size=51)
sensors = torch.randn(1, 51)
action, _ = brain(sensors)
print('✅ CfC Brain working! Output:', action.shape)
"
```

**Report back the results!**

If ncps not installed, install it first:
```bash
pip install ncps
```

Then re-run verification tests.
