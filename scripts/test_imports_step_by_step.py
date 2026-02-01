#!/usr/bin/env python3
"""Test imports step by step to find the issue."""

import sys
import traceback

def test_step(description, code):
    """Test a single import step."""
    print(f"\n[{description}]")
    try:
        exec(code, globals())
        print("  ✓ Success")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return False

print("="*60)
print("Testing imports step by step")
print("="*60)

# Step 1: Basic imports
if not test_step("Import torch", "import torch"):
    sys.exit(1)

if not test_step("Import ncps", "import ncps"):
    sys.exit(1)

if not test_step("Import ncps.torch", "from ncps.torch import CfC"):
    sys.exit(1)

if not test_step("Import ncps.wirings", "from ncps.wirings import NCP"):
    sys.exit(1)

# Step 2: Import soliter package
if not test_step("Import soliter", "import soliter"):
    sys.exit(1)

# Step 3: Import core
if not test_step("Import soliter.core", "import soliter.core"):
    sys.exit(1)

# Step 4: Import ncp_wiring
if not test_step("Import ncp_wiring", "from soliter.core import ncp_wiring"):
    sys.exit(1)

# Step 5: Import cfc_network  
if not test_step("Import cfc_network", "from soliter.core import cfc_network"):
    sys.exit(1)

# Step 6: Import CfCBrain class
if not test_step("Import CfCBrain", "from soliter.core.cfc_network import CfCBrain"):
    sys.exit(1)

# Step 7: Import agents package
if not test_step("Import soliter.agents", "import soliter.agents"):
    sys.exit(1)

# Step 8: Import soliter_agent module
if not test_step("Import soliter_agent module", "from soliter.agents import soliter_agent"):
    sys.exit(1)

# Step 9: Import SoliterAgent class
if not test_step("Import SoliterAgent", "from soliter.agents.soliter_agent import SoliterAgent"):
    sys.exit(1)

# Step 10: Import VitalsConfig
if not test_step("Import VitalsConfig", "from soliter.agents.soliter_agent import VitalsConfig"):
    sys.exit(1)

print("\n" + "="*60)
print("✓ All imports successful!")
print("="*60)

