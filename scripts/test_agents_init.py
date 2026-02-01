#!/usr/bin/env python3
"""Test the agents package __init__.py"""

import sys

print("Step 1: Import soliter package")
try:
    import soliter
    print("  ✓ Success")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\nStep 2: Import soliter.agents package")
try:
    import soliter.agents
    print("  ✓ Success")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: Check what's in soliter.agents")
for name in dir(soliter.agents):
    if not name.startswith('_'):
        print(f"  - {name}")

print("\nStep 4: Try to access SoliterAgent")
try:
    agent_class = soliter.agents.SoliterAgent
    print(f"  ✓ Success: {agent_class}")
except AttributeError as e:
    print(f"  ✗ Failed: {e}")
    
print("\nStep 5: Try direct import")
try:
    from soliter.agents import SoliterAgent
    print(f"  ✓ Success: {SoliterAgent}")
except ImportError as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

