#!/usr/bin/env python3
"""Diagnose what happens when loading soliter_agent.py"""

import sys
import importlib.util

print("Loading soliter/agents/soliter_agent.py directly...\n")

# Try to load the module and catch any errors during loading
spec = importlib.util.spec_from_file_location(
    "soliter.agents.soliter_agent",
    "soliter/agents/soliter_agent.py"
)

try:
    module = importlib.util.module_from_spec(spec)
    print("Module object created")
    
    # This is where the actual code execution happens
    print("Executing module code...")
    spec.loader.exec_module(module)
    print("✓ Module executed successfully\n")
    
    # Check what's in it
    print("Module contents:")
    for name in dir(module):
        if not name.startswith('_'):
            obj = getattr(module, name)
            print(f"  {name}: {type(obj).__name__}")
    
except Exception as e:
    print(f"\n✗ Error during module loading:")
    print(f"  {type(e).__name__}: {e}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()

