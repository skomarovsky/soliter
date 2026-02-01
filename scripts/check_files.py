#!/usr/bin/env python3
"""Check that all Phase 2 files exist."""

import os
from pathlib import Path

files_needed = [
    "soliter/__init__.py",
    "soliter/core/__init__.py",
    "soliter/core/cfc_network.py",
    "soliter/core/ncp_wiring.py",
    "soliter/agents/__init__.py",
    "soliter/agents/soliter_agent.py",
]

print("Checking Phase 2 files...\n")

all_exist = True
for filepath in files_needed:
    path = Path(filepath)
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    status = "✓" if exists else "✗"
    print(f"{status} {filepath:45s} ({size:,} bytes)")
    if not exists:
        all_exist = False

if all_exist:
    print("\n✓ All files exist!")
else:
    print("\n✗ Some files are missing. Create them first.")

