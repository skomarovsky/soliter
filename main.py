#!/usr/bin/env python3
"""
Main entry point for the Soliter project.

This script initializes and runs the Soliter agent with its biological drive system
in a simulated environment. The agent must balance its vital parameters (Energy,
Hydration, Temperature, Wakefulness) to survive by finding resources in the world.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from scripts.train_soliter import main as train_main


def main():
    """Main entry point for the soliter project."""
    print("Starting Soliter...")
    print("Running biological drive-based agent training...")
    train_main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
