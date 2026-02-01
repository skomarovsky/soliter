#!/usr/bin/env python3
"""Verify that all dependencies are properly installed."""

import sys

def check_import(module_name: str, package_name: str = None) -> bool:
    """Try to import a module and report status."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name}: {e}")
        return False

def main():
    """Run all checks."""
    print("Checking dependencies...\n")
    
    checks = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("ncps", "Neural Circuit Policies"),
        ("pygame", "PyGame"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("yaml", "PyYAML"),
        ("h5py", "h5py"),
    ]
    
    results = [check_import(module, package) for module, package in checks]
    
    print(f"\n{sum(results)}/{len(results)} dependencies installed successfully")
    
    # Check PyTorch CUDA
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Error checking PyTorch: {e}")
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
