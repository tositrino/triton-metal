#!/usr/bin/env python
"""
Simplified test script for Metal backend integration.

This script verifies that the Metal backend package is correctly installed
and accessible, without requiring the full Triton installation.
"""

import os
import sys
import importlib
import pkg_resources

def test_metal_package():
    """Test that the Metal backend package is installed"""
    print("Testing Metal backend package installation...")
    
    # Try to import the metal backend package
    try:
        import triton
        print(f"✅ Metal backend package is installed! Version: {triton.__version__}")
        return True
    except ImportError:
        try:
            # Check if it's registered as an entry point
            metal_entry_points = list(pkg_resources.iter_entry_points("triton.backends", name="metal"))
            if metal_entry_points:
                print(f"✅ Metal backend is registered as an entry point: {metal_entry_points[0]}")
                return True
            else:
                # Check if the package is installed
                installed_packages = [pkg.key for pkg in pkg_resources.working_set]
                if "triton-metal" in installed_packages:
                    print(f"✅ triton-metal package is installed but not properly imported")
                    return True
                else:
                    print("❌ Metal backend package is NOT installed!")
                    return False
        except Exception as e:
            print(f"❌ Error checking entry points: {e}")
            return False

def test_metal_modules():
    """Test that key Metal backend modules are available"""
    print("\nTesting Metal backend modules...")
    
    modules_to_check = [
        "metal_backend",
        "triton_to_metal_converter"
    ]
    
    search_paths = [
        os.path.join(os.path.dirname(__file__), "third_party", "metal", "python"),
        os.path.join(os.path.dirname(__file__), "python", "triton", "backends", "metal")
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"Checking path: {search_path}")
            sys.path.insert(0, search_path)
    
    success = True
    for module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ Successfully imported {module_name}")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            success = False
    
    return success

def test_metal_files():
    """Test that key Metal backend files exist"""
    print("\nChecking for key Metal backend files...")
    
    files_to_check = [
        "third_party/metal/python/metal_backend.py",
        "third_party/metal/python/setup.py",
        "python/triton/backends/metal/__init__.py",
        "python/triton/backends/metal/compiler.py",
        "python/triton/backends/metal/driver.py"
    ]
    
    success = True
    for file_path in files_to_check:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            success = False
    
    return success

def main():
    """Main function"""
    print("=== Metal Backend Integration Test ===\n")
    
    package_installed = test_metal_package()
    modules_available = test_metal_modules()
    files_exist = test_metal_files()
    
    print("\n=== Test Summary ===")
    print(f"Package installed: {'✅' if package_installed else '❌'}")
    print(f"Modules available: {'✅' if modules_available else '❌'}")
    print(f"Files exist:       {'✅' if files_exist else '❌'}")
    
    if package_installed and modules_available and files_exist:
        print("\n✅ Metal backend integration is successful!")
        return 0
    else:
        print("\n❌ Metal backend integration has issues that need to be addressed.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 