#!/usr/bin/env python
"""Environment check for Metal backend tests.

This script checks that the environment is correctly set up for running
Metal backend tests, including required dependencies and hardware.
"""

import os
import sys
import platform
import subprocess
import importlib.util
from typing import Dict, List, Tuple, Optional

def check_python_version() -> bool:
    """Check Python version"""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    print(f"Python version: {'.'.join(map(str, current_version))}")
    
    if current_version >= required_version:
        print("✅ Python version is sufficient")
        return True
    else:
        print(f"❌ Python version is too old. Required: {'.'.join(map(str, required_version))}")
        return False

def check_package(package_name: str) -> bool:
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        try:
            module = importlib.import_module(package_name)
            if hasattr(module, '__version__'):
                print(f"✅ {package_name} is installed (version {module.__version__})")
            else:
                print(f"✅ {package_name} is installed")
            return True
        except ImportError:
            print(f"❌ {package_name} is installed but could not be imported")
            return False
    else:
        print(f"❌ {package_name} is not installed")
        return False

def check_operating_system() -> bool:
    """Check operating system"""
    system = platform.system()
    version = platform.version()
    machine = platform.machine()
    
    print(f"Operating system: {system} {version} ({machine})")
    
    if system == "Darwin" and machine in ["arm64", "arm"]:
        print("✅ Running on Apple Silicon Mac")
        return True
    else:
        print("❌ Not running on Apple Silicon Mac")
        return False

def check_metal_hardware() -> bool:
    """Check Metal hardware availability"""
    # Try to import PyMetal or MLX to check Metal support
    try:
        import mlx.core as mx
        
        # Check if running on Apple Silicon using MLX
        # MLX only runs on Metal, so if we can create an array and execute an operation,
        # it means Metal hardware is available
        try:
            # Create a small array and perform a simple operation
            a = mx.array([1, 2, 3])
            b = a + 1
            # Force execution by converting to numpy
            import numpy as np
            np.array(b)
            print("✅ Metal hardware is available")
            return True
        except Exception as e:
            print(f"❌ Metal hardware execution failed: {e}")
            return False
    except ImportError:
        print("⚠️ Could not check Metal hardware (MLX not installed)")
        return False
    except Exception as e:
        print(f"❌ Error checking Metal hardware: {e}")
        return False

def check_xcode() -> bool:
    """Check Xcode installation and version"""
    try:
        # Check for xcode-select
        result = subprocess.run(
            ["xcode-select", "-p"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print("❌ Xcode command line tools not installed")
            print("   Install with: xcode-select --install")
            return False
        
        xcode_path = result.stdout.strip()
        print(f"Xcode path: {xcode_path}")
        
        # Check Xcode version
        result = subprocess.run(
            ["xcodebuild", "-version"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print("❌ Xcode not properly installed")
            return False
        
        version_output = result.stdout.strip()
        print(f"Xcode version: {version_output}")
        
        # Extract version
        version_line = version_output.splitlines()[0]
        version_str = version_line.split(" ")[-1]
        
        try:
            major_version = int(version_str.split(".")[0])
            if major_version >= 14:
                print("✅ Xcode version is sufficient (14+)")
                return True
            else:
                print(f"⚠️ Xcode version may be too old: {version_str}")
                print("   Recommended: Xcode 14.0 or newer")
                return False
        except (ValueError, IndexError):
            print(f"⚠️ Could not parse Xcode version: {version_str}")
            return False
    except FileNotFoundError:
        print("❌ Xcode tools not found")
        print("   Install with: xcode-select --install")
        return False
    except Exception as e:
        print(f"❌ Error checking Xcode: {e}")
        return False

def check_metal_compiler() -> bool:
    """Check Metal compiler"""
    try:
        # Check for metal command line tool
        result = subprocess.run(
            ["xcrun", "-f", "metal"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print("❌ Metal compiler not found")
            return False
        
        metal_path = result.stdout.strip()
        print(f"Metal compiler: {metal_path}")
        
        # Check metal version
        result = subprocess.run(
            ["xcrun", "metal", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"⚠️ Could not determine Metal compiler version")
            return False
        
        version_output = result.stdout.strip()
        print(f"Metal version: {version_output}")
        
        # We don't fail on specific Metal version, just report it
        print("✅ Metal compiler is available")
        return True
    except FileNotFoundError:
        print("❌ Metal compiler not found")
        return False
    except Exception as e:
        print(f"❌ Error checking Metal compiler: {e}")
        return False

def check_metal_backend() -> bool:
    """Check Metal backend installation"""
    # Check for Metal backend modules
    metal_modules = [
        "metal_backend",
        "triton_to_metal_converter"
    ]
    
    # First try importing directly
    all_found = True
    for module_name in metal_modules:
        if not check_package(module_name):
            all_found = False
    
    # If not found, check if they're within a parent package
    if not all_found:
        print("Attempting to find Metal modules in parent package...")
        
        # Add parent directory to path for imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        all_found = True
        for module_name in metal_modules:
            if not check_package(module_name):
                all_found = False
    
    return all_found

def main():
    """Main function"""
    print("=== Metal Backend Environment Check ===\n")
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check required packages
    print("\n=== Required Packages ===")
    numpy_ok = check_package("numpy")
    matplotlib_ok = check_package("matplotlib")
    mlx_ok = check_package("mlx.core")
    
    # Check operating system
    print("\n=== System Information ===")
    os_ok = check_operating_system()
    
    # Check Metal hardware
    metal_hw_ok = check_metal_hardware()
    
    # Check Xcode and toolchain
    print("\n=== Development Tools ===")
    xcode_ok = check_xcode()
    metal_compiler_ok = check_metal_compiler()
    
    # Check Metal backend
    print("\n=== Metal Backend ===")
    metal_backend_ok = check_metal_backend()
    
    # Print summary
    print("\n=== Summary ===")
    num_passed = sum(1 for x in [
        python_ok, numpy_ok, matplotlib_ok, mlx_ok, 
        os_ok, metal_hw_ok, xcode_ok, metal_compiler_ok, 
        metal_backend_ok
    ] if x)
    num_total = 9
    
    print(f"Passed: {num_passed}/{num_total} checks")
    
    if num_passed == num_total:
        print("✅ Environment is correctly set up for Metal backend tests")
        return 0
    else:
        print("❌ Environment is not correctly set up for Metal backend tests")
        print("\nMissing requirements:")
        if not python_ok:
            print("- Python 3.8+ is required")
        if not numpy_ok:
            print("- NumPy is required: pip install numpy")
        if not matplotlib_ok:
            print("- Matplotlib is required: pip install matplotlib")
        if not mlx_ok:
            print("- MLX is required: pip install mlx")
        if not os_ok:
            print("- Apple Silicon Mac is required")
        if not metal_hw_ok:
            print("- Metal hardware support is required")
        if not xcode_ok:
            print("- Xcode 14+ is required: Install from App Store or xcode-select --install")
        if not metal_compiler_ok:
            print("- Metal compiler is required: Install Xcode with Metal support")
        if not metal_backend_ok:
            print("- Metal backend modules need to be properly installed")
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 