#!/usr/bin/env python
"""
Metal Backend Compatibility Tutorial
===================================

This tutorial provides a comprehensive walkthrough of how to check system 
compatibility with Triton's Metal backend and optimize your code for Apple 
Silicon GPUs, especially M3 chips.

You will learn about:
- Checking your system compatibility with the Triton Metal backend
- Detecting and leveraging M3-specific optimizations
- Understanding memory layout optimizations for Apple Silicon
- Testing and troubleshooting your Metal backend setup

Prerequisites:
- macOS 13.5 or newer
- Apple Silicon Mac (M1, M2, or M3 series)
- Python 3.8 or newer
- MLX package installed
"""

import os
import sys
import platform
import subprocess
import importlib.util
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Define colors for output formatting
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}=== {title} ==={Colors.ENDC}")

def print_status(message: str, status: bool, details: Optional[str] = None):
    """Print a status message with colored indicators"""
    status_str = f"{Colors.GREEN}✓{Colors.ENDC}" if status else f"{Colors.RED}✗{Colors.ENDC}"
    print(f"{status_str} {message}")
    if details:
        color = Colors.GREEN if status else Colors.YELLOW
        print(f"  {color}{details}{Colors.ENDC}")

def print_code_example(code: str):
    """Print a formatted code example"""
    print(f"{Colors.BLUE}```python{Colors.ENDC}")
    print(code)
    print(f"{Colors.BLUE}```{Colors.ENDC}")

# Part 1: System Compatibility Checks
print_section("System Compatibility Check Tutorial")

print("""
Before using the Triton Metal backend, it's essential to verify that your system
meets all the necessary requirements. The Metal backend requires specific hardware
and software configurations to function correctly, especially for advanced features.

Let's check your system step by step:
""")

# Function 1: Check macOS Version
def check_macos_version() -> Tuple[bool, str]:
    """Check if the macOS version is compatible (13.5+)"""
    macos_version = platform.mac_ver()[0]
    
    try:
        major, minor, patch = map(int, macos_version.split('.'))
        is_compatible = (major > 13) or (major == 13 and minor >= 5)
        details = f"Found macOS {macos_version}"
        if not is_compatible:
            details += " (required: 13.5+)"
        return is_compatible, details
    except Exception as e:
        return False, f"Could not determine macOS version: {e}"

# Function 2: Check Apple Silicon
def check_apple_silicon() -> Tuple[bool, str]:
    """Check if running on Apple Silicon"""
    is_apple_silicon = platform.machine() == 'arm64'
    details = "Apple Silicon detected" if is_apple_silicon else "Intel CPU detected (Apple Silicon required)"
    return is_apple_silicon, details

# Function 3: Check Package Installation
def check_package_installed(package_name: str) -> Tuple[bool, str]:
    """Check if a Python package is installed"""
    spec = importlib.util.find_spec(package_name)
    is_installed = spec is not None
    
    # If installed, try to get version
    version = "unknown"
    if is_installed:
        try:
            # Try to import and get version
            module = importlib.import_module(package_name)
            if hasattr(module, '__version__'):
                version = module.__version__
            elif hasattr(module, 'version'):
                version = module.version
        except Exception:
            pass
    
    details = f"Version {version}" if is_installed else "Not installed"
    return is_installed, details

# Function 4: Check M-series generation
def check_m_series_generation() -> Tuple[bool, Optional[str], str]:
    """Check for M-series generation and compatibility"""
    try:
        # Try to import the hardware capabilities module
        from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
        
        chip_gen = getattr(hardware_capabilities, "chip_generation", None)
        if chip_gen is not None:
            generation_name = chip_gen.name
            generation_value = chip_gen.value
            
            # Check if M3 or newer
            is_m3_or_newer = False
            generation_details = f"Detected {generation_name}"
            
            if hasattr(AppleSiliconGeneration, "M3"):
                m3_value = AppleSiliconGeneration.M3.value
                is_m3_or_newer = generation_value >= m3_value
                if not is_m3_or_newer:
                    generation_details += f" (M3 or newer recommended for optimal performance)"
            else:
                generation_details += " (Could not check for M3 compatibility)"
            
            return True, generation_name, generation_details
        else:
            return False, None, "Hardware detection available but couldn't identify chip generation"
    except ImportError:
        # If we can't import the module, try to get chip info from system
        try:
            chip_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            if "Apple M" in chip_info:
                if "M3" in chip_info:
                    return True, "M3", f"Detected {chip_info}"
                elif "M2" in chip_info:
                    return True, "M2", f"Detected {chip_info} (M3 or newer recommended)"
                elif "M1" in chip_info:
                    return True, "M1", f"Detected {chip_info} (M3 or newer recommended)"
                else:
                    return True, "Unknown M-series", f"Detected {chip_info}"
            else:
                return False, None, "Not an Apple Silicon Mac"
        except:
            return False, None, "Could not detect Apple Silicon generation"

# Function 5: Check for M3-specific optimizations
def check_m3_specific_optimizations() -> Tuple[bool, List[str]]:
    """Check for available M3-specific optimizations"""
    try:
        # Try to import M3 optimization module
        import m3_optimizations
        from m3_optimizations import M3Optimizer, M3Feature
        
        # Create optimizer instance
        optimizer = M3Optimizer()
        
        # Check if we're on M3
        is_m3 = optimizer.is_m3
        
        # Get available features
        available_features = []
        if is_m3:
            for feature in M3Feature:
                if optimizer.is_feature_available(feature):
                    available_features.append(feature.name)
        
        return is_m3, available_features
    except ImportError:
        return False, []

# Now perform the checks
print("1. Checking macOS version...")
is_macos_compatible, macos_details = check_macos_version()
print_status("macOS 13.5+", is_macos_compatible, macos_details)

print("\n2. Checking for Apple Silicon...")
is_apple_silicon, silicon_details = check_apple_silicon()
print_status("Apple Silicon", is_apple_silicon, silicon_details)

if is_apple_silicon:
    print("\n3. Checking Apple Silicon generation...")
    has_generation, generation_name, generation_details = check_m_series_generation()
    print_status("M-series detection", has_generation, generation_details)
    
    # If M3 detected, check for specific optimizations
    if has_generation and generation_name and "M3" in generation_name:
        print("\n4. Checking for M3-specific optimizations...")
        is_m3, available_features = check_m3_specific_optimizations()
        if is_m3 and available_features:
            print_status("M3 optimizations available", True, f"Found optimizations: {', '.join(available_features)}")
        else:
            print_status("M3 optimizations available", False, "M3 detected but no specific optimizations found")

print("\n5. Checking for MLX package...")
has_mlx, mlx_details = check_package_installed("mlx.core")
print_status("MLX installed", has_mlx, mlx_details)

# Part 2: Using the Metal Backend
print_section("Using the Metal Backend")

print("""
Unlike traditional Triton usage, the Metal backend does not require the Triton package
itself for execution. Instead, it provides its own implementation that utilizes
MLX and Metal Performance Shaders.

To use the Metal backend, you need to set the appropriate environment variable
before importing any modules:
""")

print_code_example("""
import os
os.environ["TRITON_BACKEND"] = "metal"

# Now import the modules
import mlx.core as mx
""")

print("""
This tells the system to use the Metal implementation instead of the default Triton
implementation. The Metal backend will automatically detect your hardware and apply
the appropriate optimizations based on your chip generation.
""")

# Part 3: Memory Layout Optimizations
print_section("Memory Layout Optimizations")

print("""
One of the key optimizations available in the Metal backend is the COALESCED
memory layout for reduction operations. This layout is particularly effective
on Apple Silicon GPUs, especially on M3 chips.

Let's check if your system supports this optimization:
""")

def check_coalesced_memory_layout() -> Tuple[bool, str]:
    """Check if COALESCED memory layout is defined in metal_memory_manager"""
    try:
        from metal_memory_manager import MemoryLayout
        
        # Check if COALESCED is defined
        if hasattr(MemoryLayout, "COALESCED"):
            coalesced_value = MemoryLayout.COALESCED.value
            details = f"COALESCED layout defined with value {coalesced_value}"
            return True, details
        else:
            return False, "MemoryLayout enum exists but COALESCED is not defined"
    except ImportError:
        return False, "metal_memory_manager module not found"
    except Exception as e:
        return False, f"Error checking for COALESCED layout: {e}"

def check_optimizer_consistency() -> Tuple[bool, str]:
    """Check if COALESCED is consistently defined in memory layout optimizer"""
    try:
        from metal_memory_manager import MemoryLayout as ManagerLayout
        from memory_layout_optimizer import MemoryLayout as OptimizerLayout
        
        # Check if both have COALESCED
        if hasattr(ManagerLayout, "COALESCED") and hasattr(OptimizerLayout, "COALESCED"):
            manager_value = ManagerLayout.COALESCED.value
            optimizer_value = OptimizerLayout.COALESCED.value
            
            if manager_value == optimizer_value:
                details = f"Consistent definition in both modules (value: {manager_value})"
                return True, details
            else:
                return False, f"Inconsistent values: {manager_value} vs {optimizer_value}"
        else:
            if not hasattr(ManagerLayout, "COALESCED"):
                return False, "COALESCED not defined in metal_memory_manager.MemoryLayout"
            else:
                return False, "COALESCED not defined in memory_layout_optimizer.MemoryLayout"
    except ImportError as e:
        return False, f"Could not import required modules: {e}"
    except Exception as e:
        return False, f"Error checking consistency: {e}"

# Check for COALESCED memory layout
has_coalesced, coalesced_details = check_coalesced_memory_layout()
print_status("COALESCED memory layout defined", has_coalesced, coalesced_details)

# Check for consistency between modules
is_consistent, consistency_details = check_optimizer_consistency()
print_status("Consistent COALESCED definition", is_consistent, consistency_details)

print("""
The COALESCED memory layout optimizes reduction operations by organizing memory
in a way that maximizes memory access efficiency on Apple Silicon GPUs. This 
layout is particularly beneficial for reduction operations like sum, mean, max, etc.

When using the Metal backend, this optimization is applied automatically when appropriate.
You don't need to explicitly request it in your code.
""")

# Part 4: M3-Specific Optimizations
print_section("M3-Specific Optimizations")

print("""
If you're running on an M3 chip, the Metal backend provides additional 
optimizations that take advantage of M3-specific hardware features, including:

1. Dynamic Caching - Optimizes memory access patterns for the M3's advanced caching
2. Enhanced Matrix Coprocessor - Uses specialized hardware for matrix operations
3. Shared Memory Atomics - Leverages hardware-accelerated atomic operations
4. Enhanced SIMD capabilities - Uses the improved SIMD units in M3
5. Advanced Warp Scheduling - Optimizes thread scheduling
6. Memory Compression - Uses M3's improved memory compression
""")

if has_generation and generation_name and "M3" in generation_name:
    print(f"{Colors.GREEN}Your M3 chip supports these optimizations!{Colors.ENDC}")
    
    print("""
    Here's an example of how you can explicitly leverage M3-specific optimizations
    in your code:
    """)
    
    print_code_example("""
    import os
    os.environ["TRITON_BACKEND"] = "metal"
    
    # Import M3-specific optimizer
    from m3_optimizations import M3Optimizer
    import mlx.core as mx
    
    # Create optimizer instance (it will detect M3 automatically)
    optimizer = M3Optimizer()
    
    # Create sample matrices
    a = mx.random.normal((1024, 1024))
    b = mx.random.normal((1024, 1024))
    
    # Use M3-optimized matrix multiplication
    result = optimizer.optimize_matmul(a, b)
    
    # Use M3-optimized reduction
    sum_result = optimizer.optimize_reduction(a, mx.sum, axis=0)
    
    print(f"Matrix multiplication shape: {result.shape}")
    print(f"Reduction result shape: {sum_result.shape}")
    """)
    
    print("""
    Note that the Metal backend will automatically apply M3-specific optimizations
    where applicable, even if you don't explicitly use the M3Optimizer class. This 
    class simply gives you more fine-grained control over the optimizations.
    """)
else:
    print(f"{Colors.YELLOW}Your system does not have an M3 chip, so these specific optimizations are not available.{Colors.ENDC}")
    print("However, the Metal backend will still apply optimizations appropriate for your hardware.")

# Part 5: Troubleshooting
print_section("Troubleshooting Tips")

print("""
If you encounter issues with the Metal backend, here are some common troubleshooting steps:
""")

def get_troubleshooting_tips():
    """Generate troubleshooting tips"""
    tips = [
        f"{Colors.YELLOW}• Make sure macOS is version 13.5 or newer{Colors.ENDC}",
        f"{Colors.YELLOW}• Ensure you're using an Apple Silicon Mac (M1, M2, or M3 series){Colors.ENDC}",
        f"{Colors.YELLOW}• Install MLX package: pip install mlx{Colors.ENDC}",
        f"{Colors.YELLOW}• Set TRITON_BACKEND environment variable: os.environ['TRITON_BACKEND'] = 'metal'{Colors.ENDC}",
        f"{Colors.YELLOW}• For debugging, enable debug logs: os.environ['triton_DEBUG'] = '1'{Colors.ENDC}",
        f"{Colors.YELLOW}• Check for memory allocation errors by reducing batch sizes or tensor dimensions{Colors.ENDC}",
        f"{Colors.YELLOW}• Verify grid and block dimensions for kernel launches{Colors.ENDC}",
        f"{Colors.YELLOW}• Ensure the Metal backend is properly installed: pip install -e .[metal]{Colors.ENDC}"
    ]
    
    # Add M3-specific tip if not on M3
    if not (has_generation and generation_name and "M3" in generation_name):
        tips.append(f"{Colors.YELLOW}• For optimal performance, consider using an M3 or newer Mac{Colors.ENDC}")
    
    return tips

for tip in get_troubleshooting_tips():
    print(tip)

# Part 6: Simple Example
print_section("Simple Example")

print("""
Let's put everything together with a simple vector addition example:
""")

print_code_example("""
import os
os.environ["TRITON_BACKEND"] = "metal"

import triton
import triton.language as tl
import mlx.core as mx
import numpy as np

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Grid-stride loop
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform operation
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def main():
    # Create input data
    n_elements = 1024 * 1024
    x = mx.random.normal((n_elements,))
    y = mx.random.normal((n_elements,))
    output = mx.zeros((n_elements,))
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, 1024),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    # Verify result
    mx_output = x + y
    diff = mx.abs(output - mx_output).max()
    print(f"Max difference: {diff}")

if __name__ == "__main__":
    main()
""")

print(f"\n{Colors.GREEN}This tutorial has guided you through checking system compatibility with the Triton Metal backend, understanding memory layout optimizations, and leveraging M3-specific features if available.{Colors.ENDC}")

print(f"\n{Colors.BLUE}For more advanced examples and details, check the full documentation and examples in the Triton Metal backend repository.{Colors.ENDC}")

print_section("Summary")

# Create a summary based on the checks performed
all_checks_passed = is_macos_compatible and is_apple_silicon and has_mlx
if has_coalesced and is_consistent:
    memory_status = f"{Colors.GREEN}Supported{Colors.ENDC}"
else:
    memory_status = f"{Colors.YELLOW}Limited{Colors.ENDC}"

if has_generation and generation_name:
    m_series = generation_name
    if "M3" in generation_name:
        m_series_status = f"{Colors.GREEN}Optimal{Colors.ENDC}"
    else:
        m_series_status = f"{Colors.YELLOW}Compatible{Colors.ENDC}"
else:
    m_series = "Unknown"
    m_series_status = f"{Colors.YELLOW}Unknown{Colors.ENDC}"

print(f"""
Based on our checks, your system has:
- macOS Version: {platform.mac_ver()[0]} - {"✓" if is_macos_compatible else "✗"}
- Apple Silicon: {"✓" if is_apple_silicon else "✗"}
- M-series Chip: {m_series} - {m_series_status}
- MLX Installed: {"✓" if has_mlx else "✗"}
- COALESCED Layout: {memory_status}

Overall compatibility: {"✓" if all_checks_passed else "✗"}
""")

if all_checks_passed:
    print(f"{Colors.GREEN}Your system is compatible with the Triton Metal backend.{Colors.ENDC}")
else:
    print(f"{Colors.YELLOW}Your system may have limited compatibility with the Triton Metal backend.{Colors.ENDC}")
    print("Please address the issues marked with ✗ above.")

# Create a function to run this tutorial independently
def run_tutorial():
    """Main function to run this tutorial independently"""
    print(f"{Colors.BLUE}{Colors.BOLD}Metal Backend Compatibility Tutorial{Colors.ENDC}")
    print(f"{Colors.BLUE}This tutorial checks your system compatibility and demonstrates usage of the Metal backend.{Colors.ENDC}\n")
    # The main script has already executed, so we don't need to do anything here
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(run_tutorial()) 