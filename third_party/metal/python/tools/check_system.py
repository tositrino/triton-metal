#!/usr/bin/env python
"""
System Compatibility Check for Triton Metal Backend

This script checks if your system is compatible with the Triton Metal backend
and supports the COALESCED memory layout optimization for reduction operations.
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

# Define colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message: str, status: bool, details: Optional[str] = None):
    """Print status message with color"""
    status_str = f"{Colors.GREEN}✓{Colors.ENDC}" if status else f"{Colors.RED}✗{Colors.ENDC}"
    print(f"{status_str} {message}")
    if details and not status:
        print(f"  {Colors.YELLOW}{details}{Colors.ENDC}")

def print_section(title: str):
    """Print section title"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}=== {title} ==={Colors.ENDC}")

def check_macos_version() -> Tuple[bool, str]:
    """Check macOS version (should be 13.5+)"""
    # Get macOS version
    macos_version = platform.mac_ver()[0]
    
    # Parse major and minor version
    try:
        major, minor, patch = map(int, macos_version.split('.'))
        is_compatible = (major > 13) or (major == 13 and minor >= 5)
        details = f"Found macOS {macos_version}"
        if not is_compatible:
            details += " (required: 13.5+)"
        return is_compatible, details
    except Exception as e:
        return False, f"Could not determine macOS version: {e}"

def check_apple_silicon() -> Tuple[bool, str]:
    """Check if running on Apple Silicon"""
    is_apple_silicon = platform.machine() == 'arm64'
    details = "Apple Silicon detected" if is_apple_silicon else "Intel CPU detected (Apple Silicon required)"
    return is_apple_silicon, details

def check_m_series_generation() -> Tuple[bool, Optional[str], str]:
    """Check for M-series generation and compatibility"""
    try:
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
        return False, None, "Metal hardware detection module not found"
    except Exception as e:
        return False, None, f"Error during hardware generation detection: {e}"

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

def check_metal_hardware_detection() -> Tuple[bool, str]:
    """Check if Metal hardware detection is available"""
    try:
        from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
        
        # Check if chip_generation is available
        chip_gen = getattr(hardware_capabilities, "chip_generation", None)
        if chip_gen is not None:
            chip_name = chip_gen.name
            details = f"Detected {chip_name} hardware"
            return True, details
        else:
            return False, "Hardware detection module found but couldn't detect chip generation"
    except ImportError:
        return False, "Metal hardware detection module not found"
    except Exception as e:
        return False, f"Error during hardware detection: {e}"

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

def check_reduction_pattern() -> Tuple[bool, str]:
    """Check if ReductionLayoutPattern is properly implemented"""
    try:
        from memory_layout_optimizer import ReductionLayoutPattern, MemoryLayout
        
        # Create pattern instance
        pattern = ReductionLayoutPattern()
        
        # Check if it returns COALESCED as optimal layout
        test_shape = [1024]
        layout = pattern.get_optimal_layout(test_shape, None)
        
        if layout == MemoryLayout.COALESCED:
            details = "ReductionLayoutPattern correctly returns COALESCED as optimal layout"
            return True, details
        else:
            return False, f"Pattern returned {layout} instead of COALESCED"
    except ImportError:
        return False, "memory_layout_optimizer module not found"
    except Exception as e:
        return False, f"Error checking reduction pattern: {e}"

def check_memory_manager_implementation() -> Tuple[bool, str]:
    """Check if _optimize_reduction_memory method applies COALESCED layout"""
    try:
        from metal_memory_manager import get_metal_memory_manager, MemoryLayout
        
        # Get memory manager instance
        memory_manager = get_metal_memory_manager()
        
        # Create a test reduction operation
        test_op = {
            "type": "tt.reduce",
            "input_shapes": [[1024]],
            "args": {"axis": 0}
        }
        
        # Apply optimization
        optimized_op = memory_manager._optimize_reduction_memory(test_op)
        
        # Check if COALESCED layout was applied
        if "execution_parameters" in optimized_op:
            params = optimized_op["execution_parameters"]
            
            if "memory_layout" in params:
                layout_value = params["memory_layout"]
                
                if layout_value == MemoryLayout.COALESCED.value:
                    details = "Memory manager correctly applies COALESCED layout to reductions"
                    return True, details
                else:
                    return False, f"Memory manager applied layout {layout_value} instead of COALESCED"
            else:
                return False, "No memory_layout in execution_parameters"
        else:
            return False, "No execution_parameters in optimized operation"
    except ImportError:
        return False, "metal_memory_manager module not found"
    except Exception as e:
        return False, f"Error checking memory manager implementation: {e}"

def check_m3_specific_optimizations() -> Tuple[bool, str]:
    """Check for M3-specific optimizations in the hardware optimizer"""
    try:
        from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
        
        # Check if M3 is defined in AppleSiliconGeneration
        has_m3_enum = hasattr(AppleSiliconGeneration, "M3")
        
        # Check for M3-specific optimization functions
        hardware_optimizer_attrs = dir(hardware_capabilities)
        m3_specific_attrs = [attr for attr in hardware_optimizer_attrs if "m3" in attr.lower()]
        
        if has_m3_enum and m3_specific_attrs:
            details = f"Found M3-specific optimizations: {', '.join(m3_specific_attrs)}"
            return True, details
        elif has_m3_enum:
            return True, "M3 generation defined but no specific optimizations found"
        else:
            return False, "No M3-specific support detected in hardware optimizer"
    except ImportError:
        return False, "Failed to import metal_hardware_optimizer module"
    except Exception as e:
        return False, f"Error checking M3 optimizations: {e}"

def get_troubleshooting_tips(failed_checks: Dict[str, str]) -> List[str]:
    """Generate troubleshooting tips for failed checks"""
    tips = []
    
    if "macOS 13.5+ required" in failed_checks:
        tips.append(f"{Colors.YELLOW}• Update your macOS to version 13.5 or newer. "
                   f"Current version: {platform.mac_ver()[0]}{Colors.ENDC}")
    
    if "Apple Silicon required" in failed_checks:
        tips.append(f"{Colors.YELLOW}• This backend requires an Apple Silicon Mac (M1, M2, or M3 series).{Colors.ENDC}")
    
    if "MLX installed" in failed_checks:
        tips.append(f"{Colors.YELLOW}• Install MLX package: pip install mlx{Colors.ENDC}")
    
    if "Metal hardware detection" in failed_checks:
        tips.append(f"{Colors.YELLOW}• Ensure the metal_hardware_optimizer module is installed.{Colors.ENDC}")
    
    if "COALESCED memory layout defined" in failed_checks:
        tips.append(f"{Colors.YELLOW}• Ensure the metal_memory_manager module is installed with COALESCED layout support.{Colors.ENDC}")
    
    if "Consistent COALESCED definition" in failed_checks:
        tips.append(f"{Colors.YELLOW}• Reinstall the Metal backend to ensure consistent definitions across modules.{Colors.ENDC}")
    
    if "ReductionLayoutPattern implementation" in failed_checks or "Memory manager implementation" in failed_checks:
        tips.append(f"{Colors.YELLOW}• Update to the latest version of the Metal backend.{Colors.ENDC}")
    
    # Add general tip if any Metal backend components failed
    any_metal_failed = any(check in failed_checks for check in [
        "Metal hardware detection", "COALESCED memory layout defined", 
        "Consistent COALESCED definition", "ReductionLayoutPattern implementation",
        "Memory manager implementation"
    ])
    
    if any_metal_failed:
        tips.append(f"{Colors.YELLOW}• Try reinstalling the Triton Metal backend: pip install -e .[metal]{Colors.ENDC}")
    
    return tips

def main():
    """Run all compatibility checks"""
    print(f"{Colors.BLUE}{Colors.BOLD}Triton Metal Backend Compatibility Check{Colors.ENDC}")
    print(f"{Colors.BLUE}This script checks if your system supports the COALESCED memory layout for reduction operations.{Colors.ENDC}\n")
    
    all_checks_passed = True
    failed_checks = {}
    
    # Check system requirements
    print_section("System Requirements")
    
    # Check macOS version
    is_macos_compatible, macos_details = check_macos_version()
    print_status("macOS 13.5+ required", is_macos_compatible, macos_details)
    if not is_macos_compatible:
        failed_checks["macOS 13.5+ required"] = macos_details
    all_checks_passed = all_checks_passed and is_macos_compatible
    
    # Check if running on Apple Silicon
    is_apple_silicon, silicon_details = check_apple_silicon()
    print_status("Apple Silicon required", is_apple_silicon, silicon_details)
    if not is_apple_silicon:
        failed_checks["Apple Silicon required"] = silicon_details
    all_checks_passed = all_checks_passed and is_apple_silicon
    
    # Check M-series generation
    if is_apple_silicon:
        has_generation, generation_name, generation_details = check_m_series_generation()
        print_status("Apple Silicon generation check", has_generation, generation_details)
        if not has_generation:
            failed_checks["Apple Silicon generation check"] = generation_details
        
        # This doesn't affect overall compatibility but shows more details
        if has_generation and generation_name:
            has_m3_optimizations, m3_details = check_m3_specific_optimizations()
            is_m3 = generation_name.startswith("M3") if generation_name else False
            
            if is_m3:
                print_status("M3-specific optimizations", has_m3_optimizations, m3_details)
                if not has_m3_optimizations:
                    failed_checks["M3-specific optimizations"] = m3_details
            else:
                print(f"  {Colors.YELLOW}Note: M3 or newer chip recommended for optimal performance{Colors.ENDC}")
    
    # Check required packages
    print_section("Required Packages")
    
    # Check for MLX
    has_mlx, mlx_details = check_package_installed("mlx.core")
    print_status("MLX installed", has_mlx, mlx_details)
    if not has_mlx:
        failed_checks["MLX installed"] = mlx_details
    all_checks_passed = all_checks_passed and has_mlx
    
    # Check Metal backend components
    print_section("Metal Backend Components")
    
    # Check hardware detection
    has_hardware_detection, hardware_details = check_metal_hardware_detection()
    print_status("Metal hardware detection", has_hardware_detection, hardware_details)
    if not has_hardware_detection:
        failed_checks["Metal hardware detection"] = hardware_details
    all_checks_passed = all_checks_passed and has_hardware_detection
    
    # Check for COALESCED memory layout
    has_coalesced, coalesced_details = check_coalesced_memory_layout()
    print_status("COALESCED memory layout defined", has_coalesced, coalesced_details)
    if not has_coalesced:
        failed_checks["COALESCED memory layout defined"] = coalesced_details
    all_checks_passed = all_checks_passed and has_coalesced
    
    # Check implementation components
    print_section("Implementation Verification")
    
    # Check layout consistency between modules
    is_consistent, consistency_details = check_optimizer_consistency()
    print_status("Consistent COALESCED definition", is_consistent, consistency_details)
    if not is_consistent:
        failed_checks["Consistent COALESCED definition"] = consistency_details
    all_checks_passed = all_checks_passed and is_consistent
    
    # Check ReductionLayoutPattern
    has_reduction_pattern, pattern_details = check_reduction_pattern()
    print_status("ReductionLayoutPattern implementation", has_reduction_pattern, pattern_details)
    if not has_reduction_pattern:
        failed_checks["ReductionLayoutPattern implementation"] = pattern_details
    all_checks_passed = all_checks_passed and has_reduction_pattern
    
    # Check memory manager implementation
    has_memory_manager_impl, memory_details = check_memory_manager_implementation()
    print_status("Memory manager implementation", has_memory_manager_impl, memory_details)
    if not has_memory_manager_impl:
        failed_checks["Memory manager implementation"] = memory_details
    all_checks_passed = all_checks_passed and has_memory_manager_impl
    
    # Print summary
    print_section("Summary")
    if all_checks_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed! Your system supports COALESCED memory layout for reduction operations.{Colors.ENDC}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed. Your system may not fully support COALESCED memory layout.{Colors.ENDC}")
        print(f"{Colors.YELLOW}Please address the issues marked with ✗ above.{Colors.ENDC}\n")
        
        # Print troubleshooting tips
        tips = get_troubleshooting_tips(failed_checks)
        if tips:
            print(f"{Colors.BLUE}{Colors.BOLD}Troubleshooting Tips:{Colors.ENDC}")
            for tip in tips:
                print(tip)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 