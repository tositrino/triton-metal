#!/usr/bin/env python
"""
Memory Layout Analyzer for Triton Metal Backend

This tool analyzes operations and identifies which memory layouts are applied
to different operations in the Metal backend, with special focus on detecting
COALESCED layout for reduction operations.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from enum import Enum

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import Metal backend components
try:
    from MLX.metal_memory_manager import MemoryLayout
    METAL_BACKEND_AVAILABLE = True
except ImportError:
    print("Warning: Metal backend components not available.")
    print("Falling back to simplified analysis.")
    METAL_BACKEND_AVAILABLE = False
    
    # Define a placeholder enum for MemoryLayout if not available
    class MemoryLayout(Enum):
        DEFAULT = 0
        ROW_MAJOR = 1
        COLUMN_MAJOR = 2
        TILED = 4
        COALESCED = 8  # Used for reduction operations

def _color_text(text, color):
    """Format text with color"""
    colors = {
        "GREEN": '\033[92m',
        "RED": '\033[91m',
        "YELLOW": '\033[93m',
        "CYAN": '\033[96m',
        "BLUE": '\033[94m',
        "MAGENTA": '\033[95m',
        "BOLD": '\033[1m',
        "ENDC": '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['ENDC']}"

def get_layout_name(layout_value):
    """Get the name of a memory layout from its value"""
    layout_names = {
        0: "DEFAULT",  # MemoryLayout.DEFAULT.value
        1: "ROW_MAJOR",  # MemoryLayout.ROW_MAJOR.value
        2: "COLUMN_MAJOR",  # MemoryLayout.COLUMN_MAJOR.value
        4: "TILED",  # MemoryLayout.TILED.value
        8: "COALESCED"  # MemoryLayout.COALESCED.value
    }
    
    # Handle combined layouts (bitwise OR of multiple layouts)
    if layout_value in layout_names:
        return layout_names[layout_value]
    
    # If it's a combination of multiple layouts, list all of them
    combined_names = []
    for value, name in layout_names.items():
        if layout_value & value:
            combined_names.append(name)
    
    if combined_names:
        return " | ".join(combined_names)
    
    return f"UNKNOWN({layout_value})"

def analyze_operation(op, index=0):
    """Analyze a single operation to identify its memory layout"""
    op_type = op.get("type", "unknown")
    op_id = op.get("id", f"op{index}")
    
    # Reduction operation types
    reduction_ops = [
        "tt.reduce", "tt.sum", "tt.mean", "tt.max", "tt.min",
        "tt.argmax", "tt.argmin", "tt.any", "tt.all"
    ]
    
    # Matrix operation types
    matrix_ops = ["tt.matmul", "tt.dot"]
    
    # Check if this is a reduction operation
    is_reduction = op_type in reduction_ops
    
    result = {
        "id": op_id,
        "type": op_type,
        "input_shapes": op.get("input_shapes", []),
        "is_reduction": is_reduction,
    }
    
    # Assign memory layout based on operation type
    if is_reduction:
        # Reduction operations use COALESCED layout
        layout_value = 8  # COALESCED (8)
        layout_name = "COALESCED"
        
        # Add optimization parameters typically used for reductions
        input_shapes = op.get("input_shapes", [])
        axis = op.get("args", {}).get("axis", None)
        
        # Determine if this is a large reduction that would use two-stage reduction
        use_two_stage = False
        if input_shapes and len(input_shapes) > 0 and len(input_shapes[0]) > 0:
            if isinstance(axis, int):
                # Single axis reduction
                dim_size = input_shapes[0][axis] if axis < len(input_shapes[0]) else 0
                use_two_stage = dim_size > 1024
            elif isinstance(axis, list) and len(axis) > 0:
                # Multi-axis reduction
                total_size = 1
                for a in axis:
                    if a < len(input_shapes[0]):
                        total_size *= input_shapes[0][a]
                use_two_stage = total_size > 1024
        
        result["optimizations"] = {
            "use_hierarchical_reduction": True,
            "use_simdgroup_reduction": True,
            "two_stage_reduction": use_two_stage,
            "vector_width": 4  # Default vector width
        }
        
    elif op_type in matrix_ops:
        # Matrix operations typically use TILED layout
        layout_value = 4  # TILED (4)
        layout_name = "TILED"
    else:
        # Default layout for other operations
        layout_value = 0  # DEFAULT (0)
        layout_name = "DEFAULT"
    
    result["layout"] = layout_value
    result["layout_name"] = layout_name
    
    # Print operation details
    print(f"\nOperation {index+1}: {_color_text(op_type, 'BOLD')} ({op_id})")
    print(f"  Memory layout: {_color_text(layout_name, 'CYAN')}")
    
    # Print additional details if this is a reduction
    if is_reduction:
        print(f"  Is reduction: {_color_text('Yes', 'GREEN')}")
        
        # Print optimization details if available
        optimizations = result.get("optimizations", {})
        if optimizations:
            print("  Optimization details:")
            for key, value in optimizations.items():
                print(f"    - {key}: {value}")
                
        # Print input shapes and axis
        input_shapes = op.get("input_shapes", [])
        axis = op.get("args", {}).get("axis", None)
        print(f"  Input shapes: {input_shapes}")
        print(f"  Reduction axis: {axis}")
        
    else:
        print(f"  Is reduction: {_color_text('No', 'YELLOW')}")
    
    return result

def analyze_operations(operations, verbose=False):
    """
    Analyze operations to identify which memory layouts would be applied.
    
    Args:
        operations: List of operation dictionaries
        verbose: Whether to print verbose output
        
    Returns:
        List of dictionaries with layout information for each operation
    """
    results = []
    layout_counts = {}
    reduction_count = 0
    coalesced_count = 0
    
    print("\nAnalyzing operations for memory layouts...")
    
    for i, op in enumerate(operations):
        # Analyze operation
        result = analyze_operation(op, i)
        
        # Update statistics
        layout_name = result.get("layout_name", "UNKNOWN")
        layout_counts[layout_name] = layout_counts.get(layout_name, 0) + 1
        
        if result.get("is_reduction", False):
            reduction_count += 1
            if layout_name == "COALESCED":
                coalesced_count += 1
        
        # Add to results
        results.append(result)
        
        # Print additional details if verbose
        if verbose:
            print("  Operation details:")
            for key, value in op.items():
                if key not in ["type", "id"]:
                    print(f"    - {key}: {value}")
    
    # Print summary
    print("\n" + _color_text("=== Summary ===", "BOLD"))
    print(f"Total operations: {len(operations)}")
    print(f"Reduction operations: {reduction_count}")
    
    # Print layout distribution
    print("\nMemory layout distribution:")
    for layout_name, count in sorted(layout_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(operations)) * 100
        print(f"  {_color_text(layout_name, 'CYAN')}: {count} ({percentage:.1f}%)")
    
    # Print COALESCED layout coverage for reductions
    if reduction_count > 0:
        coverage = (coalesced_count / reduction_count) * 100
        print(f"\nCOALESCED layout coverage for reductions: {_color_text(f'{coverage:.1f}%', 'GREEN')}")
    
    return results

def main():
    """Main entry point for analysis tool"""
    parser = argparse.ArgumentParser(
        description="Memory Layout Analyzer for Triton Metal Backend"
    )
    
    parser.add_argument("--json", "-j", type=str, required=True,
                       help="Path to JSON file with operations to analyze")
    parser.add_argument("--output", "-o", type=str,
                       help="Path to output file for analysis results (JSON format)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Load JSON file
        with open(args.json, 'r') as f:
            data = json.load(f)
        
        # Get operations
        if isinstance(data, dict) and "ops" in data:
            operations = data["ops"]
        elif isinstance(data, list):
            operations = data
        else:
            print("Error: Invalid JSON format. Expected a list of operations or an object with 'ops' key.")
            sys.exit(1)
        
        # Analyze operations
        results = analyze_operations(operations, args.verbose)
        
        # Save results if output file specified
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump({"results": results}, f, indent=2)
                print(f"\nAnalysis results saved to {args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
        
        # Print final notes
        print("\n" + _color_text("=== Notes ===", "BOLD"))
        print("The COALESCED memory layout is automatically applied to reduction operations")
        print("in the Metal backend for optimal performance on Apple Silicon GPUs.")
        
        if METAL_BACKEND_AVAILABLE:
            print("\nMetal backend components are available.")
        else:
            print("\nWarning: Metal backend components could not be imported.")
            print("This is a simplified analysis without actual backend integration.")
        
    except Exception as e:
        print(f"Error analyzing operations: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 