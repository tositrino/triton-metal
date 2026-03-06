#!/usr/bin/env python
"""
Simple Memory Layout Analyzer for Reduction Operations

This tool analyzes operations to identify which reduction operations
would use the COALESCED memory layout in the Metal backend.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional, Union
from enum import Enum

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import Metal backend components
try:
    from metal_memory_manager import MemoryLayout
    METAL_BACKEND_AVAILABLE = True
except ImportError:
    print("Warning: Could not import Metal backend modules.")
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

def parse_axis(axis_str):
    """Parse axis string into an integer or list of integers"""
    if not axis_str:
        return None
    
    # Strip any whitespace and brackets
    clean_str = axis_str.strip()
    
    # Handle case where the axis is wrapped in brackets
    if clean_str.startswith('[') and clean_str.endswith(']'):
        clean_str = clean_str[1:-1]
    
    # Check if it's a multi-axis specification (comma-separated)
    if ',' in clean_str:
        try:
            # Split and convert to integers
            return [int(x.strip()) for x in clean_str.split(',')]
        except ValueError as e:
            raise ValueError(f"Error parsing multi-axis '{axis_str}': {e}")
    
    # Single axis
    try:
        return int(clean_str)
    except ValueError as e:
        raise ValueError(f"Error parsing axis '{axis_str}': {e}")

def parse_shape(shape_str):
    """Parse shape string into a list of integers"""
    if not shape_str:
        return []
    
    # Strip any whitespace
    clean_str = shape_str.strip()
    
    # Handle case where the shape is wrapped in brackets
    if clean_str.startswith('[') and clean_str.endswith(']'):
        clean_str = clean_str[1:-1]
    
    try:
        # Split and convert to integers
        return [int(dim.strip()) for dim in clean_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Error parsing shape '{shape_str}': {e}")

def parse_operation_string(op_str):
    """
    Parse an operation string into a dictionary.
    
    The format should be: "op_type:shape:axis"
    Example: "tt.reduce:[1024,512]:1"
    
    Args:
        op_str: Operation string to parse
        
    Returns:
        Dictionary with operation information
    """
    try:
        parts = op_str.split(':')
        
        if len(parts) < 2:
            raise ValueError("Operation string should have at least two parts: op_type:shape[:axis]")
        
        op_type = parts[0].strip()
        shape_str = parts[1].strip()
        
        # Parse shape
        shape = parse_shape(shape_str)
        
        # Parse axis (if provided)
        axis = None
        if len(parts) > 2:
            axis_str = parts[2].strip()
            axis = parse_axis(axis_str)
        
        # Create operation dictionary
        op = {
            "type": op_type,
            "input_shapes": [shape],
            "args": {} if axis is None else {"axis": axis}
        }
        
        return op
    
    except Exception as e:
        print(f"Error parsing operation string: {e}")
        print("Expected format: op_type:shape:axis")
        print("Example: tt.reduce:[1024,512]:1")
        print("Example with multi-axis: tt.mean:[32,64,128]:[0,1]")
        sys.exit(1)

def analyze_operation(op):
    """
    Analyze an operation to determine if it would use the COALESCED memory layout.
    
    Args:
        op: Operation dictionary
        
    Returns:
        Dictionary with analysis results
    """
    op_type = op.get("type", "unknown")
    
    # Reduction operation types
    reduction_ops = [
        "tt.reduce", "tt.sum", "tt.mean", "tt.max", "tt.min",
        "tt.argmax", "tt.argmin", "tt.any", "tt.all"
    ]
    
    # Check if this is a reduction operation
    is_reduction = op_type in reduction_ops
    
    result = {
        "type": op_type,
        "input_shapes": op.get("input_shapes", []),
        "is_reduction": is_reduction,
        "memory_layout": "COALESCED" if is_reduction else "DEFAULT",
        "memory_layout_value": MemoryLayout.COALESCED.value if is_reduction else MemoryLayout.DEFAULT.value
    }
    
    # Add details for reductions
    if is_reduction:
        axis = op.get("args", {}).get("axis", None)
        result["axis"] = axis
        
        # Add optimization parameters
        input_shapes = op.get("input_shapes", [])
        
        # Determine if this is a large reduction that would use two-stage reduction
        use_two_stage = False
        if input_shapes and len(input_shapes) > 0 and len(input_shapes[0]) > 0:
            try:
                if isinstance(axis, int):
                    # Single axis reduction
                    if 0 <= axis < len(input_shapes[0]):
                        dim_size = input_shapes[0][axis]
                        use_two_stage = dim_size > 1024
                    else:
                        print(f"Warning: Axis {axis} is out of bounds for shape {input_shapes[0]}")
                elif isinstance(axis, list) and len(axis) > 0:
                    # Multi-axis reduction
                    total_size = 1
                    for a in axis:
                        if 0 <= a < len(input_shapes[0]):
                            total_size *= input_shapes[0][a]
                        else:
                            print(f"Warning: Axis {a} is out of bounds for shape {input_shapes[0]}")
                    use_two_stage = total_size > 1024
            except Exception as e:
                print(f"Warning: Could not determine if two-stage reduction is needed: {e}")
                use_two_stage = False
        
        result["optimizations"] = {
            "use_hierarchical_reduction": True,
            "use_simdgroup_reduction": True,
            "two_stage_reduction": use_two_stage,
            "vector_width": 4  # Default vector width
        }
    
    return result

def format_axis(axis):
    """Format axis for pretty printing"""
    if isinstance(axis, list):
        return f"[{', '.join(str(a) for a in axis)}]"
    return str(axis)

def print_analysis_result(result, verbose=False):
    """Print analysis result in a human-readable format"""
    op_type = result.get("type", "unknown")
    memory_layout = result.get("memory_layout", "unknown")
    is_reduction = result.get("is_reduction", False)
    
    print("\nAnalysis Result:")
    print(f"  Operation Type: {_color_text(op_type, 'BOLD')}")
    print(f"  Memory Layout: {_color_text(memory_layout, 'CYAN')}")
    print(f"  Is Reduction: {_color_text('Yes', 'GREEN') if is_reduction else _color_text('No', 'YELLOW')}")
    
    if is_reduction:
        # Print input shapes
        input_shapes = result.get("input_shapes", [])
        if input_shapes:
            print(f"  Input Shape: {input_shapes[0]}")
        
        # Print axis
        axis = result.get("axis", None)
        if axis is not None:
            print(f"  Reduction Axis: {format_axis(axis)}")
        
        # Print optimization details
        optimizations = result.get("optimizations", {})
        if optimizations:
            print("  Optimization Details:")
            for key, value in optimizations.items():
                print(f"    - {key}: {value}")
    
    if verbose:
        print("\nFull Result:")
        for key, value in result.items():
            if key not in ["type", "memory_layout", "is_reduction"]:
                print(f"  {key}: {value}")

def analyze_json_file(json_path, verbose=False):
    """
    Analyze operations from a JSON file.
    
    Args:
        json_path: Path to JSON file
        verbose: Whether to print verbose output
        
    Returns:
        List of analysis results
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get operations
        if isinstance(data, dict) and "ops" in data:
            operations = data["ops"]
        elif isinstance(data, list):
            operations = data
        else:
            print("Error: Invalid JSON format. Expected a list of operations or an object with 'ops' key.")
            return []
        
        # Analyze each operation
        results = []
        for op in operations:
            result = analyze_operation(op)
            results.append(result)
            print_analysis_result(result, verbose)
        
        # Print summary
        reduction_count = sum(1 for r in results if r.get("is_reduction", False))
        coalesced_count = sum(1 for r in results if r.get("memory_layout", "") == "COALESCED")
        
        print("\n" + _color_text("=== Summary ===", "BOLD"))
        print(f"Total operations: {len(results)}")
        print(f"Reduction operations: {reduction_count}")
        print(f"Operations using COALESCED layout: {coalesced_count}")
        
        return results
    
    except Exception as e:
        print(f"Error analyzing JSON file: {e}")
        return []

def serialize_results(results):
    """
    Prepare results for JSON serialization, handling any non-serializable objects.
    
    Args:
        results: List of analysis results
        
    Returns:
        JSON-serializable version of the results
    """
    def make_serializable(obj):
        if isinstance(obj, Enum):
            return obj.value
        return obj
    
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, dict):
                serializable_result[key] = {k: make_serializable(v) for k, v in value.items()}
            elif isinstance(value, list):
                serializable_result[key] = [make_serializable(item) for item in value]
            else:
                serializable_result[key] = make_serializable(value)
        serializable_results.append(serializable_result)
    
    return serializable_results

def main():
    """Main entry point for the simple analyzer"""
    parser = argparse.ArgumentParser(
        description="Simple Memory Layout Analyzer for Reduction Operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze a single reduction operation:
    %(prog)s --operation "tt.reduce:[1024,512]:1"
    
  Analyze operations from a JSON file:
    %(prog)s --json path/to/operations.json
    
  Analyze with verbose output:
    %(prog)s --operation "tt.mean:[32,64,128]:[0,1]" --verbose
    
  Save analysis results to a file:
    %(prog)s --json path/to/operations.json --output results.json
    
Operation Format:
  op_type:shape:axis
  
  Examples:
    tt.reduce:[1024]:0                # 1D reduction
    tt.sum:[512,512]:1                # Row-wise sum
    tt.mean:[32,64,128]:[0,1]         # Multi-axis reduction
"""
    )
    
    # Define mutually exclusive group for input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--json", "-j", type=str,
                            help="Path to JSON file with operations to analyze")
    input_group.add_argument("--operation", "-o", type=str,
                            help="Operation string to analyze (format: op_type:shape:axis)")
    
    # Other arguments
    parser.add_argument("--output", type=str,
                       help="Path to output file for analysis results (JSON format)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--version", action="store_true",
                       help="Show version information")
    
    args = parser.parse_args()
    
    # Show version information if requested
    if args.version:
        print("Simple Memory Layout Analyzer v1.0.0")
        print(f"Metal Backend Available: {METAL_BACKEND_AVAILABLE}")
        print(f"COALESCED Layout Value: {MemoryLayout.COALESCED.value}")
        sys.exit(0)
    
    try:
        # Initialize results
        results = []
        
        # Process input source
        if args.json:
            # Analyze operations from JSON file
            results = analyze_json_file(args.json, args.verbose)
        elif args.operation:
            # Analyze single operation
            op = parse_operation_string(args.operation)
            result = analyze_operation(op)
            print_analysis_result(result, args.verbose)
            results = [result]
        
        # Save results if output file specified
        if args.output and results:
            try:
                serializable_results = serialize_results(results)
                with open(args.output, 'w') as f:
                    json.dump({"results": serializable_results}, f, indent=2)
                print(f"\nAnalysis results saved to {args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
        
        # Print final notes
        print("\n" + _color_text("=== Notes ===", "BOLD"))
        print("The COALESCED memory layout is automatically applied to reduction operations")
        print("in the Metal backend for optimal performance on Apple Silicon GPUs.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 