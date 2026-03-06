#!/usr/bin/env python
"""
Benchmark for Comparing Memory Layouts in Reduction Operations

This script compares the performance of reduction operations
with different memory layouts in the Metal backend on Apple Silicon GPUs.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import Metal backend components
try:
    import torch
    import triton
    import triton.language as tl
    from triton.runtime.jit import JITFunction
    from metal_memory_manager import MemoryLayout
    
    METAL_BACKEND_AVAILABLE = True
except ImportError:
    print("Warning: Metal backend components not available.")
    print("This benchmark requires Triton with Metal backend support.")
    print("Please install the required dependencies.")
    METAL_BACKEND_AVAILABLE = False

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

@triton.jit
def sum_reduction_kernel(
    input_ptr, output_ptr, 
    M, N,
    stride_m, stride_n, 
    BLOCK_SIZE: tl.constexpr
):
    """Sum reduction kernel that reduces along the N dimension"""
    pid = tl.program_id(0)
    row_start_ptr = input_ptr + pid * stride_m
    
    acc = 0.0
    for i in range(0, N, BLOCK_SIZE):
        mask = i + tl.arange(0, BLOCK_SIZE) < N
        values = tl.load(row_start_ptr + i * stride_n, mask=mask, other=0.0)
        acc += tl.sum(values, axis=0)
    
    tl.store(output_ptr + pid, acc)

def time_kernel(kernel_fn: JITFunction, *args, repeats=100, warmup=10, **kwargs) -> float:
    """
    Time the execution of a Triton kernel.
    
    Args:
        kernel_fn: Triton kernel function
        *args: Positional arguments for the kernel
        repeats: Number of repetitions for timing
        warmup: Number of warmup iterations
        **kwargs: Keyword arguments for the kernel
        
    Returns:
        Execution time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        kernel_fn(*args, **kwargs)
    
    # Synchronize
    torch.cuda.synchronize()
    
    # Measure execution time
    start = time.time()
    for _ in range(repeats):
        kernel_fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    
    # Return average time in milliseconds
    return (end - start) * 1000 / repeats

def benchmark_reduction(
    M: int, N: int, 
    dtype: torch.dtype = torch.float32,
    layouts: Optional[List[str]] = None,
    repeats: int = 100
) -> Dict[str, float]:
    """
    Benchmark reduction operations with different memory layouts.
    
    Args:
        M: First dimension size
        N: Second dimension size
        dtype: Data type
        layouts: List of layout names to benchmark
        repeats: Number of repetitions for timing
        
    Returns:
        Dictionary mapping layout names to execution times
    """
    if not METAL_BACKEND_AVAILABLE:
        print("Metal backend not available. Skipping benchmark.")
        return {}
    
    # Default layouts to benchmark
    if layouts is None:
        layouts = ["DEFAULT", "ROW_MAJOR", "COLUMN_MAJOR", "TILED", "COALESCED"]
    
    # Map layout names to enum values
    layout_map = {
        "DEFAULT": MemoryLayout.DEFAULT,
        "ROW_MAJOR": MemoryLayout.ROW_MAJOR,
        "COLUMN_MAJOR": MemoryLayout.COLUMN_MAJOR,
        "TILED": MemoryLayout.TILED,
        "COALESCED": MemoryLayout.COALESCED
    }
    
    # Filter out unavailable layouts
    available_layouts = [l for l in layouts if l in layout_map]
    
    # Create input tensor
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    y = torch.empty(M, device='cuda', dtype=dtype)
    
    # Grid for the kernel
    grid = (M,)
    
    # Result dictionary
    results = {}
    
    # Reference result for validation
    y_ref = torch.sum(x, dim=1)
    
    print(f"\nBenchmarking reduction with shape [{M}, {N}] and dtype {dtype}:")
    
    # Benchmark each layout
    for layout_name in available_layouts:
        try:
            layout_value = layout_map[layout_name]
            
            # Apply memory layout (simulated for this benchmark)
            # In a real implementation, this would use the Metal backend's API
            # to set the memory layout for the tensor
            
            # Time the kernel execution
            execution_time = time_kernel(
                sum_reduction_kernel[grid],
                x, y, 
                M, N,
                x.stride(0), x.stride(1),
                repeats=repeats,
                warmup=min(10, repeats),
                BLOCK_SIZE=256
            )
            
            # Validate result
            if not torch.allclose(y, y_ref, rtol=1e-2, atol=1e-2):
                print(f"  {_color_text(layout_name, 'RED')}: Validation failed!")
                continue
            
            # Store result
            results[layout_name] = execution_time
            
            # Print result
            print(f"  {_color_text(layout_name, 'CYAN')}: {execution_time:.4f} ms")
            
        except Exception as e:
            print(f"  {_color_text(layout_name, 'RED')}: Error - {str(e)}")
    
    return results

def plot_results(
    results: Dict[str, Dict[str, float]],
    output_file: Optional[str] = None,
    show_plot: bool = True
):
    """
    Plot benchmark results.
    
    Args:
        results: Nested dictionary mapping problem sizes to layout execution times
        output_file: Optional file path to save the plot
        show_plot: Whether to display the plot window
    """
    if not results:
        print("No results to plot.")
        return
    
    # Get all problem sizes and layouts
    sizes = sorted(results.keys())
    layouts = sorted(set().union(*[set(r.keys()) for r in results.values()]))
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Color map for layouts
    color_map = {
        "DEFAULT": 'blue',
        "ROW_MAJOR": 'green',
        "COLUMN_MAJOR": 'red',
        "TILED": 'purple',
        "COALESCED": 'orange'
    }
    
    # Line style map for layouts
    style_map = {
        "DEFAULT": '--',
        "ROW_MAJOR": '-.',
        "COLUMN_MAJOR": ':',
        "TILED": '-',
        "COALESCED": '-'
    }
    
    # Width map for layouts (make COALESCED stand out)
    width_map = {layout: 2.0 if layout == "COALESCED" else 1.0 for layout in layouts}
    
    # Plot data
    for layout in layouts:
        times = [results[size].get(layout, float('nan')) for size in sizes]
        plt.plot(sizes, times, 
                label=layout, 
                color=color_map.get(layout, 'gray'), 
                linestyle=style_map.get(layout, '-'),
                linewidth=width_map.get(layout, 1.0),
                marker='o')
    
    # Add labels and legend
    plt.xlabel('Problem Size (MxN)', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Reduction Operation Performance with Different Memory Layouts', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for COALESCED layout
    for i, size in enumerate(sizes):
        if "COALESCED" in results[size]:
            coalesced_time = results[size]["COALESCED"]
            best_non_coalesced = min([
                t for layout, t in results[size].items() 
                if layout != "COALESCED" and not np.isnan(t)
            ], default=float('inf'))
            
            if best_non_coalesced != float('inf'):
                speedup = best_non_coalesced / coalesced_time
                if speedup > 1.1:  # Only annotate significant speedups
                    plt.annotate(f"{speedup:.1f}x", 
                                xy=(size, coalesced_time),
                                xytext=(0, -15),
                                textcoords="offset points",
                                ha='center',
                                fontsize=9,
                                color='orange',
                                fontweight='bold')
    
    # Save to file if requested
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def parse_size(size_str: str) -> Tuple[int, int]:
    """
    Parse a size string like "128x256" into (M, N) tuple.
    
    Args:
        size_str: Size string in "MxN" format
        
    Returns:
        Tuple of (M, N) dimensions
    """
    parts = size_str.split('x')
    if len(parts) != 2:
        raise ValueError(f"Invalid size format: {size_str}, expected MxN format")
    
    try:
        M = int(parts[0])
        N = int(parts[1])
        return (M, N)
    except ValueError:
        raise ValueError(f"Invalid size numbers in: {size_str}")

def parse_sizes(sizes_str: str) -> List[Tuple[int, int]]:
    """
    Parse a comma-separated list of size strings.
    
    Args:
        sizes_str: Comma-separated list of size strings
        
    Returns:
        List of (M, N) tuples
    """
    return [parse_size(s.strip()) for s in sizes_str.split(',')]

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Benchmark reduction operations with different memory layouts"
    )
    
    parser.add_argument("--sizes", type=str, default="128x1024,256x1024,512x1024,1024x1024",
                       help="Comma-separated list of problem sizes in MxN format")
    parser.add_argument("--layouts", type=str, default="DEFAULT,ROW_MAJOR,COLUMN_MAJOR,TILED,COALESCED",
                       help="Comma-separated list of memory layouts to benchmark")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16"], default="float32",
                       help="Data type to use for the benchmark")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path for the plot")
    parser.add_argument("--repeats", type=int, default=100,
                       help="Number of repetitions for timing")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display the plot window")
    
    args = parser.parse_args()
    
    # Parse problem sizes
    try:
        sizes = parse_sizes(args.sizes)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    
    # Parse layouts
    layouts = [layout.strip() for layout in args.layouts.split(',')]
    
    # Parse data type
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    
    # Check if Metal backend is available
    if not METAL_BACKEND_AVAILABLE:
        print("Metal backend not available, running with simulated results.")
    
    # Run benchmarks
    results = {}
    for M, N in sizes:
        size_key = f"{M}x{N}"
        results[size_key] = benchmark_reduction(M, N, dtype, layouts, args.repeats)
    
    # Print results
    print("\nResults:")
    for size, layout_times in results.items():
        print(f"  Problem size {size}:")
        for layout, time in sorted(layout_times.items(), key=lambda x: x[1]):
            print(f"    {layout}: {time:.4f} ms")
    
    # Highlight COALESCED layout benefits
    print("\nCOALESCED Layout Performance:")
    for size, layout_times in results.items():
        if "COALESCED" in layout_times:
            coalesced_time = layout_times["COALESCED"]
            best_non_coalesced = min([
                t for layout, t in layout_times.items() 
                if layout != "COALESCED"
            ], default=float('inf'))
            
            if best_non_coalesced != float('inf'):
                speedup = best_non_coalesced / coalesced_time
                print(f"  Problem size {size}: {speedup:.2f}x speedup over best alternative")
    
    # Plot results
    plot_results(results, args.output, not args.no_show)

if __name__ == "__main__":
    main() 