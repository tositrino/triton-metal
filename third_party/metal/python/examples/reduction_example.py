#!/usr/bin/env python
"""
Reduction Example - Demonstrating COALESCED Memory Layout

This example shows how the Triton Metal backend automatically applies the 
COALESCED memory layout to reduction operations for optimal performance 
on Apple Silicon GPUs.
"""

import os
import sys
import time
import numpy as np
import argparse

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Triton for kernel definition
import triton
import triton.language as tl

# Define a simple reduction kernel using Triton
@triton.jit
def sum_reduction_kernel(
    x_ptr,  # pointer to input array
    y_ptr,  # pointer to output array
    n_elements,  # number of elements in input
    BLOCK_SIZE: tl.constexpr,  # number of elements per block
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Compute block start/end indices
    block_start = pid * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, n_elements)
    
    # Compute reduction within this block
    sum_value = 0.0
    for i in range(block_start, block_end):
        # Load element
        x_i = tl.load(x_ptr + i)
        # Add to accumulator
        sum_value += x_i
    
    # Store result for this block
    tl.store(y_ptr + pid, sum_value)

@triton.jit
def final_reduction_kernel(
    x_ptr,  # pointer to partial reduction results
    y_ptr,  # pointer to final result
    n_blocks,  # number of blocks to reduce
):
    # Single program to perform final reduction
    sum_value = 0.0
    for i in range(n_blocks):
        x_i = tl.load(x_ptr + i)
        sum_value += x_i
    
    # Store final result
    tl.store(y_ptr, sum_value)

# Function to run reduction using triton
def triton_sum_reduction(x, block_size=1024):
    # Get input size
    n_elements = x.size
    
    # Compute number of blocks
    n_blocks = (n_elements + block_size - 1) // block_size
    
    # Allocate output and partial results
    partial_results = np.zeros(n_blocks, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    
    # Start timing
    start = time.time()
    
    # Run reduction in two stages
    # Stage 1: Reduction within blocks
    sum_reduction_kernel[(n_blocks,)](
        x, partial_results, n_elements, block_size,
    )
    
    # Stage 2: Final reduction across blocks
    final_reduction_kernel[(1,)](
        partial_results, y, n_blocks,
    )
    
    # End timing
    end = time.time()
    elapsed = end - start
    
    # Print performance information
    print(f"Triton reduction (with COALESCED layout): {elapsed:.6f}s")
    print(f"Input size: {n_elements}, Output: {y[0]:.4f}")
    print(f"Expected result: {np.sum(x):.4f}")
    print(f"Difference: {abs(y[0] - np.sum(x)):.6f}")
    
    return y[0], elapsed

# Numpy reduction for comparison
def numpy_sum_reduction(x):
    # Start timing
    start = time.time()
    
    # Perform sum reduction with NumPy
    result = np.sum(x)
    
    # End timing
    end = time.time()
    elapsed = end - start
    
    # Print performance information
    print(f"NumPy reduction: {elapsed:.6f}s")
    print(f"Result: {result:.4f}")
    
    return result, elapsed

# Function to check if MLX is available (Apple Silicon with Metal)
def is_metal_available():
    try:
        import mlx.core
        return True
    except ImportError:
        return False

def main():
    parser = argparse.ArgumentParser(description="Reduction example with COALESCED layout")
    parser.add_argument("--size", type=int, default=10_000_000, help="Size of input array")
    parser.add_argument("--block-size", type=int, default=1024, help="Block size for reduction")
    parser.add_argument("--verify", action="store_true", help="Verify results against NumPy")
    args = parser.parse_args()
    
    # Check if running on Apple Silicon
    if not is_metal_available():
        print("Warning: MLX not available. This example is optimized for Apple Silicon with Metal.")
        print("Performance may not be optimal on this platform.")
    else:
        print("Running on Apple Silicon with Metal backend.")
        print("This will automatically use COALESCED layout in the Metal backend.")
    
    # Create random input array
    print(f"Creating random array with {args.size} elements...")
    x = np.random.rand(args.size).astype(np.float32)
    
    print("\nRunning Triton reduction (uses COALESCED layout automatically)...")
    triton_result, triton_time = triton_sum_reduction(x, args.block_size)
    
    if args.verify:
        print("\nVerifying results with NumPy...")
        numpy_result, numpy_time = numpy_sum_reduction(x)
        
        # Calculate speedup
        speedup = numpy_time / triton_time
        print(f"\nSpeedup over NumPy: {speedup:.2f}x")
        
        # Verify correctness
        error = abs(triton_result - numpy_result)
        tolerance = 1e-5
        if error < tolerance:
            print(f"✅ Results match within tolerance ({error:.8f} < {tolerance})")
        else:
            print(f"❌ Results differ by {error:.8f}, which is above tolerance {tolerance}")
    
    print("\nNote: The COALESCED memory layout (value 8) is automatically applied")
    print("for reduction operations in the Metal backend, providing optimal performance")
    print("on Apple Silicon GPUs through better memory access patterns.")

if __name__ == "__main__":
    main() 