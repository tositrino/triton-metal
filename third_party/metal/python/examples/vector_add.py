"""
Vector Addition Example Using Triton Metal Backend

This example demonstrates how to add two vectors using Triton
with the Metal backend on Apple Silicon GPUs.
"""

import os
import sys
import time
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import MLX
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    print("MLX not found. Please install it with 'pip install mlx'")
    MLX_AVAILABLE = False
    sys.exit(1)

# Try to import Triton
try:
    import triton
    import triton.language as tl
except ImportError:
    print("Triton not found. Please install it with 'pip install triton'")
    sys.exit(1)

# Import our modules
from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration

# Set backend to metal
os.environ["TRITON_BACKEND"] = "metal"

# Define vector addition kernel
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute vector addition z = x + y
    
    Args:
        x_ptr: Pointer to first vector
        y_ptr: Pointer to second vector
        output_ptr: Pointer to output vector
        n_elements: Number of elements in vectors
        BLOCK_SIZE: Number of elements to process per block
    """
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate start offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load data with bounds checking
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform vector addition
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add_triton(x, y):
    """
    Add two vectors using Triton
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Result vector
    """
    # Get vector length
    n_elements = x.size
    
    # Allocate output
    output = mx.zeros_like(x)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

def vector_add_mlx(x, y):
    """
    Add two vectors using MLX directly
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Result vector
    """
    return x + y

def test_vector_add(size=1_000_000, num_runs=5):
    """
    Test vector addition implementation
    
    Args:
        size: Vector size
        num_runs: Number of test runs
    """
    print(f"\nTesting vector addition with size={size:,}")
    
    # Create input vectors
    x = mx.random.normal((size,))
    y = mx.random.normal((size,))
    
    # MLX implementation
    mlx_times = []
    for i in range(num_runs):
        start_time = time.time()
        mlx_output = vector_add_mlx(x, y)
        mx.eval(mlx_output)  # Force evaluation
        mlx_times.append(time.time() - start_time)
    
    mlx_avg_time = sum(mlx_times) / len(mlx_times)
    print(f"MLX Vector Addition: {mlx_avg_time:.6f}s")
    
    # Triton implementation
    triton_times = []
    for i in range(num_runs):
        start_time = time.time()
        triton_output = vector_add_triton(x, y)
        mx.eval(triton_output)  # Force evaluation
        triton_times.append(time.time() - start_time)
    
    triton_avg_time = sum(triton_times) / len(triton_times)
    print(f"Triton Vector Addition: {triton_avg_time:.6f}s")
    
    # Compare results
    max_diff = mx.max(mx.abs(mlx_output - triton_output))
    print(f"Maximum difference: {max_diff}")
    
    # Check if results match
    if max_diff < 1e-5:
        print("✅ Implementations match!")
    else:
        print("❌ Implementations do not match!")
    
    # Print speedup
    if mlx_avg_time > 0:
        speedup = mlx_avg_time / triton_avg_time
        print(f"Triton/Metal speedup: {speedup:.2f}x")

def benchmark_vector_add():
    """Benchmark vector addition with different sizes"""
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    
    print("\nBenchmarking Vector Addition:")
    print(f"{'Size':>10} {'MLX (ms)':>10} {'Triton (ms)':>12} {'Speedup':>8}")
    print("-" * 42)
    
    for size in sizes:
        # Create input vectors
        x = mx.random.normal((size,))
        y = mx.random.normal((size,))
        
        # Warm up
        _ = vector_add_mlx(x, y)
        _ = vector_add_triton(x, y)
        
        # MLX implementation
        start_time = time.time()
        mlx_output = vector_add_mlx(x, y)
        mx.eval(mlx_output)  # Force evaluation
        mlx_time = time.time() - start_time
        
        # Triton implementation
        start_time = time.time()
        triton_output = vector_add_triton(x, y)
        mx.eval(triton_output)  # Force evaluation
        triton_time = time.time() - start_time
        
        # Convert to milliseconds
        mlx_ms = mlx_time * 1000
        triton_ms = triton_time * 1000
        
        # Calculate speedup
        speedup = mlx_time / triton_time if triton_time > 0 else 0
        
        # Print results
        print(f"{size:10,} {mlx_ms:10.2f} {triton_ms:12.2f} {speedup:8.2f}x")

if __name__ == "__main__":
    print(f"Apple {hardware_capabilities.chip_generation.name} detected")
    print(f"MLX version: {mx.__version__}")
    print(f"Triton version: {triton.__version__}")
    
    # Test vector addition
    test_vector_add(size=1_000_000)
    
    # Benchmark vector addition with different sizes
    benchmark_vector_add()