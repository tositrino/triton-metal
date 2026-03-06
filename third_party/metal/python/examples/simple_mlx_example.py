"""
Simple example of using MLX with Triton Metal backend
"""

import os
import sys
import numpy as np

# Add the Triton path to the Python path
triton_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(triton_path, "python"))

# Import Triton
import triton
import triton.language as tl

# Import MLX
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("WARNING: MLX not available. This example will run with CPU backend.")

# Import Metal backend
from triton_to_metal_converter import TritonToMLXConverter

# Define a simple Triton kernel
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel that adds two vectors"""
    # Program ID
    pid = tl.program_id(axis=0)
    # Block start
    block_start = pid * BLOCK_SIZE
    # Offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask for bounds checking
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Compute result
    output = x + y
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def main():
    """Main function"""
    # Print information about the environment
    print(f"MLX available: {MLX_AVAILABLE}")
    print(f"Triton version: {triton.__version__}")
    
    # Initialize data
    n_elements = 1024
    x = np.random.rand(n_elements).astype(np.float32)
    y = np.random.rand(n_elements).astype(np.float32)
    
    # Create MLX arrays if available
    if MLX_AVAILABLE:
        x_mlx = mx.array(x)
        y_mlx = mx.array(y)
        
        # Compute with MLX directly
        start_time = triton.testing.perf_counter()
        output_mlx = x_mlx + y_mlx
        mx.eval(output_mlx)  # Force evaluation
        mlx_time = triton.testing.perf_counter() - start_time
        
        print(f"MLX direct computation time: {mlx_time:.6f} seconds")
    
    # Create numpy arrays for Triton
    x_triton = np.copy(x)
    y_triton = np.copy(y)
    output_triton = np.zeros_like(x_triton)
    
    # Compute with Triton kernel
    start_time = triton.testing.perf_counter()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](
        x_triton, y_triton, output_triton,
        n_elements,
        BLOCK_SIZE=128,
    )
    triton_time = triton.testing.perf_counter() - start_time
    
    print(f"Triton kernel computation time: {triton_time:.6f} seconds")
    
    # Verify results
    if MLX_AVAILABLE:
        output_mlx_np = output_mlx.numpy()
        np.testing.assert_allclose(output_mlx_np, output_triton, rtol=1e-5)
        print("Results match between MLX and Triton!")
    
    # Demonstrate using TritonToMLXConverter
    if MLX_AVAILABLE:
        print("\nDemonstrating TritonToMLXConverter:")
        converter = TritonToMLXConverter()
        
        # Create sample operations
        ops = [
            {
                "id": "x_tensor",
                "type": "tt.make_tensor",
                "shape": [n_elements],
                "dtype": "float32",
                "init_value": x
            },
            {
                "id": "y_tensor",
                "type": "tt.make_tensor",
                "shape": [n_elements],
                "dtype": "float32",
                "init_value": y
            },
            {
                "id": "add_result",
                "type": "tt.binary.add",
                "lhs_id": "x_tensor",
                "rhs_id": "y_tensor"
            }
        ]
        
        # Convert operations
        start_time = triton.testing.perf_counter()
        results = converter.convert_operations(ops)
        converter_time = triton.testing.perf_counter() - start_time
        
        print(f"TritonToMLXConverter computation time: {converter_time:.6f} seconds")
        
        # Verify converter results
        add_result = results["add_result"]
        add_result_np = add_result.numpy()
        np.testing.assert_allclose(add_result_np, output_triton, rtol=1e-5)
        print("Results match between TritonToMLXConverter and Triton!")

if __name__ == "__main__":
    main() 