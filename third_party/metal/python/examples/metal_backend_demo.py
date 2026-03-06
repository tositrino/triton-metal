#!/usr/bin/env python
"""
Metal Backend Demo

This example demonstrates how to use the Triton Metal backend on Apple Silicon GPUs.
It shows examples of various operations supported by the Metal backend.
"""

import os
import time
import numpy as np
import triton
import triton.language as tl

# Set backend to Metal explicitly if it's not already set
if "TRITON_BACKEND" not in os.environ:
    print("Setting TRITON_BACKEND to 'metal'")
    os.environ["TRITON_BACKEND"] = "metal"

print(f"Using backend: {os.environ.get('TRITON_BACKEND', 'default')}")

def print_device_info():
    """Print information about the current device"""
    try:
        import mlx.core as mx
        device = mx.get_default_device()
        print(f"MLX device: {device}")
        
        # Try to get additional Metal info
        import MLX.metal as metal
        if hasattr(metal, "device_name"):
            print(f"Device name: {metal.device_name()}")
            print(f"macOS version: {metal.macos_version()}")
            print(f"Device memory: {metal.total_device_memory() / (1024 ** 3):.2f} GB")
    except ImportError:
        print("MLX not found or not properly installed")
    except Exception as e:
        print(f"Error getting device info: {e}")

# Print device information
print("\n=== Device Information ===")
print_device_info()

# Example 1: Vector Addition
print("\n=== Example 1: Vector Addition ===")

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple vector addition kernel"""
    # Define unique program ID for this instance
    pid = tl.program_id(0)
    
    # Calculate block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for handling the last block
    mask = offsets < n_elements
    
    # Load data with the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform addition
    output = x + y
    
    # Store result with the mask
    tl.store(output_ptr + offsets, output, mask=mask)

# Run the vector addition
def run_vector_addition(n_elements=1024*1024):
    # Create input data
    x = np.random.rand(n_elements).astype(np.float32)
    y = np.random.rand(n_elements).astype(np.float32)
    
    # Reference output for verification
    reference_output = x + y
    
    # Output array
    output = np.zeros_like(x)
    
    # Define grid
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Warm-up
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Measure performance
    start_time = time.time()
    n_runs = 10
    for _ in range(n_runs):
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    elapsed = time.time() - start_time
    
    # Calculate GB/s
    bytes_per_element = 4  # float32 = 4 bytes
    bytes_read_write = 3 * n_elements * bytes_per_element  # 2 reads, 1 write
    total_bytes = bytes_read_write * n_runs
    bandwidth = total_bytes / elapsed / 1e9  # GB/s
    
    # Verify results
    max_error = np.max(np.abs(output - reference_output))
    
    print(f"Input size: {n_elements} elements")
    print(f"Time: {elapsed / n_runs * 1000:.3f} ms")
    print(f"Bandwidth: {bandwidth:.2f} GB/s")
    print(f"Max error: {max_error}")

# Run vector addition example
run_vector_addition()

# Example 2: Matrix Multiplication
print("\n=== Example 2: Matrix Multiplication ===")

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Matrix multiplication kernel: C = A @ B"""
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Id for this program instance
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    
    # Create ranges
    rm = start_m + tl.arange(0, BLOCK_SIZE_M)
    rn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for boundary checking
    mask_m = rm < M
    mask_n = rn < N
    
    # Pointers to A and B matrices
    a_ptrs = a_ptr + (rm[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak)
    b_ptrs = b_ptr + (tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    # Initialize accumulator to zeros
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Boundary check for K
        mask_k = k + tl.arange(0, BLOCK_SIZE_K) < K
        
        # Load A and B tiles
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :])
        
        # Compute matrix multiplication for this block
        acc += tl.dot(a, b)
        
        # Increment pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Calculate output pointers
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    
    # Store result
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def run_matrix_multiplication(M=1024, N=1024, K=1024):
    # Create input matrices
    a = np.random.rand(M, K).astype(np.float32)
    b = np.random.rand(K, N).astype(np.float32)
    
    # Reference output for verification
    reference_output = a @ b
    
    # Output matrix
    c = np.zeros((M, N), dtype=np.float32)
    
    # Matrix strides
    stride_am, stride_ak = a.strides[0] // 4, a.strides[1] // 4
    stride_bk, stride_bn = b.strides[0] // 4, b.strides[1] // 4
    stride_cm, stride_cn = c.strides[0] // 4, c.strides[1] // 4
    
    # Define block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Define grid
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N)
    )
    
    # Warm-up
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Measure performance
    start_time = time.time()
    n_runs = 5
    for _ in range(n_runs):
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
    
    elapsed = time.time() - start_time
    
    # Calculate FLOPS (floating-point operations per second)
    # Each matrix multiply-add operation counts as 2 FLOPs
    flops_per_matmul = 2 * M * N * K
    total_flops = flops_per_matmul * n_runs
    tflops = total_flops / elapsed / 1e12  # TeraFLOPS
    
    # Verify results
    max_error = np.max(np.abs(c - reference_output))
    
    print(f"Matrix sizes: A({M}x{K}), B({K}x{N}), C({M}x{N})")
    print(f"Time: {elapsed / n_runs * 1000:.3f} ms")
    print(f"Performance: {tflops:.4f} TFLOPS")
    print(f"Max error: {max_error}")

# Run matrix multiplication example
run_matrix_multiplication()

# Example 3: Element-wise operations
print("\n=== Example 3: Element-wise Operations ===")

@triton.jit
def elementwise_ops_kernel(
    x_ptr, y_ptr, out1_ptr, out2_ptr, out3_ptr, out4_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel demonstrating multiple element-wise operations"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform various element-wise operations
    # 1. Square root and add
    out1 = tl.sqrt(x) + tl.sqrt(y)
    
    # 2. Exponential and multiply
    out2 = tl.exp(x) * y
    
    # 3. Maximum and sine
    out3 = tl.maximum(x, y) * tl.sin(x)
    
    # 4. Complex expression
    out4 = tl.log(x + 1.0) / (y + 0.01) + tl.sqrt(x * y + 0.01)
    
    # Store results
    tl.store(out1_ptr + offsets, out1, mask=mask)
    tl.store(out2_ptr + offsets, out2, mask=mask)
    tl.store(out3_ptr + offsets, out3, mask=mask)
    tl.store(out4_ptr + offsets, out4, mask=mask)

def run_elementwise_ops(n_elements=1024*1024):
    # Create input data (all positive for this example)
    x = np.random.rand(n_elements).astype(np.float32) + 0.1
    y = np.random.rand(n_elements).astype(np.float32) + 0.1
    
    # Output arrays
    out1 = np.zeros_like(x)
    out2 = np.zeros_like(x)
    out3 = np.zeros_like(x)
    out4 = np.zeros_like(x)
    
    # Reference outputs
    ref_out1 = np.sqrt(x) + np.sqrt(y)
    ref_out2 = np.exp(x) * y
    ref_out3 = np.maximum(x, y) * np.sin(x)
    ref_out4 = np.log(x + 1.0) / (y + 0.01) + np.sqrt(x * y + 0.01)
    
    # Define grid
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Run kernel
    elementwise_ops_kernel[grid](
        x, y, out1, out2, out3, out4,
        n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Verify results
    max_errors = [
        np.max(np.abs(out1 - ref_out1)),
        np.max(np.abs(out2 - ref_out2)),
        np.max(np.abs(out3 - ref_out3)),
        np.max(np.abs(out4 - ref_out4))
    ]
    
    print(f"Input size: {n_elements} elements")
    print(f"Max errors:")
    print(f"  Sqrt+Add: {max_errors[0]}")
    print(f"  Exp*Y: {max_errors[1]}")
    print(f"  Max*Sin: {max_errors[2]}")
    print(f"  Complex: {max_errors[3]}")

# Run element-wise operations example
run_elementwise_ops()

print("\nAll examples completed successfully!")

if __name__ == "__main__":
    pass 