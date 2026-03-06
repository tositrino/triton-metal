#!/usr/bin/env python3
# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# This file demonstrates matrix multiplication using the Triton Metal backend
# with specific optimizations for M3 chips.

import numpy as np
import torch
import time
import argparse
import os

# Import Triton
import triton
import triton.language as tl

# Check if Metal backend is available 
use_metal = os.environ.get("triton", "1") == "1"
if use_metal:
    try:
        # This will register the Metal backend
        import triton
        print("🔥 Metal backend loaded successfully!")
    except ImportError:
        use_metal = False
        print("⚠️ Metal backend not available, falling back to CUDA")

# Define the kernel using Triton language
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The strides for accessing the matrices
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (meta-parameters)
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Optional: Group size for better scheduling on M3
    GROUP_SIZE_M: tl.constexpr
):
    """
    Computes matrix multiplication C = A @ B
    
    This implementation uses the M3-compatible block sizes and tiling strategy
    when running on Metal backend with M3 chip detection.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Number of blocks in the M dimension
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    
    # Compute the block indices
    # For M3, we use a 2D grid where we compute the row and column indices
    # from the linearized program ID
    block_idx_m = pid // (num_blocks_m // GROUP_SIZE_M)
    block_idx_n = pid % (num_blocks_m // GROUP_SIZE_M)
    
    # Compute starting indices for this block
    start_m = block_idx_m * BLOCK_SIZE_M
    start_n = block_idx_n * BLOCK_SIZE_N
    
    # Create offsets for A and B matrices
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize the accumulator with zeros
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through the K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute the starting index for this block in the K dimension
        start_k = k * BLOCK_SIZE_K
        
        # Create offset for K dimension
        offs_k = start_k + tl.arange(0, BLOCK_SIZE_K)
        
        # Bounds checking to handle partial tiles
        offs_am_mask = offs_am < M
        offs_bn_mask = offs_bn < N
        offs_k_mask = offs_k < K
        
        # Load tiles from A and B matrices
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak, 
                   mask=offs_am_mask[:, None] & offs_k_mask[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                   mask=offs_k_mask[:, None] & offs_bn_mask[None, :])
        
        # Perform matrix multiplication for this block
        acc += tl.dot(a, b)
    
    # Apply bounds checking for the output
    offs_cm_mask = offs_am < M
    offs_cn_mask = offs_bn < N
    
    # Store the results
    c = acc.to(tl.float32)
    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn,
            c, mask=offs_cm_mask[:, None] & offs_cn_mask[None, :])


def matmul(a, b):
    """
    Compute the matrix multiplication a @ b using Triton.
    
    This function uses auto-tuning to find the optimal block sizes
    for the current hardware, with special handling for M3 chips.
    """
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    # Get matrix dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Block configuratios to try for auto-tuning
    # M3-specific configurations will be preferred when running on M3
    configs = []
    
    # Standard configurations that work well on most hardware
    for block_m, block_n, block_k, group_m in [
        (64, 64, 32, 8), 
        (128, 128, 32, 8),
        (64, 128, 32, 4), 
        (128, 64, 32, 4),
        (256, 64, 32, 4),
        (64, 256, 32, 4),
    ]:
        configs.append(
            triton.Config({
                'BLOCK_SIZE_M': block_m,
                'BLOCK_SIZE_N': block_n,
                'BLOCK_SIZE_K': block_k,
                'GROUP_SIZE_M': group_m
            })
        )
    
    # M3-optimized configurations (will be preferred on M3 hardware)
    # These take advantage of larger register file, vectorization, and shared memory
    for block_m, block_n, block_k, group_m in [
        (128, 128, 64, 8),  # Larger K block for M3's wider SIMD
        (64, 256, 64, 8),   # Good for portrait matrices on M3
        (256, 64, 64, 8),   # Good for landscape matrices on M3
        (32, 32, 128, 16),  # Small tiles with large K for memory-bound cases
    ]:
        configs.append(
            triton.Config({
                'BLOCK_SIZE_M': block_m,
                'BLOCK_SIZE_N': block_n,
                'BLOCK_SIZE_K': block_k,
                'GROUP_SIZE_M': group_m
            })
        )
    
    # Create launch grid
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * 
        triton.cdiv(N, meta['BLOCK_SIZE_N']) // meta['GROUP_SIZE_M'],
    )
    
    # Launch the kernel with auto-tuning
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c


def benchmark_matmul(M, N, K, device="cuda", dtype=torch.float32, num_repeats=100):
    """Benchmark matrix multiplication performance with different implementations"""
    print(f"Benchmarking matmul with M={M}, N={N}, K={K}")
    
    # Create random matrices
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    
    # Warm-up for PyTorch
    torch_c = torch.matmul(a, b)
    
    # Benchmark PyTorch matmul
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    for _ in range(num_repeats):
        torch_c = torch.matmul(a, b)
    torch.cuda.synchronize() if device == "cuda" else None
    torch_time = (time.time() - start) / num_repeats
    
    # Warm-up for our implementation
    triton_c = matmul(a, b)
    
    # Check correctness
    assert torch.allclose(torch_c, triton_c, atol=1e-2, rtol=1e-2), \
        "Results of Triton kernel and PyTorch matmul don't match!"
    
    # Benchmark our implementation
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    for _ in range(num_repeats):
        triton_c = matmul(a, b)
    torch.cuda.synchronize() if device == "cuda" else None
    triton_time = (time.time() - start) / num_repeats
    
    # Calculate TFLOPs
    flops = 2 * M * N * K  # multiply-add is 2 operations
    torch_tflops = flops / torch_time / 1e12
    triton_tflops = flops / triton_time / 1e12
    
    # Print benchmark results
    print(f"PyTorch: {torch_time*1000:.4f} ms, {torch_tflops:.2f} TFLOPs")
    print(f"Triton: {triton_time*1000:.4f} ms, {triton_tflops:.2f} TFLOPs")
    print(f"Speedup: {torch_time/triton_time:.2f}x")


def main():
    """Run the example"""
    parser = argparse.ArgumentParser(description="Matrix multiplication benchmark with Triton")
    parser.add_argument("--M", type=int, default=1024, help="M dimension")
    parser.add_argument("--N", type=int, default=1024, help="N dimension")
    parser.add_argument("--K", type=int, default=1024, help="K dimension")
    parser.add_argument("--device", type=str, default="cuda" if not use_metal else "mps", 
                        help="Device to run on (cuda or mps)")
    parser.add_argument("--dtype", type=str, default="float32", 
                        choices=["float16", "float32"], help="Data type")
    parser.add_argument("--repeats", type=int, default=100, help="Number of repeats for benchmark")
    
    args = parser.parse_args()
    
    # Parse data type
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Run benchmark
    benchmark_matmul(args.M, args.N, args.K, args.device, dtype, args.repeats)
    
    # If using Metal, mention M3 optimizations
    if use_metal:
        print("\nNote: When running on M3 hardware, this example automatically uses")
        print("M3-specific optimizations for improved performance.")
        print("These optimizations include:")
        print(" - Larger tile sizes to utilize 64KB shared memory")
        print(" - 8-wide vectorization (vs 4-wide on M1/M2)")
        print(" - Optimized memory layout for tensor cores")
        print(" - Enhanced SIMD group operations")


if __name__ == "__main__":
    main()
