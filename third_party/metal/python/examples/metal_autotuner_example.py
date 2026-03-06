#!/usr/bin/env python
"""
Example demonstrating the use of the Metal auto-tuner with a matrix multiplication kernel
"""

import os
import sys
import time
import numpy as np
import argparse

# Add parent directory to path to allow importing from metal_auto_tuner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MLX.metal_auto_tuner import (
    MetalAutoTuner,
    TunableParam,
    ParamType,
    ConfigurationResult,
    get_matmul_metal_params
)

# Import Triton if available
try:
    import triton
    import triton.language as tl
    from triton.runtime import driver
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton not available. Will run a simulated example.")

# Matrix multiplication kernel
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Meta-parameters
    GROUP_M: tl.constexpr = 8,
):
    """Matrix multiplication kernel"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block start indices
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load blocks from a and b
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + (k * BLOCK_K + offs_k[None, :]) * stride_ak, 
                   mask=(offs_am[:, None] < M) & ((k * BLOCK_K + offs_k[None, :]) < K), 
                   other=0.0)
        b = tl.load(b_ptr + (k * BLOCK_K + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn, 
                   mask=((k * BLOCK_K + offs_k[:, None]) < K) & (offs_bn[None, :] < N), 
                   other=0.0)
        # Matrix multiplication
        acc += tl.dot(a, b)
    
    # Store result
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, acc, mask=mask)


def test_matmul(M, N, K, block_m, block_n, block_k, num_warps, num_stages, group_m=8):
    """Test matrix multiplication with given parameters"""
    # Check if Triton with Metal backend is available
    if not HAS_TRITON:
        # Simulate performance based on parameters
        # In real implementation, this would run the actual kernel
        flops = 2 * M * N * K
        theoretical_gflops = 0
        
        # Simulate performance with a model:
        # - Better utilization with appropriate block sizes
        # - Performance increases with warps up to a point
        # - Multiple stages can hide memory latency
        
        # Base performance value
        base_perf = 1000.0  # GFLOPS
        
        # Block size efficiency (penalize very small or very large blocks)
        block_m_eff = 1.0 - abs(np.log2(block_m / 64)) * 0.3
        block_n_eff = 1.0 - abs(np.log2(block_n / 64)) * 0.3
        block_k_eff = 1.0 - abs(np.log2(block_k / 32)) * 0.3
        block_eff = block_m_eff * block_n_eff * block_k_eff
        
        # Warps efficiency (more warps is better up to a point)
        warps_eff = min(num_warps / 8.0, 1.2)
        
        # Stages efficiency (more stages is better up to a point)
        stages_eff = min(num_stages / 3.0, 1.5)
        
        # Group size efficiency
        group_eff = min(group_m / 4.0, 1.2)
        
        # Combine factors
        theoretical_gflops = base_perf * block_eff * warps_eff * stages_eff * group_eff
        
        # Add some noise to simulate real hardware variability
        theoretical_gflops *= (0.9 + 0.2 * np.random.random())
        
        # Calculate runtime in milliseconds
        runtime_ms = flops / (theoretical_gflops * 1e9) * 1000
        
        return ConfigurationResult(
            config={
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "group_m": group_m
            },
            runtime_ms=runtime_ms,
            success=True,
            metrics={
                "gflops": theoretical_gflops,
                "flops": flops
            }
        )
    
    # Create actual matrices
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    c = np.zeros((M, N), dtype=np.float32)
    
    # Create device arrays
    device = driver.active.get_current_device()
    a_dev = a
    b_dev = b
    c_dev = c
    
    # Launch kernel with given configuration
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )
    
    try:
        # Time the kernel execution
        start_time = time.time()
        
        matmul_kernel[grid](
            a_dev, b_dev, c_dev,
            M, N, K,
            a_dev.stride(0), a_dev.stride(1),
            b_dev.stride(0), b_dev.stride(1),
            c_dev.stride(0), c_dev.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=num_warps,
            num_stages=num_stages
        )
        
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000
        
        # Verify result with CPU computation
        c_ref = np.matmul(a, b)
        max_error = np.max(np.abs(c_dev - c_ref))
        success = max_error < 1e-3
        
        # Calculate performance in GFLOPS
        flops = 2 * M * N * K
        gflops = flops / (runtime_ms / 1000) / 1e9
        
        return ConfigurationResult(
            config={
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "group_m": group_m
            },
            runtime_ms=runtime_ms,
            success=success,
            metrics={
                "gflops": gflops,
                "max_error": float(max_error),
                "flops": flops
            }
        )
    
    except Exception as e:
        # If kernel execution fails, return failure
        print(f"Kernel execution failed: {e}")
        return ConfigurationResult(
            config={
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "group_m": group_m
            },
            runtime_ms=float("inf"),
            success=False,
            metrics={"error": str(e)}
        )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Metal Auto-Tuner Example")
    parser.add_argument("--M", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--N", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--K", type=int, default=1024, help="Matrix K dimension")
    parser.add_argument("--trials", type=int, default=20, help="Number of tuning trials")
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "grid", "bayesian"], 
                        help="Search strategy")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    args = parser.parse_args()
    
    print(f"Auto-tuning matrix multiplication {args.M}x{args.N}x{args.K}")
    
    # Get tunable parameters for matrix multiplication
    params = get_matmul_metal_params()
    
    # Add GROUP_M parameter
    params.append(
        TunableParam(
            name="group_m",
            param_type=ParamType.POWER_OF_TWO,
            default_value=8,
            min_value=1,
            max_value=16
        )
    )
    
    # Create auto-tuner
    tuner = MetalAutoTuner(
        f"matmul_{args.M}_{args.N}_{args.K}",
        params,
        n_trials=args.trials,
        search_strategy=args.strategy,
        cache_dir=args.cache_dir
    )
    
    # Define evaluation function
    def evaluate_config(config):
        return test_matmul(
            args.M, args.N, args.K,
            config["block_m"], config["block_n"], config["block_k"],
            config["num_warps"], config["num_stages"], config["group_m"]
        )
    
    # Run tuning
    print(f"Running {args.trials} tuning trials with {args.strategy} strategy...")
    best_config = tuner.tune(evaluate_config)
    
    # Print best configuration
    print(f"\nBest configuration found:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    
    # Get performance metrics for best configuration
    best_result = test_matmul(
        args.M, args.N, args.K,
        best_config["block_m"], best_config["block_n"], best_config["block_k"],
        best_config["num_warps"], best_config["num_stages"], best_config["group_m"]
    )
    
    print(f"\nPerformance of best configuration:")
    print(f"  Runtime: {best_result.runtime_ms:.3f} ms")
    print(f"  GFLOPS: {best_result.metrics.get('gflops', 0):.2f}")
    
    if 'max_error' in best_result.metrics:
        print(f"  Max Error: {best_result.metrics['max_error']:.6e}")


if __name__ == "__main__":
    main() 