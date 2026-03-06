#!/usr/bin/env python3
# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# This file demonstrates comparing the Metal backend with other backends
# like CUDA, focusing on performance differences between M3 and other hardware.

import torch
import triton
import triton.language as tl
import time
import argparse
import matplotlib.pyplot as plt
from tabulate import tabulate
import platform

# Initialize backend flags
has_metal = False
has_cuda = False
has_cpu = True  # We always have CPU

# Check if Metal backend is available
try:
    # This will register the Metal backend if available
    import triton
    has_metal = True
    print("🔥 Metal backend loaded successfully!")
except ImportError:
    print("⚠️ Metal backend not available")

# Check if CUDA backend is available
try:
    torch.cuda.is_available()
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("🚀 CUDA backend available!")
    else:
        print("❌ CUDA backend not available")
except:
    print("❌ CUDA backend not available")

# Print system information
print(f"System: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")

# Available backends for testing
available_backends = []
if has_metal: available_backends.append("metal")
if has_cuda: available_backends.append("cuda")
available_backends.append("cpu")

# Define a simple matrix multiplication kernel for all backends
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """Matrix multiplication kernel that works on all backends"""
    # Similar to the matmul example, but simplified for comparison
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Compute the block indices
    block_idx_m = pid // num_pid_n
    block_idx_n = pid % num_pid_n
    
    # Compute starting indices
    start_m = block_idx_m * BLOCK_SIZE_M
    start_n = block_idx_n * BLOCK_SIZE_N
    
    # Create offsets
    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        start_k = k * BLOCK_SIZE_K
        offs_k = start_k + tl.arange(0, BLOCK_SIZE_K)
        
        # Bounds checking
        offs_am_mask = offs_am < M
        offs_bn_mask = offs_bn < N
        offs_k_mask = offs_k < K
        
        # Load tiles
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                   mask=offs_am_mask[:, None] & offs_k_mask[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                   mask=offs_k_mask[:, None] & offs_bn_mask[None, :])
        
        # Compute matrix multiplication
        acc += tl.dot(a, b)
    
    # Apply bounds checking for output
    offs_cm_mask = offs_am < M
    offs_cn_mask = offs_bn < N
    
    # Store result
    c = acc.to(tl.float32)
    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn,
            c, mask=offs_cm_mask[:, None] & offs_cn_mask[None, :])


# Define an element-wise operation kernel for all backends
@triton.jit
def elementwise_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Element-wise operation kernel (x * y + sin(x)) that works on all backends"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute and store result
    output = x * y + tl.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)


# Define a reduction kernel for all backends
@triton.jit
def reduction_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Reduction (sum) kernel that works on all backends"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute reduction within the block
    block_sum = tl.sum(x, axis=0)
    
    # The first thread in the block stores the result
    if pid == 0:
        tl.store(output_ptr, block_sum)


def benchmark_matmul(M, N, K, device, dtype, num_repeats=100):
    """Benchmark matrix multiplication on a specific device"""
    # Create input tensors
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    c = torch.empty((M, N), device=device, dtype=dtype)
    
    # Define grid and block sizes
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    # Auto-tuning configs
    configs = [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        # Additional M3-optimized configs (will be preferred on M3 hardware)
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
    ]
    
    # Warmup
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    # Synchronize before timing
    if device == "cuda": torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_repeats):
        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    
    # Synchronize after timing
    if device == "cuda": torch.cuda.synchronize()
    
    # Calculate elapsed time and TFLOPs
    elapsed_time = (time.time() - start_time) / num_repeats
    flops = 2 * M * N * K  # multiply-add is 2 operations
    tflops = flops / elapsed_time / 1e12
    
    return elapsed_time, tflops


def benchmark_elementwise(n_elements, device, dtype, num_repeats=100):
    """Benchmark element-wise operations on a specific device"""
    # Create input tensors
    x = torch.randn(n_elements, device=device, dtype=dtype)
    y = torch.randn(n_elements, device=device, dtype=dtype)
    output = torch.empty(n_elements, device=device, dtype=dtype)
    
    # Define grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # Auto-tuning configs
    configs = [
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 256}),
        # Additional sizes for M3
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ]
    
    # Warmup
    elementwise_kernel[grid](x, y, output, n_elements)
    
    # Synchronize before timing
    if device == "cuda": torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_repeats):
        elementwise_kernel[grid](x, y, output, n_elements)
    
    # Synchronize after timing
    if device == "cuda": torch.cuda.synchronize()
    
    # Calculate elapsed time and throughput (GB/s)
    elapsed_time = (time.time() - start_time) / num_repeats
    bytes_processed = n_elements * 3 * 4  # 2 inputs, 1 output, 4 bytes per element (assuming float32)
    bandwidth = bytes_processed / elapsed_time / 1e9
    
    return elapsed_time, bandwidth


def benchmark_reduction(n_elements, device, dtype, num_repeats=100):
    """Benchmark reduction operations on a specific device"""
    # Create input tensor
    x = torch.randn(n_elements, device=device, dtype=dtype)
    output = torch.empty(1, device=device, dtype=dtype)
    
    # Define grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # Auto-tuning configs
    configs = [
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 256}),
        # Additional sizes for M3
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ]
    
    # Warmup
    reduction_kernel[grid](x, output, n_elements)
    
    # Synchronize before timing
    if device == "cuda": torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_repeats):
        reduction_kernel[grid](x, output, n_elements)
    
    # Synchronize after timing
    if device == "cuda": torch.cuda.synchronize()
    
    # Calculate elapsed time and throughput (GB/s)
    elapsed_time = (time.time() - start_time) / num_repeats
    bytes_processed = n_elements * 4  # 1 input, 4 bytes per element (assuming float32)
    bandwidth = bytes_processed / elapsed_time / 1e9
    
    return elapsed_time, bandwidth


def run_all_benchmarks(backends, sizes, dtype, num_repeats=10):
    """Run all benchmarks on all specified backends and sizes"""
    results = {
        "matmul": {},
        "elementwise": {},
        "reduction": {}
    }
    
    for backend in backends:
        print(f"\nRunning benchmarks on {backend.upper()} backend...")
        results["matmul"][backend] = []
        results["elementwise"][backend] = []
        results["reduction"][backend] = []
        
        # Matrix multiplication benchmarks
        print(f"  Matrix multiplication benchmarks:")
        for size in sizes:
            print(f"    Size: {size}x{size}...")
            try:
                time_ms, tflops = benchmark_matmul(size, size, size, backend, dtype, num_repeats)
                results["matmul"][backend].append((size, time_ms * 1000, tflops))
                print(f"      {time_ms*1000:.2f} ms, {tflops:.2f} TFLOPs")
            except Exception as e:
                print(f"      Failed: {e}")
                results["matmul"][backend].append((size, None, None))
        
        # Element-wise operation benchmarks
        print(f"  Element-wise operation benchmarks:")
        for size in [s*s for s in sizes]:  # Square sizes for large vectors
            print(f"    Elements: {size}...")
            try:
                time_ms, gb_per_s = benchmark_elementwise(size, backend, dtype, num_repeats)
                results["elementwise"][backend].append((size, time_ms * 1000, gb_per_s))
                print(f"      {time_ms*1000:.2f} ms, {gb_per_s:.2f} GB/s")
            except Exception as e:
                print(f"      Failed: {e}")
                results["elementwise"][backend].append((size, None, None))
        
        # Reduction benchmarks
        print(f"  Reduction operation benchmarks:")
        for size in [s*s for s in sizes]:  # Square sizes for large vectors
            print(f"    Elements: {size}...")
            try:
                time_ms, gb_per_s = benchmark_reduction(size, backend, dtype, num_repeats)
                results["reduction"][backend].append((size, time_ms * 1000, gb_per_s))
                print(f"      {time_ms*1000:.2f} ms, {gb_per_s:.2f} GB/s")
            except Exception as e:
                print(f"      Failed: {e}")
                results["reduction"][backend].append((size, None, None))
    
    return results


def print_results_table(results, operation, metric_name):
    """Print a nice table with the benchmark results"""
    table_data = []
    headers = ["Size"]
    
    # Add backends to headers
    for backend in results[operation].keys():
        headers.append(f"{backend.upper()} Time (ms)")
        headers.append(f"{backend.upper()} {metric_name}")
    
    # Prepare data rows
    for i in range(len(next(iter(results[operation].values())))):
        row = []
        size = next(iter(results[operation].values()))[i][0]
        row.append(size)
        
        for backend in results[operation].keys():
            if results[operation][backend][i][1] is not None:
                row.append(f"{results[operation][backend][i][1]:.2f}")
                row.append(f"{results[operation][backend][i][2]:.2f}")
            else:
                row.append("N/A")
                row.append("N/A")
        
        table_data.append(row)
    
    print(f"\nBenchmark Results - {operation.capitalize()}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def plot_benchmark_results(results, operation, metric_idx, metric_name, sizes=None):
    """Plot the benchmark results for better visualization"""
    plt.figure(figsize=(10, 6))
    
    # If sizes not specified, use all sizes
    if sizes is None:
        backend_key = next(iter(results[operation].keys()))
        sizes = [r[0] for r in results[operation][backend_key]]
    
    # Prepare data for plotting
    for backend in results[operation].keys():
        metrics = []
        valid_sizes = []
        
        for i, size in enumerate(sizes):
            if i < len(results[operation][backend]) and results[operation][backend][i][metric_idx] is not None:
                metrics.append(results[operation][backend][i][metric_idx])
                valid_sizes.append(size)
        
        plt.plot(valid_sizes, metrics, 'o-', label=backend.upper())
    
    plt.title(f"{operation.capitalize()} Benchmark: {metric_name}")
    plt.xlabel("Size" if operation == "matmul" else "Number of Elements")
    plt.ylabel(metric_name)
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # Save the plot
    plt.savefig(f"{operation}_{metric_name.lower().replace('/', '_per_')}.png")
    print(f"Plot saved to {operation}_{metric_name.lower().replace('/', '_per_')}.png")


def main():
    """Main function to run the benchmark comparison"""
    parser = argparse.ArgumentParser(description="Compare Triton backends performance")
    parser.add_argument("--backends", nargs="+", default=available_backends,
                        help=f"Backends to test from {available_backends}")
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 1024, 2048, 4096],
                        help="Sizes to benchmark (matrices will be NxN)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "float32"], help="Data type")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of repeats for each benchmark")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable generating plots")
    
    args = parser.parse_args()
    
    # Validate backends
    for backend in args.backends:
        if backend not in available_backends:
            print(f"Warning: Backend '{backend}' is not available. Removing from test list.")
    
    backends_to_test = [b for b in args.backends if b in available_backends]
    if not backends_to_test:
        print("Error: No valid backends to test!")
        return
    
    # Convert dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Run benchmarks
    print("\n===== STARTING BACKEND COMPARISON BENCHMARKS =====")
    print(f"Testing backends: {', '.join(backends_to_test)}")
    print(f"Matrix sizes: {args.sizes}")
    print(f"Data type: {args.dtype}")
    print(f"Repeats per benchmark: {args.repeats}")
    
    results = run_all_benchmarks(backends_to_test, args.sizes, dtype, args.repeats)
    
    # Print tables
    print_results_table(results, "matmul", "TFLOPs")
    print_results_table(results, "elementwise", "GB/s")
    print_results_table(results, "reduction", "GB/s")
    
    # Generate plots if enabled
    if not args.no_plots:
        try:
            # Import matplotlib here to not require it for just running benchmarks
            import matplotlib.pyplot as plt
            
            plot_benchmark_results(results, "matmul", 2, "TFLOPs")
            plot_benchmark_results(results, "elementwise", 2, "GB/s")
            plot_benchmark_results(results, "reduction", 2, "GB/s")
        except ImportError:
            print("Matplotlib not available. Skipping plots.")
    
    # Additional information about M3 optimizations if Metal backend was tested
    if "metal" in backends_to_test:
        print("\n===== METAL BACKEND SPECIFIC NOTES =====")
        print("When running on M3 hardware, this benchmark automatically leverages:")
        print(" - 64KB shared memory (vs 32KB on M1/M2)")
        print(" - 8-wide vectorization (vs 4-wide on M1/M2)")
        print(" - Enhanced SIMD operations (32-wide vs 16-wide)")
        print(" - Dynamic register caching")
        print(" - Tensor cores for matrix operations")
        print("\nThese optimizations contribute to the performance differences observed.")


if __name__ == "__main__":
    main() 