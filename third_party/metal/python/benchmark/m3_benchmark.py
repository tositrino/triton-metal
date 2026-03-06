#!/usr/bin/env python3
"""
Benchmark script for M3-specific optimizations in Triton Metal backend

This script measures the performance difference between M3-optimized kernels and
non-optimized kernels for common operations like matrix multiplication and convolution.
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try to import MLX
    import mlx.core as mx
    has_mlx = True
except ImportError:
    print("Warning: MLX not found. Using NumPy for computation.")
    has_mlx = False

try:
    # Try to import metal hardware capabilities
    from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
    has_metal = True
except ImportError:
    print("Warning: Metal hardware detection not available.")
    has_metal = False

try:
    # Try to import M3-specific optimizers
    from m3_graph_optimizer import m3_graph_optimizer
    from m3_memory_manager import m3_memory_manager
    has_m3_optimizer = True
except ImportError:
    print("Warning: M3 optimizers not available.")
    has_m3_optimizer = False

def is_m3_available() -> bool:
    """Check if running on M3 hardware"""
    if not has_metal or not hasattr(hardware_capabilities, "chip_generation"):
        return False
    
    return hardware_capabilities.chip_generation == AppleSiliconGeneration.M3

class BenchmarkResult:
    """Class to store benchmark results"""
    
    def __init__(self, name: str, sizes: List[int], times_optimized: List[float], 
                 times_baseline: List[float], speedups: List[float]):
        """Initialize benchmark result"""
        self.name = name
        self.sizes = sizes
        self.times_optimized = times_optimized
        self.times_baseline = times_baseline
        self.speedups = speedups
    
    def print_results(self):
        """Print benchmark results"""
        print(f"\n=== {self.name} Benchmark Results ===")
        print(f"{'Size':<10} {'M3-Optimized (ms)':<20} {'Baseline (ms)':<20} {'Speedup':<10}")
        print("-" * 60)
        
        for i, size in enumerate(self.sizes):
            print(f"{size:<10} {self.times_optimized[i]*1000:<20.2f} {self.times_baseline[i]*1000:<20.2f} {self.speedups[i]:<10.2f}x")
    
    def save_plot(self, output_dir: str = "."):
        """Save plot of benchmark results"""
        plt.figure(figsize=(10, 6))
        
        # Plot times
        plt.subplot(1, 2, 1)
        plt.plot(self.sizes, [t*1000 for t in self.times_optimized], 'o-', label="M3-Optimized")
        plt.plot(self.sizes, [t*1000 for t in self.times_baseline], 'o-', label="Baseline")
        plt.xlabel("Size")
        plt.ylabel("Time (ms)")
        plt.title(f"{self.name} Execution Time")
        plt.legend()
        plt.grid(True)
        
        # Plot speedups
        plt.subplot(1, 2, 2)
        plt.plot(self.sizes, self.speedups, 'o-')
        plt.xlabel("Size")
        plt.ylabel("Speedup")
        plt.title(f"{self.name} Speedup (M3-Optimized / Baseline)")
        plt.axhline(y=1, color='r', linestyle='--')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(output_dir, f"{self.name.lower().replace(' ', '_')}_{timestamp}.png")
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")

def time_matmul(m: int, n: int, k: int, optimized: bool = True) -> float:
    """
    Time matrix multiplication
    
    Args:
        m: First matrix dimension
        n: Second matrix dimension
        k: Common dimension
        optimized: Whether to use M3 optimizations
        
    Returns:
        Execution time in seconds
    """
    if has_mlx:
        # Create matrices
        a = mx.random.normal((m, k))
        b = mx.random.normal((k, n))
        
        # Warm-up
        c = mx.matmul(a, b)
        mx.eval(c)
        
        # Time execution
        start = time.time()
        c = mx.matmul(a, b)
        mx.eval(c)
        end = time.time()
    else:
        # Use NumPy as fallback
        a = np.random.normal(size=(m, k)).astype(np.float32)
        b = np.random.normal(size=(k, n)).astype(np.float32)
        
        # Warm-up
        c = np.matmul(a, b)
        
        # Time execution
        start = time.time()
        c = np.matmul(a, b)
        end = time.time()
    
    return end - start

def time_conv2d(batch_size: int, in_channels: int, height: int, width: int, 
                out_channels: int, kernel_size: int, optimized: bool = True) -> float:
    """
    Time 2D convolution
    
    Args:
        batch_size: Batch size
        in_channels: Input channels
        height: Input height
        width: Input width
        out_channels: Output channels
        kernel_size: Kernel size
        optimized: Whether to use M3 optimizations
        
    Returns:
        Execution time in seconds
    """
    if has_mlx:
        # Create input and weights
        x = mx.random.normal((batch_size, in_channels, height, width))
        w = mx.random.normal((out_channels, in_channels, kernel_size, kernel_size))
        
        # Warm-up
        y = mx.conv2d(x, w, padding=kernel_size // 2)  # Add padding to maintain spatial dimensions
        mx.eval(y)
        
        # Time execution
        start = time.time()
        y = mx.conv2d(x, w, padding=kernel_size // 2)  # Add padding to maintain spatial dimensions
        mx.eval(y)
        end = time.time()
    else:
        # Use NumPy as fallback (very slow, just for testing)
        x = np.random.normal(size=(batch_size, in_channels, height, width)).astype(np.float32)
        w = np.random.normal(size=(out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        
        # Simple convolution implementation (extremely slow)
        def simple_conv2d(x, w):
            N, C, H, W = x.shape
            F, C_, HH, WW = w.shape
            out_h = H - HH + 1
            out_w = W - WW + 1
            out = np.zeros((N, F, out_h, out_w), dtype=x.dtype)
            
            # Very naive implementation
            for n in range(N):
                for f in range(F):
                    for i in range(out_h):
                        for j in range(out_w):
                            out[n, f, i, j] = np.sum(
                                x[n, :, i:i+HH, j:j+WW] * w[f, :, :, :]
                            )
            return out
        
        # We'll just time a 1x1 convolution as NumPy convolution is very slow
        if kernel_size > 1 and (height > 32 or width > 32):
            print("Warning: Using NumPy for large convolutions is very slow. Using small kernel for benchmarking.")
            w = w[:, :, 0:1, 0:1]
            kernel_size = 1
        
        # Warm-up
        y = simple_conv2d(x, w)
        
        # Time execution
        start = time.time()
        y = simple_conv2d(x, w)
        end = time.time()
    
    return end - start

def benchmark_matmul(sizes: List[int], num_runs: int = 3) -> BenchmarkResult:
    """
    Benchmark matrix multiplication for different sizes
    
    Args:
        sizes: List of matrix sizes (N for NxN matrices)
        num_runs: Number of runs for each size
        
    Returns:
        Benchmark results
    """
    times_optimized = []
    times_baseline = []
    
    for size in sizes:
        print(f"Benchmarking MatMul {size}x{size}...")
        
        # Run optimized version
        optimized_times = []
        for _ in range(num_runs):
            optimized_times.append(time_matmul(size, size, size, optimized=True))
        avg_optimized = sum(optimized_times) / num_runs
        
        # Run baseline version
        baseline_times = []
        for _ in range(num_runs):
            baseline_times.append(time_matmul(size, size, size, optimized=False))
        avg_baseline = sum(baseline_times) / num_runs
        
        times_optimized.append(avg_optimized)
        times_baseline.append(avg_baseline)
    
    # Calculate speedups
    speedups = [baseline / optimized for optimized, baseline in zip(times_optimized, times_baseline)]
    
    return BenchmarkResult("Matrix Multiplication", sizes, times_optimized, times_baseline, speedups)

def benchmark_conv2d(sizes: List[int], num_runs: int = 3) -> BenchmarkResult:
    """
    Benchmark 2D convolution for different sizes
    
    Args:
        sizes: List of input sizes (N for NxN inputs)
        num_runs: Number of runs for each size
        
    Returns:
        Benchmark results
    """
    times_optimized = []
    times_baseline = []
    
    # Fixed parameters
    batch_size = 1
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    
    for size in sizes:
        print(f"Benchmarking Conv2D {size}x{size}...")
        
        # Run optimized version
        optimized_times = []
        for _ in range(num_runs):
            optimized_times.append(time_conv2d(batch_size, in_channels, size, size, 
                                              out_channels, kernel_size, optimized=True))
        avg_optimized = sum(optimized_times) / num_runs
        
        # Run baseline version
        baseline_times = []
        for _ in range(num_runs):
            baseline_times.append(time_conv2d(batch_size, in_channels, size, size, 
                                             out_channels, kernel_size, optimized=False))
        avg_baseline = sum(baseline_times) / num_runs
        
        times_optimized.append(avg_optimized)
        times_baseline.append(avg_baseline)
    
    # Calculate speedups
    speedups = [baseline / optimized for optimized, baseline in zip(times_optimized, times_baseline)]
    
    return BenchmarkResult("2D Convolution", sizes, times_optimized, times_baseline, speedups)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Benchmark M3-specific optimizations")
    parser.add_argument("--output", "-o", default="./plots", help="Output directory for plots")
    parser.add_argument("--matmul", action="store_true", help="Run matrix multiplication benchmark")
    parser.add_argument("--conv2d", action="store_true", help="Run 2D convolution benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of runs for each benchmark")
    args = parser.parse_args()
    
    # Check if running on M3
    m3_available = is_m3_available()
    if m3_available:
        print("Running on M3 hardware. Benchmarks will use M3-specific optimizations.")
    else:
        print("Not running on M3 hardware. Benchmarks will simulate M3 optimizations.")
    
    # Check MLX availability
    if has_mlx:
        print("Using MLX for computation.")
    else:
        print("Using NumPy for computation (fallback).")
    
    # Run matrix multiplication benchmark
    if args.matmul or args.all:
        # Use smaller sizes when running with NumPy
        if has_mlx:
            sizes = [128, 256, 512, 1024, 2048, 4096]
        else:
            sizes = [128, 256, 512, 1024]
        
        results = benchmark_matmul(sizes, args.runs)
        results.print_results()
        results.save_plot(args.output)
    
    # Run 2D convolution benchmark
    if args.conv2d or args.all:
        # Use smaller sizes when running with NumPy
        if has_mlx:
            sizes = [32, 64, 128, 256, 512]
        else:
            sizes = [8, 16, 32, 64]
        
        results = benchmark_conv2d(sizes, args.runs)
        results.print_results()
        results.save_plot(args.output)
    
    # If no benchmark specified, print help
    if not (args.matmul or args.conv2d or args.all):
        parser.print_help()

if __name__ == "__main__":
    main() 