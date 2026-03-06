#!/usr/bin/env python
"""Performance benchmarks for the Metal backend.

This script runs performance benchmarks for various operations with
different configurations to measure the Metal backend's performance.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not found. Cannot run benchmarks.")
    sys.exit(1)

try:
    from metal_backend import MetalOptions
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False
    print("Warning: Metal backend modules not found. Cannot run benchmarks.")
    sys.exit(1)

# Default configurations
DEFAULT_DTYPES = ["float32", "float16"]
DEFAULT_SIZES = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
DEFAULT_OPS = ["matmul", "reduction", "elementwise"]
DEFAULT_NUM_RUNS = 10
DEFAULT_WARMUP = 5

class BenchmarkResult:
    """Benchmark result container"""
    
    def __init__(self, name: str, size: Tuple[int, ...], dtype: str, 
                 config: Dict[str, Any], avg_time: float, std_dev: float,
                 flops: Optional[float] = None):
        """Initialize benchmark result
        
        Args:
            name: Benchmark name
            size: Input size
            dtype: Data type
            config: Configuration
            avg_time: Average execution time in ms
            std_dev: Standard deviation of execution time
            flops: Floating-point operations per second (if applicable)
        """
        self.name = name
        self.size = size
        self.dtype = dtype
        self.config = config
        self.avg_time = avg_time
        self.std_dev = std_dev
        self.flops = flops
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "name": self.name,
            "size": self.size,
            "dtype": self.dtype,
            "config": self.config,
            "avg_time_ms": self.avg_time,
            "std_dev_ms": self.std_dev
        }
        if self.flops is not None:
            result["gflops"] = self.flops / 1e9
        return result
    
    def __str__(self) -> str:
        """String representation"""
        size_str = "x".join(map(str, self.size))
        if self.flops is not None:
            return f"{self.name} {size_str} {self.dtype}: {self.avg_time:.2f}ms ± {self.std_dev:.2f}ms ({self.flops/1e9:.2f} GFLOPS)"
        else:
            return f"{self.name} {size_str} {self.dtype}: {self.avg_time:.2f}ms ± {self.std_dev:.2f}ms"

def time_function(func: Callable, args: Tuple = (), kwargs: Dict = {},
                 warmup: int = 5, num_runs: int = 10) -> Tuple[float, float]:
    """Time a function execution
    
    Args:
        func: Function to time
        args: Function arguments
        kwargs: Function keyword arguments
        warmup: Number of warmup runs
        num_runs: Number of timed runs
        
    Returns:
        Tuple of (average time in ms, standard deviation in ms)
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_dev = np.std(times)
    
    return avg_time, std_dev

def calculate_matmul_flops(m: int, n: int, k: int) -> float:
    """Calculate floating-point operations for matrix multiplication
    
    For a matrix multiplication C = A * B where A is m x k and B is k x n,
    the number of floating-point operations is approximately 2 * m * n * k.
    
    Args:
        m: First dimension of result
        n: Second dimension of result
        k: Inner dimension
        
    Returns:
        Number of floating-point operations
    """
    return 2 * m * n * k

def calculate_reduction_flops(size: Tuple[int, ...], axis: Optional[int] = None) -> float:
    """Calculate floating-point operations for reduction
    
    Args:
        size: Input size
        axis: Reduction axis (None for global reduction)
        
    Returns:
        Number of floating-point operations
    """
    if axis is None:
        # Global reduction: n-1 operations for n elements
        return np.prod(size) - 1
    else:
        # Axis reduction: n-1 operations for each slice along the axis
        return size[axis] - 1

def calculate_elementwise_flops(size: Tuple[int, ...], num_ops: int = 1) -> float:
    """Calculate floating-point operations for elementwise operations
    
    Args:
        size: Input size
        num_ops: Number of operations per element
        
    Returns:
        Number of floating-point operations
    """
    return np.prod(size) * num_ops

class MetalBenchmark:
    """Benchmark suite for Metal backend"""
    
    def __init__(self, dtypes: List[str] = DEFAULT_DTYPES, 
                 sizes: List[Tuple[int, ...]] = DEFAULT_SIZES,
                 ops: List[str] = DEFAULT_OPS,
                 num_runs: int = DEFAULT_NUM_RUNS,
                 warmup: int = DEFAULT_WARMUP,
                 output_dir: Optional[str] = None,
                 verbose: bool = False):
        """Initialize benchmark suite
        
        Args:
            dtypes: Data types to benchmark
            sizes: Input sizes to benchmark
            ops: Operations to benchmark
            num_runs: Number of timed runs
            warmup: Number of warmup runs
            output_dir: Directory to save results and plots
            verbose: Whether to print verbose output
        """
        self.dtypes = dtypes
        self.sizes = sizes
        self.ops = ops
        self.num_runs = num_runs
        self.warmup = warmup
        self.output_dir = output_dir
        self.verbose = verbose
        self.results = []
        
        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks
        
        Returns:
            List of benchmark results
        """
        # Run benchmarks
        for op in self.ops:
            if op == "matmul":
                self.run_matmul_benchmarks()
            elif op == "reduction":
                self.run_reduction_benchmarks()
            elif op == "elementwise":
                self.run_elementwise_benchmarks()
            else:
                print(f"Unknown operation: {op}")
        
        return self.results
    
    def run_matmul_benchmarks(self) -> None:
        """Run matrix multiplication benchmarks
        
        This benchmark tests matrix multiplication with various sizes and data types.
        """
        for dtype in self.dtypes:
            for size in self.sizes:
                m, n = size
                k = n  # Use square matrices for benchmarks
                
                try:
                    # Create random matrices
                    if dtype == "float32":
                        a = mx.random.uniform(shape=(m, k), dtype=mx.float32)
                        b = mx.random.uniform(shape=(k, n), dtype=mx.float32)
                    elif dtype == "float16":
                        a = mx.random.uniform(shape=(m, k), dtype=mx.float16)
                        b = mx.random.uniform(shape=(k, n), dtype=mx.float16)
                    else:
                        print(f"Unsupported dtype for matmul: {dtype}")
                        continue
                    
                    # Define matmul function
                    def run_matmul():
                        c = mx.matmul(a, b)
                        mx.eval(c)
                    
                    # Time execution
                    avg_time, std_dev = time_function(
                        run_matmul,
                        warmup=self.warmup,
                        num_runs=self.num_runs
                    )
                    
                    # Calculate FLOPS
                    flops = calculate_matmul_flops(m, n, k) / (avg_time / 1000)
                    
                    # Record result
                    result = BenchmarkResult(
                        name="MatMul",
                        size=(m, k, n),
                        dtype=dtype,
                        config={},
                        avg_time=avg_time,
                        std_dev=std_dev,
                        flops=flops
                    )
                    
                    self.results.append(result)
                    
                    if self.verbose:
                        print(result)
                except Exception as e:
                    print(f"Error in matmul benchmark {m}x{k}x{n} {dtype}: {e}")
    
    def run_reduction_benchmarks(self) -> None:
        """Run reduction benchmarks
        
        This benchmark tests reduction operations with various sizes, data types,
        and configurations.
        """
        for dtype in self.dtypes:
            for size in self.sizes:
                m, n = size
                
                for reduction_op in ["sum", "max", "mean"]:
                    for axis in [0, 1, None]:
                        try:
                            # Create random matrix
                            if dtype == "float32":
                                a = mx.random.uniform(shape=(m, n), dtype=mx.float32)
                            elif dtype == "float16":
                                a = mx.random.uniform(shape=(m, n), dtype=mx.float16)
                            else:
                                print(f"Unsupported dtype for reduction: {dtype}")
                                continue
                            
                            # Define reduction function
                            if reduction_op == "sum":
                                def run_reduction():
                                    c = mx.sum(a, axis=axis)
                                    mx.eval(c)
                            elif reduction_op == "max":
                                def run_reduction():
                                    c = mx.max(a, axis=axis)
                                    mx.eval(c)
                            elif reduction_op == "mean":
                                def run_reduction():
                                    c = mx.mean(a, axis=axis)
                                    mx.eval(c)
                            
                            # Time execution
                            avg_time, std_dev = time_function(
                                run_reduction,
                                warmup=self.warmup,
                                num_runs=self.num_runs
                            )
                            
                            # Calculate FLOPS
                            if axis is None:
                                # Global reduction
                                flops = calculate_reduction_flops((m, n)) / (avg_time / 1000)
                            else:
                                # Axis reduction
                                flops = calculate_reduction_flops((m, n), axis) / (avg_time / 1000)
                            
                            # Record result
                            axis_name = "all" if axis is None else f"axis{axis}"
                            result = BenchmarkResult(
                                name=f"{reduction_op.capitalize()}",
                                size=(m, n),
                                dtype=dtype,
                                config={"axis": axis_name},
                                avg_time=avg_time,
                                std_dev=std_dev,
                                flops=flops
                            )
                            
                            self.results.append(result)
                            
                            if self.verbose:
                                print(result)
                        except Exception as e:
                            print(f"Error in {reduction_op} reduction benchmark {m}x{n} {dtype} axis={axis}: {e}")
    
    def run_elementwise_benchmarks(self) -> None:
        """Run elementwise operation benchmarks
        
        This benchmark tests elementwise operations with various sizes and data types.
        """
        for dtype in self.dtypes:
            for size in self.sizes:
                m, n = size
                
                for op_name, num_ops in [("add", 1), ("multiply", 1), ("combined", 3)]:
                    try:
                        # Create random matrices
                        if dtype == "float32":
                            a = mx.random.uniform(shape=(m, n), dtype=mx.float32)
                            b = mx.random.uniform(shape=(m, n), dtype=mx.float32)
                            c = mx.random.uniform(shape=(m, n), dtype=mx.float32)
                        elif dtype == "float16":
                            a = mx.random.uniform(shape=(m, n), dtype=mx.float16)
                            b = mx.random.uniform(shape=(m, n), dtype=mx.float16)
                            c = mx.random.uniform(shape=(m, n), dtype=mx.float16)
                        else:
                            print(f"Unsupported dtype for elementwise: {dtype}")
                            continue
                        
                        # Define elementwise function
                        if op_name == "add":
                            def run_elementwise():
                                d = a + b
                                mx.eval(d)
                        elif op_name == "multiply":
                            def run_elementwise():
                                d = a * b
                                mx.eval(d)
                        elif op_name == "combined":
                            def run_elementwise():
                                d = a * b + c
                                mx.eval(d)
                        
                        # Time execution
                        avg_time, std_dev = time_function(
                            run_elementwise,
                            warmup=self.warmup,
                            num_runs=self.num_runs
                        )
                        
                        # Calculate FLOPS
                        flops = calculate_elementwise_flops((m, n), num_ops) / (avg_time / 1000)
                        
                        # Record result
                        result = BenchmarkResult(
                            name=f"Elementwise_{op_name}",
                            size=(m, n),
                            dtype=dtype,
                            config={},
                            avg_time=avg_time,
                            std_dev=std_dev,
                            flops=flops
                        )
                        
                        self.results.append(result)
                        
                        if self.verbose:
                            print(result)
                    except Exception as e:
                        print(f"Error in elementwise {op_name} benchmark {m}x{n} {dtype}: {e}")
    
    def save_results(self) -> None:
        """Save benchmark results to JSON file"""
        if self.output_dir is None:
            return
        
        results_dict = {
            "metadata": {
                "dtypes": self.dtypes,
                "sizes": self.sizes,
                "ops": self.ops,
                "num_runs": self.num_runs,
                "warmup": self.warmup,
                "timestamp": time.time()
            },
            "results": [result.to_dict() for result in self.results]
        }
        
        output_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def generate_plots(self) -> None:
        """Generate plots from benchmark results"""
        if self.output_dir is None or not self.results:
            return
        
        # Group results by operation type
        op_groups = {}
        for result in self.results:
            op_name = result.name.split("_")[0]
            if op_name not in op_groups:
                op_groups[op_name] = []
            op_groups[op_name].append(result)
        
        # Create plots for each operation type
        for op_name, results in op_groups.items():
            # Plot execution time vs matrix size
            self._plot_time_vs_size(op_name, results)
            
            # Plot FLOPS vs matrix size (if available)
            if all(result.flops is not None for result in results):
                self._plot_flops_vs_size(op_name, results)
    
    def _plot_time_vs_size(self, op_name: str, results: List[BenchmarkResult]) -> None:
        """Plot execution time vs matrix size
        
        Args:
            op_name: Operation name
            results: Benchmark results for the operation
        """
        plt.figure(figsize=(10, 6))
        
        # Group by dtype
        dtype_groups = {}
        for result in results:
            if result.dtype not in dtype_groups:
                dtype_groups[result.dtype] = []
            dtype_groups[result.dtype].append(result)
        
        # Plot each dtype
        for dtype, dtype_results in dtype_groups.items():
            # Sort by matrix size (assuming 2D matrices)
            dtype_results.sort(key=lambda r: r.size[0] * r.size[1])
            
            # Extract sizes and times
            sizes = [result.size[0] for result in dtype_results]  # First dimension
            times = [result.avg_time for result in dtype_results]
            errors = [result.std_dev for result in dtype_results]
            
            # Plot with error bars
            plt.errorbar(sizes, times, yerr=errors, marker="o", label=f"{dtype}")
        
        plt.title(f"{op_name} Execution Time vs Matrix Size")
        plt.xlabel("Matrix Size (N for NxN)")
        plt.ylabel("Execution Time (ms)")
        plt.xscale("log2")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f"{op_name}_time_vs_size.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_flops_vs_size(self, op_name: str, results: List[BenchmarkResult]) -> None:
        """Plot FLOPS vs matrix size
        
        Args:
            op_name: Operation name
            results: Benchmark results for the operation
        """
        plt.figure(figsize=(10, 6))
        
        # Group by dtype
        dtype_groups = {}
        for result in results:
            if result.dtype not in dtype_groups:
                dtype_groups[result.dtype] = []
            dtype_groups[result.dtype].append(result)
        
        # Plot each dtype
        for dtype, dtype_results in dtype_groups.items():
            # Sort by matrix size (assuming 2D matrices)
            dtype_results.sort(key=lambda r: r.size[0] * r.size[1])
            
            # Extract sizes and GFLOPS
            sizes = [result.size[0] for result in dtype_results]  # First dimension
            gflops = [result.flops / 1e9 for result in dtype_results]
            
            # Plot
            plt.plot(sizes, gflops, marker="o", label=f"{dtype}")
        
        plt.title(f"{op_name} Performance (GFLOPS) vs Matrix Size")
        plt.xlabel("Matrix Size (N for NxN)")
        plt.ylabel("GFLOPS")
        plt.xscale("log2")
        plt.grid(True)
        plt.legend()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f"{op_name}_flops_vs_size.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Performance benchmarks for Metal backend")
    parser.add_argument("--dtypes", nargs="+", default=DEFAULT_DTYPES,
                        help="Data types to benchmark")
    parser.add_argument("--sizes", nargs="+", type=lambda s: tuple(map(int, s.split("x"))),
                        default=DEFAULT_SIZES,
                        help="Matrix sizes to benchmark (format: MxN)")
    parser.add_argument("--ops", nargs="+", default=DEFAULT_OPS,
                        help="Operations to benchmark")
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS,
                        help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="Number of warmup runs")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results and plots")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    
    args = parser.parse_args()
    
    # Create benchmark and run tests
    print("=== Metal Backend Performance Benchmarks ===\n")
    
    benchmark = MetalBenchmark(
        dtypes=args.dtypes,
        sizes=args.sizes,
        ops=args.ops,
        num_runs=args.num_runs,
        warmup=args.warmup,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    results = benchmark.run_all_benchmarks()
    
    # Save results and generate plots
    benchmark.save_results()
    benchmark.generate_plots()
    
    print(f"\nBenchmarking complete. Total benchmarks: {len(results)}")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 