#!/usr/bin/env python
"""
Metal Backend Benchmark

This script benchmarks the Triton Metal backend on Apple Silicon.
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    print("MLX not found. Please install it with 'pip install mlx'")
    MLX_AVAILABLE = False
    sys.exit(1)

# Import our modules
try:
    from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
    from operation_mapping import MLXDispatcher, OpCategory
    from metal_fusion_optimizer import FusionOptimizer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this from the metal module root directory")
    sys.exit(1)

# Try to import PyTorch for comparison
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Benchmark will only run MLX.")

@dataclass
class BenchmarkResult:
    """Benchmark result for a single operation"""
    operation: str
    input_shape: Tuple[int, ...]
    mlx_time: float
    torch_time: Optional[float] = None
    speedup: Optional[float] = None
    
    def __str__(self) -> str:
        """Get string representation"""
        torch_str = f"{self.torch_time:.6f}s" if self.torch_time is not None else "N/A"
        speedup_str = f"{self.speedup:.2f}x" if self.speedup is not None else "N/A"
        
        return (
            f"{self.operation:<15} | Shape: {str(self.input_shape):<20} | "
            f"MLX: {self.mlx_time:.6f}s | PyTorch: {torch_str} | Speedup: {speedup_str}"
        )

class MetalBackendBenchmark:
    """Benchmark the Metal backend for Triton on Apple Silicon"""
    
    def __init__(self, runs: int = 5, warmup: int = 2, sizes: List[int] = None):
        """
        Initialize benchmark
        
        Args:
            runs: Number of benchmark runs
            warmup: Number of warmup runs
            sizes: List of sizes to benchmark
        """
        self.runs = runs
        self.warmup = warmup
        self.sizes = sizes or [128, 512, 1024, 2048, 4096]
        self.dispatcher = MLXDispatcher(hardware_capabilities)
        self.fusion_optimizer = FusionOptimizer(hardware_capabilities)
        
        self.device_name = f"Apple {hardware_capabilities.chip_generation.name} GPU"
        print(f"Benchmarking on {self.device_name}")
        
        # Create result storage
        self.results = []
        
        # Print hardware info
        print(f"\nMetal Feature Set: {hardware_capabilities.feature_set.name}")
        print(f"GPU Family: {hardware_capabilities.gpu_family}")
        print(f"SIMD Width: {hardware_capabilities.simd_width}")
        print(f"Max Threads per Threadgroup: {hardware_capabilities.max_threads_per_threadgroup}")
        print(f"Max Threadgroups per Grid: {hardware_capabilities.max_threadgroups_per_grid}")
        print(f"Shared Memory Size: {hardware_capabilities.shared_memory_size} bytes")
    
    def _benchmark_mlx(self, func, *args, **kwargs) -> float:
        """
        Benchmark an MLX function
        
        Args:
            func: MLX function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Average execution time in seconds
        """
        # Warm up
        for _ in range(self.warmup):
            result = func(*args, **kwargs)
            mx.eval(result)
        
        # Benchmark
        start_time = time.time()
        for _ in range(self.runs):
            result = func(*args, **kwargs)
            mx.eval(result)
        end_time = time.time()
        
        return (end_time - start_time) / self.runs
    
    def _benchmark_torch(self, func, *args, **kwargs) -> float:
        """
        Benchmark a PyTorch function
        
        Args:
            func: PyTorch function to benchmark
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Average execution time in seconds
        """
        if not TORCH_AVAILABLE:
            return None
        
        # Set up CUDA stream for accurate timing
        torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Move args to device
        device_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device_args.append(arg.to(torch_device))
            else:
                device_args.append(arg)
                
        # Warm up
        for _ in range(self.warmup):
            result = func(*device_args, **kwargs)
            if torch_device.type == "mps":
                torch.mps.synchronize()
            
        # Benchmark
        start_time = time.time()
        for _ in range(self.runs):
            result = func(*device_args, **kwargs)
            if torch_device.type == "mps":
                torch.mps.synchronize()
        end_time = time.time()
        
        return (end_time - start_time) / self.runs
    
    def benchmark_matmul(self):
        """Benchmark matrix multiplication"""
        print("\nBenchmarking Matrix Multiplication...")
        
        for size in self.sizes:
            # Create input tensors
            mlx_a = mx.random.normal((size, size))
            mlx_b = mx.random.normal((size, size))
            
            # MLX benchmark
            mlx_time = self._benchmark_mlx(mx.matmul, mlx_a, mlx_b)
            
            # PyTorch benchmark
            torch_time = None
            if TORCH_AVAILABLE:
                torch_a = torch.randn((size, size))
                torch_b = torch.randn((size, size))
                torch_time = self._benchmark_torch(torch.matmul, torch_a, torch_b)
            
            # Compute speedup
            speedup = torch_time / mlx_time if torch_time is not None else None
            
            # Store result
            result = BenchmarkResult(
                operation="matmul",
                input_shape=(size, size),
                mlx_time=mlx_time,
                torch_time=torch_time,
                speedup=speedup
            )
            self.results.append(result)
            print(result)
    
    def benchmark_elementwise(self):
        """Benchmark elementwise operations"""
        print("\nBenchmarking Elementwise Operations...")
        
        for op_name, mlx_op, torch_op in [
            ("add", mx.add, torch.add if TORCH_AVAILABLE else None),
            ("mul", mx.multiply, torch.mul if TORCH_AVAILABLE else None),
            ("exp", mx.exp, torch.exp if TORCH_AVAILABLE else None),
            ("tanh", mx.tanh, torch.tanh if TORCH_AVAILABLE else None),
        ]:
            # Skip if torch op not available
            if op_name != "add" and not TORCH_AVAILABLE:
                continue
                
            for size in self.sizes:
                # Create input tensors
                mlx_a = mx.random.normal((size, size))
                mlx_b = mx.random.normal((size, size)) if op_name in ["add", "mul"] else None
                
                # MLX benchmark
                if mlx_b is not None:
                    mlx_time = self._benchmark_mlx(mlx_op, mlx_a, mlx_b)
                else:
                    mlx_time = self._benchmark_mlx(mlx_op, mlx_a)
                
                # PyTorch benchmark
                torch_time = None
                if TORCH_AVAILABLE:
                    torch_a = torch.randn((size, size))
                    torch_b = torch.randn((size, size)) if op_name in ["add", "mul"] else None
                    
                    if torch_b is not None:
                        torch_time = self._benchmark_torch(torch_op, torch_a, torch_b)
                    else:
                        torch_time = self._benchmark_torch(torch_op, torch_a)
                
                # Compute speedup
                speedup = torch_time / mlx_time if torch_time is not None else None
                
                # Store result
                result = BenchmarkResult(
                    operation=op_name,
                    input_shape=(size, size),
                    mlx_time=mlx_time,
                    torch_time=torch_time,
                    speedup=speedup
                )
                self.results.append(result)
                print(result)
    
    def benchmark_reduction(self):
        """Benchmark reduction operations"""
        print("\nBenchmarking Reduction Operations...")
        
        for op_name, mlx_op, torch_op in [
            ("sum", mx.sum, torch.sum if TORCH_AVAILABLE else None),
            ("mean", mx.mean, torch.mean if TORCH_AVAILABLE else None),
        ]:
            for size in self.sizes:
                # Create input tensors
                mlx_a = mx.random.normal((size, size))
                
                # MLX benchmark
                mlx_time = self._benchmark_mlx(mlx_op, mlx_a)
                
                # PyTorch benchmark
                torch_time = None
                if TORCH_AVAILABLE:
                    torch_a = torch.randn((size, size))
                    torch_time = self._benchmark_torch(torch_op, torch_a)
                
                # Compute speedup
                speedup = torch_time / mlx_time if torch_time is not None else None
                
                # Store result
                result = BenchmarkResult(
                    operation=op_name,
                    input_shape=(size, size),
                    mlx_time=mlx_time,
                    torch_time=torch_time,
                    speedup=speedup
                )
                self.results.append(result)
                print(result)
    
    def benchmark_softmax(self):
        """Benchmark softmax"""
        print("\nBenchmarking Softmax...")
        
        for size in self.sizes:
            # Create input tensors
            mlx_a = mx.random.normal((size, size))
            
            # MLX benchmark
            mlx_time = self._benchmark_mlx(mx.softmax, mlx_a)
            
            # PyTorch benchmark
            torch_time = None
            if TORCH_AVAILABLE:
                torch_a = torch.randn((size, size))
                torch_time = self._benchmark_torch(torch.nn.functional.softmax, torch_a, dim=-1)
            
            # Compute speedup
            speedup = torch_time / mlx_time if torch_time is not None else None
            
            # Store result
            result = BenchmarkResult(
                operation="softmax",
                input_shape=(size, size),
                mlx_time=mlx_time,
                torch_time=torch_time,
                speedup=speedup
            )
            self.results.append(result)
            print(result)
    
    def benchmark_attention(self):
        """Benchmark attention mechanism"""
        print("\nBenchmarking Attention Mechanism...")
        
        for size in self.sizes:
            batch_size = 1
            seq_len = size
            hidden_dim = 512
            num_heads = 8
            head_dim = hidden_dim // num_heads
            
            # Create input tensors
            mlx_q = mx.random.normal((batch_size, seq_len, hidden_dim))
            mlx_k = mx.random.normal((batch_size, seq_len, hidden_dim))
            mlx_v = mx.random.normal((batch_size, seq_len, hidden_dim))
            
            # Define the MLX function (using nn.MultiHeadAttention for MLX)
            def mlx_attention(q, k, v):
                mha = nn.MultiHeadAttention(hidden_dim, num_heads)
                return mha(q, k, v)
            
            # MLX benchmark
            mlx_time = self._benchmark_mlx(mlx_attention, mlx_q, mlx_k, mlx_v)
            
            # PyTorch benchmark
            torch_time = None
            if TORCH_AVAILABLE:
                torch_q = torch.randn((batch_size, seq_len, hidden_dim))
                torch_k = torch.randn((batch_size, seq_len, hidden_dim))
                torch_v = torch.randn((batch_size, seq_len, hidden_dim))
                
                def torch_scaled_dot_product(q, k, v):
                    # Manual implementation of scaled dot-product attention
                    d_k = q.size(-1)
                    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
                    attn = torch.nn.functional.softmax(scores, dim=-1)
                    return torch.matmul(attn, v)
                
                torch_time = self._benchmark_torch(torch_scaled_dot_product, torch_q, torch_k, torch_v)
            
            # Compute speedup
            speedup = torch_time / mlx_time if torch_time is not None else None
            
            # Store result
            result = BenchmarkResult(
                operation="attention",
                input_shape=(batch_size, seq_len, hidden_dim),
                mlx_time=mlx_time,
                torch_time=torch_time,
                speedup=speedup
            )
            self.results.append(result)
            print(result)

    def benchmark_autotuning(self):
        """Benchmark auto-tuning effectiveness"""
        print("\nBenchmarking Auto-Tuning Effectiveness...")
        
        # Import auto-tuner
        try:
            from metal_auto_tuner import (
                MetalAutoTuner,
                TunableParam,
                ParamType,
                ConfigurationResult,
                get_matmul_metal_params
            )
            
            # Create dummy class to hold triton-like interface for testing
            class DummyTriton:
                @staticmethod
                def cdiv(x, y):
                    return (x + y - 1) // y
            
            triton = DummyTriton()
            
            # Define simple matrix multiplication function
            def matmul_untuned(a, b, c, M, N, K):
                # Simple implementation with fixed parameters
                block_m, block_n, block_k = 64, 64, 32
                num_warps = 4
                num_stages = 2
                
                # MLX implementation
                c[:] = mx.matmul(a, b)
                return c
            
            def matmul_tuned(a, b, c, M, N, K, config=None):
                # Implementation with tuned parameters
                if config is None:
                    block_m, block_n, block_k = 64, 64, 32
                    num_warps = 4
                    num_stages = 2
                else:
                    block_m = config.get("block_m", 64)
                    block_n = config.get("block_n", 64)
                    block_k = config.get("block_k", 32)
                    num_warps = config.get("num_warps", 4)
                    num_stages = config.get("num_stages", 2)
                
                # Use the dispatcher to find optimized implementation based on config
                c[:] = self.dispatcher.dispatch_matmul(a, b, hardware_specific_config={
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_k": block_k,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                    "chip_generation": hardware_capabilities.chip_generation
                })
                return c
            
            # Create auto-tuner
            matmul_params = get_matmul_metal_params()
            
            for size in [512, 1024, 2048]:
                M, N, K = size, size, size
                
                # Create input and output tensors
                a_data = mx.random.normal((M, K))
                b_data = mx.random.normal((K, N))
                c_untuned = mx.zeros((M, N))
                c_tuned = mx.zeros((M, N))
                
                # Benchmark untuned implementation
                def run_untuned():
                    return matmul_untuned(a_data, b_data, c_untuned, M, N, K)
                
                untuned_time = self._benchmark_mlx(run_untuned)
                
                # Perform auto-tuning
                print(f"  Auto-tuning matmul for size {M}x{N}x{K}...")
                tuner = MetalAutoTuner(
                    f"matmul_{M}_{N}_{K}",
                    matmul_params,
                    n_trials=10,  # Limited trials for benchmark
                    search_strategy="random"
                )
                
                # Define evaluation function
                def evaluate_config(config):
                    try:
                        c_eval = mx.zeros((M, N))
                        
                        # Time the kernel execution with this config
                        start_time = time.time()
                        matmul_tuned(a_data, b_data, c_eval, M, N, K, config)
                        mx.eval(c_eval)
                        end_time = time.time()
                        runtime_ms = (end_time - start_time) * 1000
                        
                        return ConfigurationResult(
                            config=config,
                            runtime_ms=runtime_ms,
                            success=True
                        )
                    except Exception as e:
                        return ConfigurationResult(
                            config=config,
                            runtime_ms=float("inf"),
                            success=False,
                            metrics={"error": str(e)}
                        )
                
                # Run tuning
                best_config = tuner.tune(evaluate_config, max_trials=10)
                
                # Benchmark tuned implementation
                def run_tuned():
                    return matmul_tuned(a_data, b_data, c_tuned, M, N, K, best_config)
                
                tuned_time = self._benchmark_mlx(run_tuned)
                
                # Calculate speedup
                speedup = untuned_time / tuned_time
                
                print(f"  Matrix Size: {M}x{N}x{K}")
                print(f"  Untuned: {untuned_time:.6f}s")
                print(f"  Tuned  : {tuned_time:.6f}s")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Best config: {best_config}")
                print()
                
                # Store result
                result = BenchmarkResult(
                    operation="matmul_autotuned",
                    input_shape=(M, N, K),
                    mlx_time=tuned_time,
                    torch_time=untuned_time,  # Reusing this field for untuned time
                    speedup=speedup
                )
                self.results.append(result)
        
        except ImportError as e:
            print(f"  Auto-tuning benchmark skipped: {e}")

    def benchmark_all(self):
        """Run all benchmarks"""
        self.benchmark_matmul()
        self.benchmark_elementwise()
        self.benchmark_reduction()
        self.benchmark_softmax()
        self.benchmark_attention()
        self.benchmark_autotuning()  # Added auto-tuning benchmark
        self.generate_plots()
    
    def generate_plots(self):
        """Generate performance comparison plots"""
        # Create directory for plots
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Group results by operation
        ops = {}
        for r in self.results:
            if r.operation not in ops:
                ops[r.operation] = []
            ops[r.operation].append(r)
        
        # Plot for each operation
        for op_name, results in ops.items():
            # Filter results with both MLX and PyTorch times
            valid_results = [r for r in results if r.torch_time is not None]
            
            if not valid_results:
                continue
            
            # Sort by input size
            valid_results.sort(key=lambda r: np.prod(r.input_shape))
            
            # Extract data
            labels = [f"{r.input_shape}" for r in valid_results]
            mlx_times = [r.mlx_time for r in valid_results]
            torch_times = [r.torch_time for r in valid_results]
            
            # Create figure
            plt.figure(figsize=(10, 6))
            x = np.arange(len(labels))
            width = 0.35
            
            # Plot bars
            plt.bar(x - width/2, mlx_times, width, label='MLX')
            plt.bar(x + width/2, torch_times, width, label='PyTorch')
            
            # Add labels and title
            plt.xlabel('Input Shape')
            plt.ylabel('Time (seconds)')
            plt.title(f'Performance Comparison for {op_name}')
            plt.xticks(x, labels, rotation=45)
            plt.legend()
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{op_name}_comparison.png"))
            plt.close()
        
        # Create speedup plot
        valid_results = [r for r in self.results if r.speedup is not None]
        if valid_results:
            operations = sorted(set(r.operation for r in valid_results))
            speedups = []
            
            for op in operations:
                op_results = [r for r in valid_results if r.operation == op]
                avg_speedup = sum(r.speedup for r in op_results) / len(op_results)
                speedups.append(avg_speedup)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            plt.bar(operations, speedups)
            
            # Add labels and title
            plt.xlabel('Operation')
            plt.ylabel('Speedup (PyTorch/MLX)')
            plt.title('Average Speedup by Operation')
            plt.xticks(rotation=45)
            
            # Add reference line at y=1
            plt.axhline(y=1, color='r', linestyle='-')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "speedup_by_operation.png"))
            plt.close()

def main():
    """Run benchmarks"""
    parser = argparse.ArgumentParser(description="Metal Backend Benchmark")
    parser.add_argument("-r", "--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("-w", "--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("-s", "--sizes", type=int, nargs="+", default=[128, 512, 1024, 2048],
                        help="Sizes to benchmark")
    parser.add_argument("-o", "--operation", type=str, choices=[
                        "all", "matmul", "elementwise", "reduction", "softmax", "attention"],
                        default="all", help="Operation to benchmark")
    
    args = parser.parse_args()
    
    benchmark = MetalBackendBenchmark(
        runs=args.runs,
        warmup=args.warmup,
        sizes=args.sizes
    )
    
    if args.operation == "all":
        benchmark.benchmark_all()
    elif args.operation == "matmul":
        benchmark.benchmark_matmul()
    elif args.operation == "elementwise":
        benchmark.benchmark_elementwise()
    elif args.operation == "reduction":
        benchmark.benchmark_reduction()
    elif args.operation == "softmax":
        benchmark.benchmark_softmax()
    elif args.operation == "attention":
        benchmark.benchmark_attention()

if __name__ == "__main__":
    main() 