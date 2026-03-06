#!/usr/bin/env python
"""
Performance test for special math functions.
Compares MLX implementations with numpy/scipy reference implementations.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Try to import MLX
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    print("Warning: MLX not found. Performance tests will be skipped.")
    HAS_MLX = False

# Try to import scipy
try:
    import scipy.special
    HAS_SCIPY = True
except ImportError:
    print("Warning: scipy not found. Performance tests will be limited.")
    HAS_SCIPY = False

def to_numpy(x):
    """Convert MLX array to numpy array safely"""
    try:
        return x.numpy()
    except AttributeError:
        return np.array(x.tolist())

def run_performance_comparison():
    """Run performance comparison between MLX implementations and reference"""
    if not HAS_MLX or not HAS_SCIPY:
        print("MLX or scipy not available. Skipping performance comparison.")
        return None, None
        
    try:
        from MLX.special_ops import SpecialMathFunctions, NumericalFunctions
        
        special_math = SpecialMathFunctions()
        numerical = NumericalFunctions()
        
        # Test functions to compare
        functions = [
            ("erf", special_math.erf, scipy.special.erf),
            ("lgamma", special_math.lgamma, scipy.special.gammaln),
            ("bessel_j0", special_math.bessel_j0, scipy.special.j0),
            ("fast_sigmoid", numerical.fast_sigmoid, lambda x: 1.0 / (1.0 + np.exp(-x))),
            ("fast_tanh", numerical.fast_tanh, np.tanh),
        ]
        
        # Input sizes to test
        sizes = [100, 1000, 10000, 100000]
        
        # Results dictionary
        results = {}
        
        for name, mlx_fn, ref_fn in functions:
            mlx_times = []
            ref_times = []
            
            print(f"\nPerformance testing: {name}")
            print("-" * 50)
            print(f"{'Size':<10} {'MLX Time (ms)':<15} {'Reference Time (ms)':<20} {'Speedup':<10}")
            print("-" * 50)
            
            for size in sizes:
                # Generate test data
                np_data = np.random.rand(size).astype(np.float32)
                mx_data = mx.array(np_data)
                
                # Test MLX implementation
                start = time.time()
                result = mlx_fn(mx_data)
                _ = to_numpy(result)  # Include conversion to numpy for fair comparison
                mlx_time = (time.time() - start) * 1000  # ms
                
                # Test reference implementation
                start = time.time()
                _ = ref_fn(np_data)
                ref_time = (time.time() - start) * 1000  # ms
                
                # Record results
                mlx_times.append(mlx_time)
                ref_times.append(ref_time)
                
                # Print results
                speedup = ref_time / mlx_time if mlx_time > 0 else float('inf')
                print(f"{size:<10} {mlx_time:<15.3f} {ref_time:<20.3f} {speedup:<10.2f}x")
            
            results[name] = (mlx_times, ref_times)
            
        return results, sizes
            
    except ImportError as e:
        print(f"Performance comparison skipped: {e}")
        return None, None

def plot_performance_results(results, sizes):
    """Plot performance comparison results"""
    if results is None or sizes is None:
        return
        
    try:
        # Create output directory if it doesn't exist
        os.makedirs("test_results", exist_ok=True)
        
        # Create a figure for each function
        for name, (mlx_times, ref_times) in results.items():
            plt.figure(figsize=(10, 6))
            plt.plot(sizes, mlx_times, 'o-', label='MLX Implementation')
            plt.plot(sizes, ref_times, 's-', label='Reference Implementation')
            plt.title(f'Performance Comparison: {name}')
            plt.xlabel('Array Size')
            plt.ylabel('Time (ms)')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"test_results/{name}_performance.png")
            plt.close()
            
        # Create a combined speedup figure
        plt.figure(figsize=(12, 8))
        
        for name, (mlx_times, ref_times) in results.items():
            speedups = [r/m if m > 0 else 0 for m, r in zip(mlx_times, ref_times)]
            plt.plot(sizes, speedups, 'o-', label=name)
            
        plt.title('MLX Implementation Speedup')
        plt.xlabel('Array Size')
        plt.ylabel('Speedup (Reference/MLX)')
        plt.xscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)  # Line at y=1 for reference
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig("test_results/speedup_comparison.png")
        plt.close()
            
    except ImportError as e:
        print(f"Performance plotting skipped: {e}")

def main():
    """Main function to run performance comparison"""
    print("\n=== Running Performance Comparison ===\n")
    results, sizes = run_performance_comparison()
    
    if results:
        print("\n=== Generating Performance Plots ===\n")
        plot_performance_results(results, sizes)
        print("Performance plots saved to test_results/ directory")
    else:
        print("\n=== Performance comparison failed or skipped ===\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 