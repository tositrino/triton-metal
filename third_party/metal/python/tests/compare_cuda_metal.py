#!/usr/bin/env python
"""Comparison tool for Metal and CUDA backends.

This script helps compare the performance of Metal backend on Apple Silicon 
with CUDA backend on NVIDIA GPUs by loading and visualizing benchmark results.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

def load_benchmark_results(results_file: str) -> Dict[str, Any]:
    """Load benchmark results from a JSON file
    
    Args:
        results_file: Path to the JSON results file
        
    Returns:
        Dictionary containing the benchmark results
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    return results

def compare_results(metal_results: Dict[str, Any], cuda_results: Dict[str, Any], 
                    output_dir: str) -> None:
    """Compare Metal and CUDA benchmark results
    
    Args:
        metal_results: Metal benchmark results
        cuda_results: CUDA benchmark results
        output_dir: Directory to save comparison plots
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by operation type
    metal_ops = group_by_operation(metal_results["results"])
    cuda_ops = group_by_operation(cuda_results["results"])
    
    # Find common operations
    common_ops = set(metal_ops.keys()).intersection(set(cuda_ops.keys()))
    
    for op_name in common_ops:
        metal_op_results = metal_ops[op_name]
        cuda_op_results = cuda_ops[op_name]
        
        # Compare execution time
        plot_time_comparison(op_name, metal_op_results, cuda_op_results, output_dir)
        
        # Compare GFLOPS if available
        if all("gflops" in r for r in metal_op_results) and all("gflops" in r for r in cuda_op_results):
            plot_flops_comparison(op_name, metal_op_results, cuda_op_results, output_dir)
            
            # Calculate speedup/slowdown
            plot_speedup(op_name, metal_op_results, cuda_op_results, output_dir)

def group_by_operation(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group benchmark results by operation type
    
    Args:
        results: List of benchmark results
        
    Returns:
        Dictionary mapping operation names to lists of results
    """
    ops = {}
    for result in results:
        op_name = result["name"].split("_")[0]
        if op_name not in ops:
            ops[op_name] = []
        ops[op_name].append(result)
    return ops

def plot_time_comparison(op_name: str, metal_results: List[Dict[str, Any]], 
                         cuda_results: List[Dict[str, Any]], output_dir: str) -> None:
    """Plot execution time comparison between Metal and CUDA
    
    Args:
        op_name: Operation name
        metal_results: Metal benchmark results for the operation
        cuda_results: CUDA benchmark results for the operation
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Group by dtype
    metal_by_dtype = group_by_dtype(metal_results)
    cuda_by_dtype = group_by_dtype(cuda_results)
    
    # Common dtypes
    common_dtypes = set(metal_by_dtype.keys()).intersection(set(cuda_by_dtype.keys()))
    
    # Plot each dtype
    for dtype in common_dtypes:
        metal_dtype_results = metal_by_dtype[dtype]
        cuda_dtype_results = cuda_by_dtype[dtype]
        
        # Sort by size
        metal_dtype_results.sort(key=lambda r: r["size"][0])
        cuda_dtype_results.sort(key=lambda r: r["size"][0])
        
        # Extract sizes and times
        metal_sizes = [r["size"][0] for r in metal_dtype_results]  # First dimension
        metal_times = [r["avg_time_ms"] for r in metal_dtype_results]
        
        cuda_sizes = [r["size"][0] for r in cuda_dtype_results]
        cuda_times = [r["avg_time_ms"] for r in cuda_dtype_results]
        
        # Plot
        plt.plot(metal_sizes, metal_times, marker="o", linestyle="-", label=f"Metal {dtype}")
        plt.plot(cuda_sizes, cuda_times, marker="s", linestyle="--", label=f"CUDA {dtype}")
    
    plt.title(f"{op_name} Execution Time: Metal vs CUDA")
    plt.xlabel("Matrix Size (N for NxN)")
    plt.ylabel("Execution Time (ms)")
    plt.xscale("log2")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{op_name}_time_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_flops_comparison(op_name: str, metal_results: List[Dict[str, Any]], 
                         cuda_results: List[Dict[str, Any]], output_dir: str) -> None:
    """Plot GFLOPS comparison between Metal and CUDA
    
    Args:
        op_name: Operation name
        metal_results: Metal benchmark results for the operation
        cuda_results: CUDA benchmark results for the operation
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Group by dtype
    metal_by_dtype = group_by_dtype(metal_results)
    cuda_by_dtype = group_by_dtype(cuda_results)
    
    # Common dtypes
    common_dtypes = set(metal_by_dtype.keys()).intersection(set(cuda_by_dtype.keys()))
    
    # Plot each dtype
    for dtype in common_dtypes:
        metal_dtype_results = metal_by_dtype[dtype]
        cuda_dtype_results = cuda_by_dtype[dtype]
        
        # Sort by size
        metal_dtype_results.sort(key=lambda r: r["size"][0])
        cuda_dtype_results.sort(key=lambda r: r["size"][0])
        
        # Extract sizes and GFLOPS
        metal_sizes = [r["size"][0] for r in metal_dtype_results]
        metal_gflops = [r["gflops"] for r in metal_dtype_results]
        
        cuda_sizes = [r["size"][0] for r in cuda_dtype_results]
        cuda_gflops = [r["gflops"] for r in cuda_dtype_results]
        
        # Plot
        plt.plot(metal_sizes, metal_gflops, marker="o", linestyle="-", label=f"Metal {dtype}")
        plt.plot(cuda_sizes, cuda_gflops, marker="s", linestyle="--", label=f"CUDA {dtype}")
    
    plt.title(f"{op_name} Performance: Metal vs CUDA")
    plt.xlabel("Matrix Size (N for NxN)")
    plt.ylabel("GFLOPS")
    plt.xscale("log2")
    plt.grid(True)
    plt.legend()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{op_name}_flops_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_speedup(op_name: str, metal_results: List[Dict[str, Any]], 
                cuda_results: List[Dict[str, Any]], output_dir: str) -> None:
    """Plot speedup/slowdown of Metal relative to CUDA
    
    Args:
        op_name: Operation name
        metal_results: Metal benchmark results for the operation
        cuda_results: CUDA benchmark results for the operation
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Group by dtype
    metal_by_dtype = group_by_dtype(metal_results)
    cuda_by_dtype = group_by_dtype(cuda_results)
    
    # Common dtypes
    common_dtypes = set(metal_by_dtype.keys()).intersection(set(cuda_by_dtype.keys()))
    
    # Plot each dtype
    for dtype in common_dtypes:
        metal_dtype_results = metal_by_dtype[dtype]
        cuda_dtype_results = cuda_by_dtype[dtype]
        
        # Create size-to-time mapping for easy lookup
        metal_time_map = {tuple(r["size"]): r["avg_time_ms"] for r in metal_dtype_results}
        cuda_time_map = {tuple(r["size"]): r["avg_time_ms"] for r in cuda_dtype_results}
        
        # Find common sizes
        common_sizes = set(metal_time_map.keys()).intersection(set(cuda_time_map.keys()))
        
        if not common_sizes:
            continue
        
        # Calculate speedup for each common size
        sizes = []
        speedups = []
        for size in sorted(common_sizes, key=lambda s: s[0]):
            metal_time = metal_time_map[size]
            cuda_time = cuda_time_map[size]
            speedup = cuda_time / metal_time
            sizes.append(size[0])  # First dimension
            speedups.append(speedup)
        
        # Plot
        plt.plot(sizes, speedups, marker="o", label=f"{dtype}")
        
        # Add horizontal line at y=1 (equal performance)
        plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    
    plt.title(f"{op_name} Speedup: Metal vs CUDA")
    plt.xlabel("Matrix Size (N for NxN)")
    plt.ylabel("Speedup (CUDA time / Metal time)")
    plt.xscale("log2")
    plt.grid(True)
    plt.legend()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{op_name}_speedup.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def group_by_dtype(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group benchmark results by data type
    
    Args:
        results: List of benchmark results
        
    Returns:
        Dictionary mapping dtype names to lists of results
    """
    dtypes = {}
    for result in results:
        dtype = result["dtype"]
        if dtype not in dtypes:
            dtypes[dtype] = []
        dtypes[dtype].append(result)
    return dtypes

def generate_summary_table(metal_results: Dict[str, Any], cuda_results: Dict[str, Any],
                          output_file: str) -> None:
    """Generate a summary table comparing Metal and CUDA
    
    Args:
        metal_results: Metal benchmark results
        cuda_results: CUDA benchmark results
        output_file: Path to save the summary table
    """
    # Group results by operation type
    metal_ops = group_by_operation(metal_results["results"])
    cuda_ops = group_by_operation(cuda_results["results"])
    
    # Find common operations
    common_ops = set(metal_ops.keys()).intersection(set(cuda_ops.keys()))
    
    # Prepare summary data
    summary = []
    for op_name in common_ops:
        metal_op_results = metal_ops[op_name]
        cuda_op_results = cuda_ops[op_name]
        
        # Group by dtype
        metal_by_dtype = group_by_dtype(metal_op_results)
        cuda_by_dtype = group_by_dtype(cuda_op_results)
        
        # Common dtypes
        common_dtypes = set(metal_by_dtype.keys()).intersection(set(cuda_by_dtype.keys()))
        
        for dtype in common_dtypes:
            metal_dtype_results = metal_by_dtype[dtype]
            cuda_dtype_results = cuda_by_dtype[dtype]
            
            # Create size-to-time mapping for easy lookup
            metal_time_map = {tuple(r["size"]): r["avg_time_ms"] for r in metal_dtype_results}
            cuda_time_map = {tuple(r["size"]): r["avg_time_ms"] for r in cuda_dtype_results}
            
            # Create size-to-flops mapping if available
            metal_gflops_map = {}
            cuda_gflops_map = {}
            if all("gflops" in r for r in metal_dtype_results) and all("gflops" in r for r in cuda_dtype_results):
                metal_gflops_map = {tuple(r["size"]): r["gflops"] for r in metal_dtype_results}
                cuda_gflops_map = {tuple(r["size"]): r["gflops"] for r in cuda_dtype_results}
            
            # Find common sizes
            common_sizes = set(metal_time_map.keys()).intersection(set(cuda_time_map.keys()))
            
            for size in sorted(common_sizes, key=lambda s: s[0]):
                metal_time = metal_time_map[size]
                cuda_time = cuda_time_map[size]
                speedup = cuda_time / metal_time
                
                entry = {
                    "op_name": op_name,
                    "dtype": dtype,
                    "size": "x".join(map(str, size)),
                    "metal_time_ms": metal_time,
                    "cuda_time_ms": cuda_time,
                    "speedup": speedup
                }
                
                if metal_gflops_map and cuda_gflops_map:
                    entry["metal_gflops"] = metal_gflops_map[size]
                    entry["cuda_gflops"] = cuda_gflops_map[size]
                
                summary.append(entry)
    
    # Write summary to CSV
    with open(output_file, "w") as f:
        # Write header
        if summary and "metal_gflops" in summary[0]:
            f.write("Operation,Dtype,Size,Metal Time (ms),CUDA Time (ms),Speedup,Metal GFLOPS,CUDA GFLOPS\n")
            for entry in summary:
                f.write(f"{entry['op_name']},{entry['dtype']},{entry['size']},"
                        f"{entry['metal_time_ms']:.2f},{entry['cuda_time_ms']:.2f},"
                        f"{entry['speedup']:.2f},{entry['metal_gflops']:.2f},{entry['cuda_gflops']:.2f}\n")
        else:
            f.write("Operation,Dtype,Size,Metal Time (ms),CUDA Time (ms),Speedup\n")
            for entry in summary:
                f.write(f"{entry['op_name']},{entry['dtype']},{entry['size']},"
                        f"{entry['metal_time_ms']:.2f},{entry['cuda_time_ms']:.2f},"
                        f"{entry['speedup']:.2f}\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare Metal and CUDA backend performance")
    parser.add_argument("--metal-results", type=str, required=True,
                        help="Path to Metal benchmark results JSON file")
    parser.add_argument("--cuda-results", type=str, required=True,
                        help="Path to CUDA benchmark results JSON file")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                        help="Directory to save comparison plots and tables")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load benchmark results
    try:
        metal_results = load_benchmark_results(args.metal_results)
        cuda_results = load_benchmark_results(args.cuda_results)
    except Exception as e:
        print(f"Error loading benchmark results: {e}")
        return 1
    
    # Compare results
    try:
        compare_results(metal_results, cuda_results, args.output_dir)
    except Exception as e:
        print(f"Error comparing results: {e}")
        return 1
    
    # Generate summary table
    try:
        summary_file = os.path.join(args.output_dir, "summary.csv")
        generate_summary_table(metal_results, cuda_results, summary_file)
        print(f"Summary table saved to {summary_file}")
    except Exception as e:
        print(f"Error generating summary table: {e}")
        return 1

    print(f"Comparison complete. Results saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())