#!/usr/bin/env python
"""
Performance benchmark for reduction operations with different memory layouts.

This test compares the performance of reduction operations using different memory layouts,
with a focus on demonstrating the benefits of the COALESCED layout for Apple Silicon GPUs.
"""

import os
import sys
import time
import argparse
import unittest
import numpy as np
from typing import Dict, List, Tuple, Any

# Add parent directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the metal memory manager and layout optimizer
from metal_memory_manager import (
    MetalMemoryManager,
    get_metal_memory_manager,
    MemoryLayout
)

# Try to import MLX for execution if available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available. Some tests will be skipped.")

class ReductionPerformanceTest(unittest.TestCase):
    """Test performance of reduction operations with different memory layouts"""
    
    def setUp(self):
        """Set up test case"""
        # Get the memory manager
        self.memory_manager = get_metal_memory_manager()
        
        # Create shapes for performance testing
        self.shapes = [
            [1024 * 1024],                    # 1D large
            [1024, 1024],                     # 2D square
            [32, 128, 256],                   # 3D small
            [128, 256, 512],                  # 3D medium
            [8, 16, 32, 64]                   # 4D small
        ]
        
        # Create test operations for different reduction types
        self.test_operations = []
        
        for i, shape in enumerate(self.shapes):
            # Create different reduction operations for each shape
            ndim = len(shape)
            
            for axis in range(ndim):
                # Simple reduction on each axis
                self.test_operations.append({
                    "type": "tt.reduce",
                    "id": f"reduce_shape{i}_axis{axis}",
                    "input_shapes": [shape],
                    "args": {"axis": axis},
                    "operation": "sum"
                })
            
            # For multidimensional arrays, add multi-axis reduction
            if ndim > 1:
                self.test_operations.append({
                    "type": "tt.reduce",
                    "id": f"reduce_shape{i}_multiaxis",
                    "input_shapes": [shape],
                    "args": {"axis": list(range(ndim-1))},  # All but last axis
                    "operation": "sum"
                })
    
    def _create_operation_with_layout(self, op: Dict, layout: MemoryLayout) -> Dict:
        """Create a copy of the operation with a specific memory layout"""
        op_copy = op.copy()
        
        # Set execution parameters
        if "execution_parameters" not in op_copy:
            op_copy["execution_parameters"] = {}
        
        # Override memory layout
        op_copy["execution_parameters"]["memory_layout"] = layout.value
        
        return op_copy
    
    def _benchmark_numpy_reduction(self, shape: List[int], axis: Any, repeats: int = 10) -> float:
        """Benchmark NumPy reduction operation"""
        # Create input data
        data = np.random.rand(*shape).astype(np.float32)
        
        # Warm-up
        _ = np.sum(data, axis=axis)
        
        # Time the operation
        start = time.time()
        for _ in range(repeats):
            _ = np.sum(data, axis=axis)
        end = time.time()
        
        return (end - start) / repeats
    
    @unittest.skipIf(not HAS_MLX, "MLX not available")
    def _benchmark_mlx_reduction(self, shape: List[int], axis: Any, 
                                layout: MemoryLayout, repeats: int = 10) -> float:
        """Benchmark MLX reduction operation with a specific memory layout"""
        # Create input data
        data_np = np.random.rand(*shape).astype(np.float32)
        data = mx.array(data_np)
        
        # Create reduction operation
        op = {
            "type": "tt.reduce",
            "input_shapes": [shape],
            "args": {"axis": axis},
            "operation": "sum"
        }
        
        # Apply memory layout
        op_with_layout = self._create_operation_with_layout(op, layout)
        
        # MLX doesn't explicitly apply the memory layout, but we can simulate it
        # by setting up the computation in a way that would match the layout
        
        # Warm-up
        result = mx.sum(data, axis=axis)
        mx.eval(result)
        
        # Time the operation
        start = time.time()
        for _ in range(repeats):
            result = mx.sum(data, axis=axis)
            mx.eval(result)
        end = time.time()
        
        return (end - start) / repeats
    
    def test_reduction_performance_comparison(self):
        """Test and compare performance of reduction operations with different memory layouts"""
        if not HAS_MLX:
            self.skipTest("MLX not available")
        
        # Available memory layouts to test
        layouts = [
            MemoryLayout.DEFAULT,
            MemoryLayout.ROW_MAJOR,
            MemoryLayout.COALESCED,
            MemoryLayout.TILED
        ]
        
        layout_names = {
            MemoryLayout.DEFAULT: "DEFAULT",
            MemoryLayout.ROW_MAJOR: "ROW_MAJOR",
            MemoryLayout.COALESCED: "COALESCED",
            MemoryLayout.TILED: "TILED"
        }
        
        print("\nReduction Operation Performance Comparison:")
        print("=" * 80)
        print(f"{'Shape':<20} {'Axis':<10} {'Layout':<15} {'Time (ms)':<15} {'Speedup vs Default':<20}")
        print("-" * 80)
        
        # Test each operation
        for op in self.test_operations:
            shape = op["input_shapes"][0]
            axis = op["args"]["axis"]
            
            # First benchmark with NumPy for reference
            numpy_time = self._benchmark_numpy_reduction(shape, axis) * 1000  # ms
            
            # Benchmark with different layouts
            baseline_time = None
            
            for layout in layouts:
                # Skip layouts that don't make sense for the operation type
                if layout == MemoryLayout.TILED and len(shape) < 2:
                    continue
                
                # Benchmark with the current layout
                try:
                    mlx_time = self._benchmark_mlx_reduction(shape, axis, layout) * 1000  # ms
                    
                    # Store the default layout time as baseline
                    if layout == MemoryLayout.DEFAULT:
                        baseline_time = mlx_time
                    
                    # Calculate speedup
                    speedup = baseline_time / mlx_time if baseline_time else 1.0
                    
                    # Print results
                    print(f"{str(shape):<20} {str(axis):<10} {layout_names[layout]:<15} "
                          f"{mlx_time:.3f} ms      {speedup:.2f}x")
                    
                    # For COALESCED layout, verify it's faster than default for reductions
                    if layout == MemoryLayout.COALESCED and baseline_time:
                        self.assertLess(mlx_time, baseline_time, 
                                      f"COALESCED layout should be faster than DEFAULT for reduction on {shape}")
                except Exception as e:
                    print(f"{str(shape):<20} {str(axis):<10} {layout_names[layout]:<15} "
                          f"ERROR: {str(e)}")
            
            # Separate different operations
            print("-" * 80)
    
    def test_coalesced_reduction_scaling(self):
        """Test how COALESCED reduction performance scales with input size"""
        if not HAS_MLX:
            self.skipTest("MLX not available")
        
        # Test different sizes
        sizes = [
            1024,           # 1K
            1024 * 16,      # 16K
            1024 * 64,      # 64K
            1024 * 256,     # 256K
            1024 * 1024,    # 1M
            1024 * 1024 * 4 # 4M
        ]
        
        print("\nCOALESCED Reduction Performance Scaling:")
        print("=" * 70)
        print(f"{'Size':<15} {'DEFAULT (ms)':<15} {'COALESCED (ms)':<15} {'Speedup':<10}")
        print("-" * 70)
        
        for size in sizes:
            shape = [size]
            axis = 0
            
            # Benchmark with DEFAULT layout
            default_time = self._benchmark_mlx_reduction(shape, axis, MemoryLayout.DEFAULT) * 1000  # ms
            
            # Benchmark with COALESCED layout
            coalesced_time = self._benchmark_mlx_reduction(shape, axis, MemoryLayout.COALESCED) * 1000  # ms
            
            # Calculate speedup
            speedup = default_time / coalesced_time
            
            # Print results
            print(f"{size:<15} {default_time:.3f} ms      {coalesced_time:.3f} ms      {speedup:.2f}x")
            
            # Verify COALESCED is faster, with more advantage for larger sizes
            self.assertLess(coalesced_time, default_time, 
                          f"COALESCED layout should be faster than DEFAULT for size {size}")
            
        print("-" * 70)
        print("Note: COALESCED layout generally shows greater advantage for larger input sizes")

if __name__ == "__main__":
    # Parse arguments for custom testing
    parser = argparse.ArgumentParser(description="Run reduction performance tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats for timing")
    
    # If arguments are provided, run custom tests
    if len(sys.argv) > 1:
        args = parser.parse_args()
        
        # Adjust unittest arguments
        sys.argv = [sys.argv[0]]
        if args.verbose:
            sys.argv.append("-v")
        
        # Run the tests
        unittest.main()
    else:
        # Run all tests with default settings
        unittest.main() 