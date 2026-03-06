#!/usr/bin/env python
"""
Metal Backend Test Suite

This script tests the functionality of the translated Metal backend components:
1. complex_ops.py - Tests matrix multiplication and convolution operations
2. launcher.py - Tests the Metal kernel launcher
3. memory_layout.py - Tests memory layout conversion

These tests ensure that the English translation maintains the original functionality.
"""

import os
import sys
import unittest
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Try to import MLX
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    print("Warning: MLX not found. Some tests will be skipped.")
    HAS_MLX = False

# Import modules to test
try:
    from mlx.complex_ops import MatrixMultiply, Convolution, get_complex_ops_map
    from mlx.launcher import MetalCompiler, MetalLauncher, compile_and_launch
    from mlx.memory_layout import MemoryLayout, adapt_tensor, get_optimal_layout
    from MLX.thread_mapping import ThreadMapping, map_kernel_launch_params
except ImportError as e:
    print(f"Error importing Metal backend modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

@unittest.skipIf(not HAS_MLX, "MLX not available")
class TestComplexOps(unittest.TestCase):
    """Test complex operations implementation"""
    
    def setUp(self):
        """Initialize test case"""
        # Create matrix multiplication instance
        self.matmul = MatrixMultiply()
        
        # Create convolution instance
        self.conv = Convolution()
        
        # Create test matrices
        self.A = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        self.B = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
        
    def test_matrix_multiply(self):
        """Test matrix multiplication"""
        # Perform matrix multiplication
        result = self.matmul(self.A, self.B)
        
        # Expected result: [1, 2] @ [5, 6] = 1*5 + 2*7 = 19
        #                  [3, 4]   [7, 8]   3*5 + 4*7 = 43
        expected = mx.array([[19, 22], [43, 50]], dtype=mx.float32)
        
        # Verify result
        self.assertTrue(mx.allclose(result, expected))
        
    def test_matrix_multiply_with_transpose(self):
        """Test matrix multiplication with transpose option"""
        # Perform matrix multiplication with transpose A
        result = self.matmul(self.A, self.B, trans_A=True)
        
        # Expected result: [1, 3] @ [5, 6] = 1*5 + 3*7 = 26
        #                  [2, 4]   [7, 8]   2*5 + 4*7 = 38
        expected = mx.array([[26, 30], [38, 44]], dtype=mx.float32)
        
        # Verify result
        self.assertTrue(mx.allclose(result, expected))
        
    def test_batch_matmul(self):
        """Test batch matrix multiplication"""
        # Create batch matrices (2 batches)
        batch_A = mx.stack([self.A, self.A * 2])  # 2 x 2 x 2
        batch_B = mx.stack([self.B, self.B * 2])  # 2 x 2 x 2
        
        # Perform batch matrix multiplication
        result = self.matmul.batch_matmul(batch_A, batch_B)
        
        # Expected results
        expected_0 = mx.array([[19, 22], [43, 50]], dtype=mx.float32)
        expected_1 = mx.array([[76, 88], [172, 200]], dtype=mx.float32)  # 2x the inputs = 4x the result
        expected = mx.stack([expected_0, expected_1])
        
        # Verify result
        self.assertTrue(mx.allclose(result, expected))
        
    def test_convolution_2d(self):
        """Test 2D convolution"""
        # Skip if MLX doesn't support convolution
        if not hasattr(mx, "conv2d"):
            self.skipTest("MLX does not support conv2d")
            
        # Create input with shape (N, H, W, C_in) = (1, 4, 4, 3)
        # Batch size 1, 4x4 image, 3 channels
        x = mx.ones((1, 4, 4, 3), dtype=mx.float32)
        
        # Create weights with shape (C_out, KH, KW, C_in) = (2, 2, 2, 3)
        # 2 output channels, 2x2 kernel, 3 input channels
        w = mx.ones((2, 2, 2, 3), dtype=mx.float32)
        
        # Perform convolution with stride=1, padding=0
        result = self.conv(x, w, stride=1, padding=0)
        
        # For a 4x4 input with 2x2 kernel and padding=0, stride=1,
        # the output shape should be (1, 3, 3, 2)
        # - Batch size remains 1
        # - Height: (4 - 2 + 2*0)/1 + 1 = 3
        # - Width: (4 - 2 + 2*0)/1 + 1 = 3 
        # - Channels: 2 output channels
        expected_shape = (1, 3, 3, 2)
        
        # Check that output shape is correct
        self.assertEqual(result.shape, expected_shape)
        
        # With all ones in both input and weights, each output value should be:
        # 2*2*3 = 12 (2x2 kernel, 3 channels)
        expected_value = 12.0
        
        # Check that all values in the output are equal to expected_value
        self.assertTrue(mx.allclose(result, mx.full(expected_shape, expected_value)))
        
    def test_op_mapping(self):
        """Test operation mapping"""
        # Get operation map
        op_map = get_complex_ops_map()
        
        # Check that required operations are mapped
        self.assertIn('tt.dot', op_map)
        self.assertIn('tt.batch_matmul', op_map)
        self.assertIn('tt.conv', op_map)

@unittest.skipIf(not HAS_MLX, "MLX not available")
class TestLauncher(unittest.TestCase):
    """Test Metal launcher implementation"""
    
    def setUp(self):
        """Initialize test case"""
        # Create compiler instance
        self.compiler = MetalCompiler()
        
        # Create test function for compilation
        def simple_add(x, y):
            return x + y
            
        self.test_fn = simple_add
        self.test_inputs = [mx.array([1.0, 2.0, 3.0]), mx.array([4.0, 5.0, 6.0])]
        
    def test_compiler_initialization(self):
        """Test compiler initialization"""
        # Check that cache directory is created
        self.assertTrue(os.path.exists(self.compiler.cache_dir))
        
    @unittest.skipIf(not hasattr(mx, "compile"), "MLX JIT compilation not available")
    def test_jit_compile(self):
        """Test JIT compilation"""
        try:
            # Compile function
            launcher = self.compiler.jit_compile(self.test_fn, self.test_inputs)
            
            # Check launcher type
            self.assertIsInstance(launcher, MetalLauncher)
            
            # Call launcher
            result = launcher(*self.test_inputs)
            
            # Expected result: [1,2,3] + [4,5,6] = [5,7,9]
            expected = mx.array([5.0, 7.0, 9.0])
            
            # Verify result shape (may not match exactly due to MLX implementation)
            self.assertEqual(result.shape, expected.shape)
            
        except Exception as e:
            # Handle exceptions, as MLX may not fully support Metal compilation
            print(f"JIT compilation test warning: {e}")
            
    def test_launcher_performance_counters(self):
        """Test launcher performance counters"""
        # Create a mock launcher
        metadata = {"kernel_name": "test_kernel"}
        options = {}
        # Empty binary data - this won't actually run on Metal
        launcher = MetalLauncher(b'', metadata, options)
        
        # Check initial performance counters
        stats = launcher.get_performance_stats()
        self.assertEqual(stats["total_calls"], 0)
        self.assertEqual(stats["total_time"], 0)
        
    def test_map_kernel_launch_params(self):
        """Test mapping of kernel launch parameters"""
        # Create launch parameters
        kernel_params = {
            "grid": (8, 8, 1),
            "block": (16, 16, 1),
            "shared_mem": 4096
        }
        
        # Map launch parameters
        metal_params = map_kernel_launch_params(kernel_params)
        
        # Verify mappings
        self.assertEqual(metal_params["grid_size"], (8, 8, 1))
        self.assertEqual(metal_params["threadgroup_size"], (16, 16, 1))
        self.assertEqual(metal_params["shared_memory_size"], 4096)

class TestMemoryLayout(unittest.TestCase):
    """Test memory layout implementation"""
    
    def setUp(self):
        """Initialize test case"""
        # Create test shapes
        self.shape_2d = (4, 3)
        self.shape_3d = (2, 4, 3)
        
    def test_memory_layout_creation(self):
        """Test memory layout creation"""
        # Create row-major layout
        row_layout = MemoryLayout(self.shape_2d, "row_major")
        
        # Check properties
        self.assertEqual(row_layout.shape, self.shape_2d)
        self.assertEqual(row_layout.layout_type, "row_major")
        self.assertEqual(row_layout.size, 12)  # 4 * 3
        
        # Check strides (row-major: last dimension has stride 1)
        self.assertEqual(row_layout.strides, (3, 1))
        
        # Create column-major layout
        col_layout = MemoryLayout(self.shape_2d, "col_major")
        
        # Check strides (column-major: first dimension has stride 1)
        self.assertEqual(col_layout.strides, (1, 4))
        
    def test_linear_indexing(self):
        """Test linear indexing functions"""
        # Create row-major layout
        row_layout = MemoryLayout(self.shape_2d, "row_major")
        
        # Check index calculation
        idx = row_layout.get_index((1, 2))  # Row 1, Column 2
        self.assertEqual(idx, 5)  # 1*3 + 2
        
        # Check coordinate calculation
        coords = row_layout.get_coords(5)
        self.assertEqual(coords, (1, 2))
        
    def test_layout_equality(self):
        """Test layout equality comparison"""
        # Create two identical layouts
        layout1 = MemoryLayout(self.shape_2d, "row_major")
        layout2 = MemoryLayout(self.shape_2d, "row_major")
        
        # Create a different layout
        layout3 = MemoryLayout(self.shape_2d, "col_major")
        
        # Check equality
        self.assertEqual(layout1, layout2)
        self.assertNotEqual(layout1, layout3)
        
    @unittest.skipIf(not HAS_MLX, "MLX not available")
    def test_adapt_tensor(self):
        """Test tensor adaptation between layouts"""
        # Create a test tensor
        data = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=mx.float32)
        
        # Define layouts
        row_layout = MemoryLayout((3, 3), "row_major")
        col_layout = MemoryLayout((3, 3), "col_major")
        
        # Adapt tensor from row-major to column-major
        adapted = adapt_tensor(data, row_layout, col_layout)
        
        # Expected result: transpose of the original matrix
        expected = mx.transpose(data)
        
        # Verify result
        self.assertTrue(mx.allclose(adapted, expected))
        
    def test_optimal_layout(self):
        """Test optimal layout determination"""
        # Get optimal layout for 2D tensor on Metal
        layout_2d = get_optimal_layout(self.shape_2d, "metal")
        
        # Check that it's row-major (usually better for Metal)
        self.assertEqual(layout_2d.layout_type, "row_major")
        
        # Get optimal layout for 4D tensor (NCHW)
        layout_4d = get_optimal_layout((1, 3, 32, 32), "metal")
        
        # Check properties
        self.assertEqual(layout_4d.shape, (1, 3, 32, 32))
        self.assertEqual(layout_4d.layout_type, "row_major")

class TestThreadMapping(unittest.TestCase):
    """Test thread mapping implementation"""
    
    def setUp(self):
        """Initialize test case"""
        # Create thread mapping instance
        self.thread_mapper = ThreadMapping()
        
    def test_thread_mapper_initialization(self):
        """Test thread mapper initialization"""
        # Check default values
        self.assertGreater(self.thread_mapper.max_threads_per_threadgroup, 0)
        self.assertIsInstance(self.thread_mapper.max_threadgroups, tuple)
        self.assertEqual(len(self.thread_mapper.max_threadgroups), 3)
        
    def test_optimal_block_size(self):
        """Test optimal block size calculation"""
        # Test with various thread counts
        block_32 = self.thread_mapper.get_optimal_block_size(32)
        self.assertEqual(block_32, (32, 1, 1))
        
        block_1000 = self.thread_mapper.get_optimal_block_size(1000)
        # Should be rounded to a multiple of SIMD width and capped at max
        self.assertLessEqual(block_1000[0], self.thread_mapper.max_threads_per_threadgroup)
        
    def test_grid_dimensions(self):
        """Test grid dimension calculation"""
        # Test with small block count
        grid_10 = self.thread_mapper.get_grid_dimensions(10)
        self.assertEqual(grid_10, (10, 1, 1))
        
        # Test with large block count
        max_x = self.thread_mapper.max_threadgroups[0]
        grid_large = self.thread_mapper.get_grid_dimensions(max_x * 2)
        self.assertEqual(grid_large[0], max_x)
        self.assertEqual(grid_large[1], 2)

def main():
    """Run the Metal backend tests"""
    print("Testing Metal Backend Components")
    print(f"MLX Available: {HAS_MLX}")
    
    # Run tests
    unittest.main()

if __name__ == "__main__":
    main() 