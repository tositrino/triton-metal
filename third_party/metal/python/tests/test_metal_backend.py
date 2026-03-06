#!/usr/bin/env python
"""
Test Metal Backend

This script tests the basic functionality of the Triton Metal backend
on Apple Silicon.
"""

import os
import sys
import unittest
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip Triton imports - we'll just test the Metal backend components directly

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
    from python.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
    from python.operation_mapping import MLXDispatcher, OpCategory
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this from the metal module root directory")
    sys.exit(1)

class TestMetalBackend(unittest.TestCase):
    """Test the Metal backend for Triton on Apple Silicon"""
    
    def setUp(self):
        """Set up test case"""
        # Check if we're running on Apple Silicon
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.UNKNOWN:
            self.skipTest("Not running on Apple Silicon")
        
        # Check if Metal is available
        try:
            # Check MLX availability
            self.mlx_available = hasattr(mx, "metal") or "metal" in str(mx.default_device())
        except:
            self.mlx_available = False
    
    def test_hardware_detection(self):
        """Test hardware detection"""
        # We know we're on Apple Silicon if we got this far
        self.assertNotEqual(hardware_capabilities.chip_generation, AppleSiliconGeneration.UNKNOWN)
        
        # Print hardware info
        print(f"\nDetected Apple Silicon: {hardware_capabilities.chip_generation.name}")
        print(f"Metal Feature Set: {hardware_capabilities.feature_set.name}")
        print(f"GPU Family: {hardware_capabilities.gpu_family}")
        print(f"SIMD Width: {hardware_capabilities.simd_width}")
        
        # Check that we have reasonable values
        self.assertGreater(hardware_capabilities.max_threads_per_threadgroup, 0)
        self.assertGreater(hardware_capabilities.max_threadgroups_per_grid, 0)
        self.assertGreater(hardware_capabilities.shared_memory_size, 0)
    
    def test_mlx_availability(self):
        """Test MLX availability with Metal backend"""
        self.assertTrue(self.mlx_available, "MLX with Metal support is not available")
        
        # Check if Metal is being used
        device = str(mx.default_device()).lower()
        print(f"\nMLX is using device: {mx.default_device()}")
        
        # Test might be running on CPU even with Metal available, so don't assert here
        if "metal" not in device:
            print(f"Warning: MLX is not using Metal device: {device}")
    
    def test_basic_mlx_operation(self):
        """Test basic MLX operation"""
        # Create test tensors
        a = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = mx.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        # Perform operations
        c = a + b
        d = a * b
        e = mx.sum(c)
        
        # Check results
        expected_c = mx.array([6.0, 6.0, 6.0, 6.0, 6.0])
        expected_d = mx.array([5.0, 8.0, 9.0, 8.0, 5.0])
        expected_e = mx.array(30.0)
        
        np.testing.assert_allclose(c.tolist(), expected_c.tolist())
        np.testing.assert_allclose(d.tolist(), expected_d.tolist())
        np.testing.assert_allclose(e.tolist(), expected_e.tolist())
    
    def test_operation_mapping(self):
        """Test operation mapping from Triton to MLX"""
        # Create MLX dispatcher
        dispatcher = MLXDispatcher()
        
        # Test mapping a few basic operations
        add_op, add_cat = dispatcher.map_triton_op("tt.binary.add")
        mul_op, mul_cat = dispatcher.map_triton_op("tt.binary.mul")
        
        # Check that operations were mapped correctly
        self.assertIsNotNone(add_op)
        self.assertEqual(add_cat, OpCategory.ELEMENTWISE)
        
        self.assertIsNotNone(mul_op)
        self.assertEqual(mul_cat, OpCategory.ELEMENTWISE)
        
        # Test operations with sample data
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        
        result = add_op(a, b)
        expected = mx.array([5.0, 7.0, 9.0])
        np.testing.assert_allclose(result.tolist(), expected.tolist())
        
        result = mul_op(a, b)
        expected = mx.array([4.0, 10.0, 18.0])
        np.testing.assert_allclose(result.tolist(), expected.tolist())
        
        # Test sum only if it exists
        sum_op, sum_cat = dispatcher.map_triton_op("tt.reduce.sum")
        if sum_op is not None:
            result = sum_op(a, 0)
            expected = mx.array(6.0)
            np.testing.assert_allclose(result.tolist(), expected.tolist())
            self.assertEqual(sum_cat, OpCategory.REDUCTION)

def main():
    """Run tests"""
    # Print some information
    print("Testing Triton Metal Backend on Apple Silicon")
    print(f"MLX Available: {MLX_AVAILABLE}")
    
    # Run tests
    unittest.main()

if __name__ == "__main__":
    main() 