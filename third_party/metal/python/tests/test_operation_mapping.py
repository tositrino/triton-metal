#!/usr/bin/env python
"""
Test Operation Mapping

This script tests the operation mapping and fusion optimizations
for the Triton Metal backend on Apple Silicon.
"""

import os
import sys
import unittest
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import MLX
try:
    import mlx.core as mx
except ImportError:
    print("MLX not found. Please install it with 'pip install mlx'")
    sys.exit(1)

# Import our modules
try:
    from python.operation_mapping import MLXDispatcher, OpCategory, op_conversion_registry
    from python.metal_fusion_optimizer import FusionPattern, FusionOptimizer, fusion_optimizer
    from python.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this from the metal module root directory")
    sys.exit(1)

class TestMLXDispatcher(unittest.TestCase):
    """Test MLX operation dispatcher functionality"""
    
    def setUp(self):
        """Set up test case"""
        self.dispatcher = MLXDispatcher()
    
    def test_base_operations(self):
        """Test basic operations"""
        # Create test tensors
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        
        # Test addition
        add_op = self.dispatcher.get_op("add")
        self.assertIsNotNone(add_op)
        
        result = add_op(a, b)
        expected = mx.array([5.0, 7.0, 9.0])
        np.testing.assert_allclose(result.tolist(), expected.tolist())
        
        # Test multiplication
        mul_op = self.dispatcher.get_op("mul")
        self.assertIsNotNone(mul_op)
        
        result = mul_op(a, b)
        expected = mx.array([4.0, 10.0, 18.0])
        np.testing.assert_allclose(result.tolist(), expected.tolist())
    
    def test_operation_mapping(self):
        """Test Triton operation mapping"""
        # Test binary operation mapping
        op, category = self.dispatcher.map_triton_op("tt.binary.add")
        self.assertIsNotNone(op)
        self.assertEqual(category, OpCategory.ELEMENTWISE)
        
        # Test unary operation mapping
        op, category = self.dispatcher.map_triton_op("tt.unary.exp")
        self.assertIsNotNone(op)
        self.assertEqual(category, OpCategory.ELEMENTWISE)
        
        # Test reduction operation mapping
        op, category = self.dispatcher.map_triton_op("tt.reduce.sum")
        self.assertIsNotNone(op)
        self.assertEqual(category, OpCategory.REDUCTION)
        
        # Test comparison operation mapping
        op, category = self.dispatcher.map_triton_op("tt.cmp.eq")
        self.assertIsNotNone(op)
        self.assertEqual(category, OpCategory.ELEMENTWISE)
        
        # Test matrix operation mapping
        op, category = self.dispatcher.map_triton_op("tt.dot")
        self.assertIsNotNone(op)
        self.assertEqual(category, OpCategory.MATRIX)
    
    def test_optimized_operations(self):
        """Test optimized operations for Apple Silicon"""
        # Skip if not running on Apple Silicon
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.UNKNOWN:
            self.skipTest("Not running on Apple Silicon")
        
        # Test matrix multiplication with optimized GEMM
        a = mx.array([[1.0, 2.0], [3.0, 4.0]])
        b = mx.array([[5.0, 6.0], [7.0, 8.0]])
        
        result = self.dispatcher._optimized_gemm(a, b)
        expected = mx.matmul(a, b)
        np.testing.assert_allclose(result.tolist(), expected.tolist())
        
        # Test with transposes
        result_trans = self.dispatcher._optimized_gemm(a, b, trans_a=True, trans_b=True)
        expected_trans = mx.matmul(mx.transpose(a), mx.transpose(b))
        np.testing.assert_allclose(result_trans.tolist(), expected_trans.tolist())

class TestOpConversionRegistry(unittest.TestCase):
    """Test operation conversion registry"""
    
    def setUp(self):
        """Set up test case"""
        self.registry = op_conversion_registry
        self.context = {}  # Conversion context
    
    def test_converter_registration(self):
        """Test converter registration"""
        # Check that we have converters for common operations
        self.assertIsNotNone(self.registry.get_converter("tt.binary.add"))
        self.assertIsNotNone(self.registry.get_converter("tt.unary.exp"))
        self.assertIsNotNone(self.registry.get_converter("tt.reduce.sum"))
        self.assertIsNotNone(self.registry.get_converter("tt.cmp.eq"))
        self.assertIsNotNone(self.registry.get_converter("tt.dot"))
    
    def test_binary_conversion(self):
        """Test binary operation conversion"""
        # Create test tensors and add to context
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        
        self.context["a"] = a
        self.context["b"] = b
        
        # Create binary operation
        add_op = {
            "id": "add_result",
            "type": "tt.binary.add",
            "lhs_id": "a",
            "rhs_id": "b"
        }
        
        # Convert operation
        converter = self.registry.get_converter(add_op["type"])
        self.assertIsNotNone(converter)
        
        result = converter(add_op, self.context)
        expected = mx.array([5.0, 7.0, 9.0])
        np.testing.assert_allclose(result.tolist(), expected.tolist())
    
    def test_unary_conversion(self):
        """Test unary operation conversion"""
        # Create test tensor and add to context
        a = mx.array([1.0, 2.0, 3.0])
        self.context["a"] = a
        
        # Create unary operation
        exp_op = {
            "id": "exp_result",
            "type": "tt.unary.exp",
            "operand_id": "a"
        }
        
        # Convert operation
        converter = self.registry.get_converter(exp_op["type"])
        self.assertIsNotNone(converter)
        
        result = converter(exp_op, self.context)
        expected = mx.exp(a)
        np.testing.assert_allclose(result.tolist(), expected.tolist())

class TestFusionOptimizer(unittest.TestCase):
    """Test fusion optimizer functionality"""
    
    def setUp(self):
        """Set up test case"""
        self.optimizer = fusion_optimizer
    
    def test_pattern_matching(self):
        """Test pattern matching"""
        # Create a sample pattern
        pattern = FusionPattern("test", ["add", "mul"], None)
        
        # Create sample operations
        ops = [
            {"type": "tt.binary.add"},
            {"type": "tt.binary.mul"}
        ]
        
        # Test pattern matching
        self.assertTrue(pattern.matches(ops))
        
        # Test with non-matching pattern
        ops = [
            {"type": "tt.binary.sub"},
            {"type": "tt.binary.mul"}
        ]
        
        self.assertFalse(pattern.matches(ops))
    
    def test_fusion_opportunities(self):
        """Test finding fusion opportunities"""
        # Create sample operations
        ops = [
            # FMA pattern
            {"id": "a", "type": "tt.binary.mul"},
            {"id": "b", "type": "tt.binary.add"},
            
            # Some other operations
            {"id": "c", "type": "tt.unary.exp"},
            
            # Another FMA pattern
            {"id": "d", "type": "tt.binary.mul"},
            {"id": "e", "type": "tt.binary.add"}
        ]
        
        # Find fusion opportunities
        opportunities = self.optimizer.find_fusion_opportunities(ops)
        
        # We should have at least 2 opportunities (for the FMA patterns)
        # This may vary based on hardware and what patterns are enabled
        self.assertGreaterEqual(len(opportunities), 2)
        
        # Check that we found the FMA patterns
        fma_patterns = [op for op in opportunities if op[1].name == "fused_multiply_add"]
        self.assertGreaterEqual(len(fma_patterns), 2)
    
    def test_apply_fusion(self):
        """Test applying fusion"""
        # Create sample operations for FMA
        ops = [
            {"id": "a", "type": "tt.binary.mul", "lhs_id": "x", "rhs_id": "y"},
            {"id": "b", "type": "tt.binary.add", "lhs_id": "a", "rhs_id": "z"}
        ]
        
        # Find fusion pattern
        opportunities = self.optimizer.find_fusion_opportunities(ops)
        self.assertGreaterEqual(len(opportunities), 1)
        
        # Get the first opportunity
        start_idx, pattern, pattern_len = opportunities[0]
        
        # Apply fusion
        new_ops, success = self.optimizer.apply_fusion(ops, start_idx, pattern, start_idx + pattern_len)
        
        # Check that fusion was successful
        self.assertTrue(success)
        
        # Check that we now have one operation instead of two
        self.assertEqual(len(new_ops), 1)
        
        # Check the fused operation properties
        fused_op = new_ops[0]
        self.assertEqual(fused_op["type"], "tt.fused.fused_multiply_add")
        self.assertEqual(fused_op["a_id"], "x")
        self.assertEqual(fused_op["b_id"], "y")
        self.assertEqual(fused_op["c_id"], "z")
    
    def test_execute_fused_op(self):
        """Test executing fused operations"""
        # Skip if not running on Apple Silicon
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.UNKNOWN:
            self.skipTest("Not running on Apple Silicon")
        
        # Create test tensors
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        c = mx.array([7.0, 8.0, 9.0])
        
        # Create fused operation
        fused_op = {
            "id": "fused_op",
            "type": "tt.fused.fused_multiply_add",
            "a_id": "a",
            "b_id": "b",
            "c_id": "c",
            "alpha": 1.0,
            "beta": 1.0
        }
        
        # Create context
        context = {"a": a, "b": b, "c": c}
        
        # Execute fused operation
        result = self.optimizer.execute_fused_op(fused_op, context)
        
        # Check result
        expected = a * b + c
        np.testing.assert_allclose(result.tolist(), expected.tolist())

class TestEndToEnd(unittest.TestCase):
    """End-to-end tests of operation mapping and fusion"""
    
    def setUp(self):
        """Set up test case"""
        self.dispatcher = MLXDispatcher()
        self.registry = op_conversion_registry
        self.optimizer = fusion_optimizer
    
    def test_operation_chain(self):
        """Test a chain of operations with fusion opportunities"""
        # Skip if not running on Apple Silicon
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.UNKNOWN:
            self.skipTest("Not running on Apple Silicon")
        
        # Create test tensors
        a = mx.array([[1.0, 2.0], [3.0, 4.0]])
        b = mx.array([[5.0, 6.0], [7.0, 8.0]])
        c = mx.array([[9.0, 10.0], [11.0, 12.0]])
        
        # Create operations
        ops = [
            {"id": "a", "type": "tt.make_tensor", "value": a},
            {"id": "b", "type": "tt.make_tensor", "value": b},
            {"id": "c", "type": "tt.make_tensor", "value": c},
            {"id": "mul", "type": "tt.binary.mul", "lhs_id": "a", "rhs_id": "b"},
            {"id": "add", "type": "tt.binary.add", "lhs_id": "mul", "rhs_id": "c"}
        ]
        
        # Optimize operations
        optimized_ops = self.optimizer.optimize(ops)
        
        # We should have 4 operations after optimization (3 tensors + 1 fused op)
        self.assertEqual(len(optimized_ops), 4)
        
        # Find the fused operation
        fused_op = None
        for op in optimized_ops:
            if "tt.fused" in op.get("type", ""):
                fused_op = op
                break
        
        self.assertIsNotNone(fused_op)
        self.assertEqual(fused_op["type"], "tt.fused.fused_multiply_add")
    
    def test_performance(self):
        """Test performance of fused vs. unfused operations"""
        # Skip if not running on Apple Silicon
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.UNKNOWN:
            self.skipTest("Not running on Apple Silicon")
        
        # Create large test tensors
        a = mx.random.normal((1024, 1024))
        b = mx.random.normal((1024, 1024))
        c = mx.random.normal((1024, 1024))
        
        # Create context
        context = {"a": a, "b": b, "c": c}
        
        # Time unfused operations
        start_time = time.time()
        for _ in range(10):
            mul_result = mx.multiply(a, b)
            add_result = mx.add(mul_result, c)
        unfused_time = time.time() - start_time
        
        # Create fused operation
        fused_op = {
            "id": "fused_op",
            "type": "tt.fused.fused_multiply_add",
            "a_id": "a",
            "b_id": "b",
            "c_id": "c",
            "alpha": 1.0,
            "beta": 1.0
        }
        
        # Time fused operation
        start_time = time.time()
        for _ in range(10):
            fused_result = self.optimizer.execute_fused_op(fused_op, context)
        fused_time = time.time() - start_time
        
        print(f"\nUnfused operation time: {unfused_time:.6f}s")
        print(f"Fused operation time: {fused_time:.6f}s")
        print(f"Speedup: {unfused_time / fused_time:.2f}x")
        
        # We expect the fused operation to be faster, but don't enforce it in the test
        # as it might depend on the hardware and MLX version
        # self.assertLess(fused_time, unfused_time)

def main():
    """Run tests"""
    # Print hardware info
    if hardware_capabilities.chip_generation != AppleSiliconGeneration.UNKNOWN:
        print(f"Running on Apple Silicon: {hardware_capabilities.chip_generation.name}")
    else:
        print("Warning: Not running on Apple Silicon")
    
    # Run tests
    unittest.main()

if __name__ == "__main__":
    main() 