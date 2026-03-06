"""
Test module for Triton to MLX converter
"""

import unittest
import numpy as np
from typing import Dict, Any, List
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from triton_to_metal_converter import TritonToMLXConverter

@unittest.skipIf(not MLX_AVAILABLE, "MLX not available")
class TestTritonToMLXConverter(unittest.TestCase):
    """Test cases for Triton to MLX converter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.converter = TritonToMLXConverter()
    
    def test_type_conversion(self):
        """Test type conversion"""
        self.assertEqual(self.converter.convert_tensor_type("float32"), mx.float32)
        self.assertEqual(self.converter.convert_tensor_type("int32"), mx.int32)
        self.assertEqual(self.converter.convert_tensor_type("bool"), mx.bool_)
        # Test default for unknown type
        self.assertEqual(self.converter.convert_tensor_type("unknown"), mx.float32)
    
    def test_tensor_creation(self):
        """Test tensor creation"""
        # Test with shape only
        tensor1 = self.converter.create_tensor([2, 3])
        self.assertEqual(tensor1.shape, (2, 3))
        self.assertEqual(tensor1.dtype, mx.float32)
        np.testing.assert_allclose(mx.array(tensor1).astype(np.float32), np.zeros((2, 3), dtype=np.float32))
        
        # Test with shape and dtype
        tensor2 = self.converter.create_tensor([2, 3], "int32")
        self.assertEqual(tensor2.shape, (2, 3))
        self.assertEqual(tensor2.dtype, mx.int32)
        
        # Test with shape, dtype, and init value
        tensor3 = self.converter.create_tensor([2, 3], "float32", 1.0)
        self.assertEqual(tensor3.shape, (2, 3))
        self.assertEqual(tensor3.dtype, mx.float32)
        np.testing.assert_allclose(mx.array(tensor3).astype(np.float32), np.ones((2, 3), dtype=np.float32))
        
        # Test with array-like init value
        init_array = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        tensor4 = self.converter.create_tensor([2, 3], "float32", init_array)
        self.assertEqual(tensor4.shape, (2, 3))
        self.assertEqual(tensor4.dtype, mx.float32)
        np.testing.assert_allclose(mx.array(tensor4).astype(np.float32), np.array(init_array, dtype=np.float32))
    
    def test_binary_operations(self):
        """Test binary operations"""
        a = self.converter.create_tensor([2, 3], "float32", 2.0)
        b = self.converter.create_tensor([2, 3], "float32", 3.0)
        
        # Test addition
        result = self.converter.convert_binary_op("add", a, b)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.full((2, 3), 5.0, dtype=np.float32))
        
        # Test subtraction
        result = self.converter.convert_binary_op("sub", a, b)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.full((2, 3), -1.0, dtype=np.float32))
        
        # Test multiplication
        result = self.converter.convert_binary_op("mul", a, b)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.full((2, 3), 6.0, dtype=np.float32))
        
        # Test division
        result = self.converter.convert_binary_op("div", a, b)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.full((2, 3), 2/3, dtype=np.float32))
        
        # Test power
        result = self.converter.convert_binary_op("pow", a, b)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.full((2, 3), 8.0, dtype=np.float32))
        
        # Test maximum
        a = self.converter.create_tensor([2, 3], "float32", 2.0)
        b = self.converter.create_tensor([2, 3], "float32", 3.0)
        result = self.converter.convert_binary_op("max", a, b)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.full((2, 3), 3.0, dtype=np.float32))
        
        # Test minimum
        result = self.converter.convert_binary_op("min", a, b)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.full((2, 3), 2.0, dtype=np.float32))
        
        # Test logical operations
        a_bool = self.converter.create_tensor([2, 3], "bool", True)
        b_bool = self.converter.create_tensor([2, 3], "bool", False)
        
        result = self.converter.convert_binary_op("and", a_bool, b_bool)
        np.testing.assert_equal(mx.array(result).astype(bool), np.full((2, 3), False, dtype=bool))
        
        result = self.converter.convert_binary_op("or", a_bool, b_bool)
        np.testing.assert_equal(mx.array(result).astype(bool), np.full((2, 3), True, dtype=bool))
        
        # Test invalid operation
        with self.assertRaises(ValueError):
            self.converter.convert_binary_op("invalid_op", a, b)
    
    def test_unary_operations(self):
        """Test unary operations"""
        # Test exp
        a = self.converter.create_tensor([2, 3], "float32", 1.0)
        result = self.converter.convert_unary_op("exp", a)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.full((2, 3), np.exp(1.0), dtype=np.float32))
        
        # Test sqrt
        a = self.converter.create_tensor([2, 3], "float32", 4.0)
        result = self.converter.convert_unary_op("sqrt", a)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.full((2, 3), 2.0, dtype=np.float32))
        
        # Test negative
        a = self.converter.create_tensor([2, 3], "float32", 1.0)
        result = self.converter.convert_unary_op("neg", a)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.full((2, 3), -1.0, dtype=np.float32))
        
        # Test logical not
        a_bool = self.converter.create_tensor([2, 3], "bool", True)
        result = self.converter.convert_unary_op("not", a_bool)
        np.testing.assert_equal(mx.array(result).astype(bool), np.full((2, 3), False, dtype=bool))
        
        # Test invalid operation
        with self.assertRaises(ValueError):
            self.converter.convert_unary_op("invalid_op", a)
    
    def test_reduction_operations(self):
        """Test reduction operations"""
        # Create a 2x3 tensor with values [[1,2,3], [4,5,6]]
        init_array = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        a = self.converter.create_tensor([2, 3], "float32", init_array)
        
        # Test sum along axis 0
        result = self.converter.convert_reduction_op("sum", a, [0])
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.array([5.0, 7.0, 9.0], dtype=np.float32))
        
        # Test sum along axis 1
        result = self.converter.convert_reduction_op("sum", a, [1])
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.array([6.0, 15.0], dtype=np.float32))
        
        # Test max along axis 0
        result = self.converter.convert_reduction_op("max", a, [0])
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32))
        
        # Test min along axis 1
        result = self.converter.convert_reduction_op("min", a, [1])
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.array([1.0, 4.0], dtype=np.float32))
        
        # Test mean along axis 0
        result = self.converter.convert_reduction_op("mean", a, [0])
        np.testing.assert_allclose(mx.array(result).astype(np.float32), np.array([2.5, 3.5, 4.5], dtype=np.float32))
        
        # Test invalid operation
        with self.assertRaises(ValueError):
            self.converter.convert_reduction_op("invalid_op", a, [0])
    
    def test_comparison_operations(self):
        """Test comparison operations"""
        a = self.converter.create_tensor([2, 3], "float32", 2.0)
        b = self.converter.create_tensor([2, 3], "float32", 3.0)
        c = self.converter.create_tensor([2, 3], "float32", 2.0)
        
        # Test equal
        result = self.converter.convert_comparison_op("eq", a, b)
        np.testing.assert_equal(mx.array(result).astype(bool), np.full((2, 3), False, dtype=bool))
        
        result = self.converter.convert_comparison_op("eq", a, c)
        np.testing.assert_equal(mx.array(result).astype(bool), np.full((2, 3), True, dtype=bool))
        
        # Test not equal
        result = self.converter.convert_comparison_op("ne", a, b)
        np.testing.assert_equal(mx.array(result).astype(bool), np.full((2, 3), True, dtype=bool))
        
        # Test less than
        result = self.converter.convert_comparison_op("lt", a, b)
        np.testing.assert_equal(mx.array(result).astype(bool), np.full((2, 3), True, dtype=bool))
        
        # Test greater than
        result = self.converter.convert_comparison_op("gt", a, b)
        np.testing.assert_equal(mx.array(result).astype(bool), np.full((2, 3), False, dtype=bool))
        
        # Test invalid operation
        with self.assertRaises(ValueError):
            self.converter.convert_comparison_op("invalid_op", a, b)
    
    def test_matmul(self):
        """Test matrix multiplication"""
        # Create 2x3 and 3x2 matrices for multiplication
        a_init = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        b_init = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        
        a = self.converter.create_tensor([2, 3], "float32", a_init)
        b = self.converter.create_tensor([3, 2], "float32", b_init)
        
        # Test matrix multiplication
        result = self.converter.convert_matmul(a, b)
        expected = np.array([[22.0, 28.0], [49.0, 64.0]], dtype=np.float32)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), expected)
        
        # Test matrix multiplication with transpose
        result = self.converter.convert_matmul(a, b, trans_a=True, trans_b=False)
        a_trans = np.array(a_init).transpose()
        expected = np.matmul(a_trans, b_init)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), expected)
    
    def test_reshape_transpose(self):
        """Test reshape and transpose operations"""
        # Create a 2x3 tensor with values [[1,2,3], [4,5,6]]
        init_array = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        a = self.converter.create_tensor([2, 3], "float32", init_array)
        
        # Test reshape to 3x2
        result = self.converter.convert_reshape(a, [3, 2])
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), expected)
        
        # Test reshape to 6x1
        result = self.converter.convert_reshape(a, [6, 1])
        expected = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=np.float32)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), expected)
        
        # Test transpose
        result = self.converter.convert_transpose(a)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), expected)
        
        # Test transpose with explicit permutation
        result = self.converter.convert_transpose(a, [1, 0])
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), expected)
    
    def test_select(self):
        """Test select operation with MLX tensors"""
        # Create tensors for condition, true_value, and false_value
        condition = self.converter.create_tensor([2, 3], "bool", [[True, False, True], [False, True, False]])
        true_value = self.converter.create_tensor([2, 3], "float32", 1.0)
        false_value = self.converter.create_tensor([2, 3], "float32", 0.0)
        
        # Test select
        result = self.converter.convert_select(condition, true_value, false_value)
        expected = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(mx.array(result).astype(np.float32), expected)
    
    def test_convert_operations(self):
        """Test converting a list of operations"""
        # Create a list of operations
        ops = [
            {
                "id": "tensor1",
                "type": "tt.make_tensor",
                "shape": [2, 3],
                "dtype": "float32",
                "init_value": 1.0
            },
            {
                "id": "tensor2",
                "type": "tt.make_tensor",
                "shape": [2, 3],
                "dtype": "float32",
                "init_value": 2.0
            },
            {
                "id": "add_result",
                "type": "tt.binary.add",
                "lhs_id": "tensor1",
                "rhs_id": "tensor2"
            },
            {
                "id": "barrier",
                "type": "tt.debug_barrier",
                "memory_scope": "threadgroup"
            }
        ]
        
        # Convert operations
        results = self.converter.convert_operations(ops)
        
        # Check results
        self.assertIn("tensor1", results)
        self.assertIn("tensor2", results)
        self.assertIn("add_result", results)
        self.assertIn("barrier", results)
        self.assertIn("__metal_code__", results)
        
        # Check tensor values
        np.testing.assert_allclose(mx.array(results["tensor1"]).astype(np.float32), np.full((2, 3), 1.0, dtype=np.float32))
        np.testing.assert_allclose(mx.array(results["tensor2"]).astype(np.float32), np.full((2, 3), 2.0, dtype=np.float32))
        np.testing.assert_allclose(mx.array(results["add_result"]).astype(np.float32), np.full((2, 3), 3.0, dtype=np.float32))
        
        # Check Metal code
        self.assertIn("threadgroup_barrier", results["__metal_code__"])

if __name__ == "__main__":
    unittest.main() 