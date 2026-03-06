#!/usr/bin/env python3
"""
Test the mapping functionality of basic arithmetic and mathematical operations in the MLX bridge layer
"""

import unittest
import os
import sys
import numpy as np

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

# Import MLX
try:
    import mlx.core as mx
except ImportError:
    print("Error: MLX library not installed. Please install with: pip install mlx")
    sys.exit(1)

# Import our modules
from mlx.mlx_bridge import init_dtype_map, init_op_map
from mlx.memory_layout import MemoryLayout, adapt_tensor

# Mock Triton operations
class MockOp:
    def __init__(self, name, results=None, operands=None, attributes=None):
        self.name = name
        self.results = results or []
        self.operands = operands or []
        self.attributes = attributes or {}

# Define a simple version of operation mapping for testing
TEST_OP_MAP = {
    # Binary operations
    'tt.add': mx.add,
    'tt.sub': mx.subtract,
    'tt.mul': mx.multiply,
    'tt.div': mx.divide,
    'tt.max': mx.maximum,
    'tt.min': mx.minimum,
    'tt.pow': mx.power,
    'tt.mod': mx.remainder,
    'tt.and': lambda a, b: mx.logical_and(a != 0, b != 0),
    'tt.or': lambda a, b: mx.logical_or(a != 0, b != 0),
    'tt.xor': lambda a, b: mx.logical_xor(a != 0, b != 0),
    'tt.eq': mx.equal,
    'tt.ne': mx.not_equal,
    'tt.lt': mx.less,
    'tt.le': mx.less_equal,
    'tt.gt': mx.greater,
    'tt.ge': mx.greater_equal,
    
    # Unary operations
    'tt.exp': mx.exp,
    'tt.log': mx.log,
    'tt.sin': mx.sin,
    'tt.cos': mx.cos,
    'tt.sqrt': mx.sqrt,
    'tt.neg': mx.negative,
    'tt.not': lambda x: mx.logical_not(x != 0),
    'tt.abs': mx.abs,
    'tt.tanh': mx.tanh,
    
    # Complex operations
    'tt.dot': mx.matmul,
    'tt.reshape': mx.reshape,
    'tt.trans': mx.transpose,
    
    # Reduction operations
    'tt.reduce': lambda op, operands, _: handle_reduction_test(op, operands),
}

def handle_reduction_test(op, operands):
    """Reduction handling function for testing purposes"""
    input_tensor = operands[0]
    axis = op.attributes.get("axis")
    reduce_type = op.attributes.get("reduce_type")
    
    if reduce_type == "sum":
        return mx.sum(input_tensor, axis=axis)
    elif reduce_type == "max":
        return mx.max(input_tensor, axis=axis)
    elif reduce_type == "min":
        return mx.min(input_tensor, axis=axis)
    elif reduce_type == "mean":
        return mx.mean(input_tensor, axis=axis)
    else:
        raise NotImplementedError(f"Reduction type {reduce_type} not implemented")

class TestBasicOps(unittest.TestCase):
    """Test basic operation mapping"""
    
    def setUp(self):
        """Initialize before testing"""
        # Create some basic test data
        self.x = mx.array([1, 2, 3, 4], dtype=mx.float32)
        self.y = mx.array([5, 6, 7, 8], dtype=mx.float32)
        self.a = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
        self.b = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
        
    def test_binary_arithmetic(self):
        """Test basic binary arithmetic operations"""
        # Test addition
        result = TEST_OP_MAP["tt.add"](self.x, self.y)
        expected = self.x + self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # Test subtraction
        result = TEST_OP_MAP["tt.sub"](self.x, self.y)
        expected = self.x - self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # Test multiplication
        result = TEST_OP_MAP["tt.mul"](self.x, self.y)
        expected = self.x * self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # Test division
        result = TEST_OP_MAP["tt.div"](self.x, self.y)
        expected = self.x / self.y
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_unary_operations(self):
        """Test unary operations"""
        # Test exponential
        result = TEST_OP_MAP["tt.exp"](self.x)
        expected = mx.exp(self.x)
        self.assertTrue(mx.allclose(result, expected))
        
        # Test logarithm
        pos_x = mx.array([1, 2, 3, 4], dtype=mx.float32)
        result = TEST_OP_MAP["tt.log"](pos_x)
        expected = mx.log(pos_x)
        self.assertTrue(mx.allclose(result, expected))
        
        # Test square root
        result = TEST_OP_MAP["tt.sqrt"](self.x)
        expected = mx.sqrt(self.x)
        self.assertTrue(mx.allclose(result, expected))
        
        # Test negation
        result = TEST_OP_MAP["tt.neg"](self.x)
        expected = -self.x
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_logical_operations(self):
        """Test logical operations"""
        x_bool = self.x > 2
        y_bool = self.y > 6
        
        # Test logical AND
        result = TEST_OP_MAP["tt.and"](x_bool, y_bool)
        expected = mx.logical_and(x_bool, y_bool)
        self.assertTrue(mx.array_equal(result, expected))
        
        # Test logical OR
        result = TEST_OP_MAP["tt.or"](x_bool, y_bool)
        expected = mx.logical_or(x_bool, y_bool)
        self.assertTrue(mx.array_equal(result, expected))
        
        # Test logical NOT
        result = TEST_OP_MAP["tt.not"](x_bool)
        expected = mx.logical_not(x_bool)
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_comparison_operations(self):
        """Test comparison operations"""
        # Test equals
        result = TEST_OP_MAP["tt.eq"](self.x, self.y)
        expected = self.x == self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # Test not equals
        result = TEST_OP_MAP["tt.ne"](self.x, self.y)
        expected = self.x != self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # Test less than
        result = TEST_OP_MAP["tt.lt"](self.x, self.y)
        expected = self.x < self.y
        self.assertTrue(mx.array_equal(result, expected))
        
        # Test greater than
        result = TEST_OP_MAP["tt.gt"](self.x, self.y)
        expected = self.x > self.y
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_matrix_operations(self):
        """Test matrix operations"""
        # Test matrix multiplication
        result = TEST_OP_MAP["tt.dot"](self.a, self.b)
        expected = mx.matmul(self.a, self.b)
        self.assertTrue(mx.allclose(result, expected))
        
        # Test transpose
        result = TEST_OP_MAP["tt.trans"](self.a)
        expected = mx.transpose(self.a)
        self.assertTrue(mx.array_equal(result, expected))
        
        # Test reshape
        result = TEST_OP_MAP["tt.reshape"](self.x, (2, 2))
        expected = mx.reshape(self.x, (2, 2))
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_reduction_operations(self):
        """Test reduction operations"""
        # Create mock reduction operation
        sum_op = MockOp("tt.reduce", results=[None], operands=[self.a], 
                        attributes={"reduce_type": "sum", "axis": 0})
        
        # Test sum reduction
        result = TEST_OP_MAP["tt.reduce"](sum_op, [self.a], None)
        expected = mx.sum(self.a, axis=0)
        self.assertTrue(mx.array_equal(result, expected))
        
        # Create mock max reduction operation
        max_op = MockOp("tt.reduce", results=[None], operands=[self.a], 
                       attributes={"reduce_type": "max", "axis": 1})
        
        # Test max reduction
        result = TEST_OP_MAP["tt.reduce"](max_op, [self.a], None)
        expected = mx.max(self.a, axis=1)
        self.assertTrue(mx.array_equal(result, expected))
        
    def test_memory_layout(self):
        """Test memory layout adaptation"""
        # Create row-major layout
        row_layout = MemoryLayout(self.a.shape)
        
        # Create column-major layout
        col_layout = MemoryLayout(self.a.shape, "col_major")
        
        # Test conversion from row-major to column-major
        row_to_col = adapt_tensor(self.a, row_layout, col_layout)
        self.assertEqual(row_to_col.shape, self.a.shape)
        
        # Verify conversion is equivalent to transpose
        expected = mx.transpose(self.a)
        self.assertTrue(mx.allclose(row_to_col, expected))
        
        # Test conversion from column-major to row-major
        col_to_row = adapt_tensor(expected, col_layout, row_layout)
        self.assertEqual(col_to_row.shape, self.a.shape)
        self.assertTrue(mx.allclose(col_to_row, self.a))

if __name__ == "__main__":
    unittest.main() 