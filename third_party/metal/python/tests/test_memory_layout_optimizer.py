"""
Tests for Memory Layout Optimizer for Triton Metal Backend
"""

import os
import sys
import unittest
import json
from enum import Enum
from typing import Dict, List, Any

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the module to test
from MLX.memory_layout_optimizer import (
    MemoryLayoutOptimizer, 
    LayoutOptimizationLevel,
    MatrixLayoutPattern,
    ConvolutionLayoutPattern,
    ReductionLayoutPattern,
    optimize_memory_layout,
    get_metal_layout_optimizer,
    TensorType,
    MemoryLayout,
    MemoryAccessPattern
)

# Mock hardware capabilities if needed
try:
    from MLX.metal_hardware_optimizer import AppleSiliconGeneration
except ImportError:
    # Create mock enum
    class DummyEnum(Enum):
        UNKNOWN = 0
        M1 = 1
        M2 = 2
        M3 = 3
    
    AppleSiliconGeneration = DummyEnum

# No need to redefine enums since we imported them directly from memory_layout_optimizer

class TestMemoryLayoutPatterns(unittest.TestCase):
    """Test the memory layout pattern classes"""
    
    def test_matrix_layout_pattern(self):
        """Test matrix layout pattern"""
        # Create pattern
        pattern = MatrixLayoutPattern()
        
        # Check applicability
        matmul_op = {"type": "tt.matmul", "input_shapes": [[128, 256], [256, 512]]}
        self.assertTrue(pattern.is_applicable(matmul_op, AppleSiliconGeneration.M3))
        
        # Test with non-matrix op
        conv_op = {"type": "tt.conv2d", "input_shapes": [[1, 3, 224, 224], [64, 3, 3, 3]]}
        self.assertFalse(pattern.is_applicable(conv_op, AppleSiliconGeneration.M3))
        
        # Test optimal layout
        layout = pattern.get_optimal_layout([128, 256], AppleSiliconGeneration.M3)
        self.assertEqual(layout, MemoryLayout.BLOCK_BASED)
        
        # Test parameters
        params = pattern.get_parameters([128, 256], AppleSiliconGeneration.M3)
        self.assertEqual(params["vector_width"], 8)
        self.assertEqual(params["block_size"], 128)
        self.assertTrue(params["use_tensor_cores"])
        
        # Test with smaller matrix
        layout = pattern.get_optimal_layout([32, 64], AppleSiliconGeneration.M3)
        self.assertEqual(layout, MemoryLayout.ROW_MAJOR)
        
        params = pattern.get_parameters([32, 64], AppleSiliconGeneration.M3)
        self.assertEqual(params["block_size"], 64)
    
    def test_convolution_layout_pattern(self):
        """Test convolution layout pattern"""
        # Create pattern
        pattern = ConvolutionLayoutPattern()
        
        # Check applicability
        conv_op = {"type": "tt.conv2d", "input_shapes": [[1, 3, 224, 224], [64, 3, 3, 3]]}
        self.assertTrue(pattern.is_applicable(conv_op, AppleSiliconGeneration.M3))
        
        # Test with non-conv op
        matmul_op = {"type": "tt.matmul", "input_shapes": [[128, 256], [256, 512]]}
        self.assertFalse(pattern.is_applicable(matmul_op, AppleSiliconGeneration.M3))
        
        # Test optimal layout
        layout = pattern.get_optimal_layout([1, 3, 224, 224], AppleSiliconGeneration.M3)
        self.assertEqual(layout, MemoryLayout.TEXTURE_OPTIMIZED)
        
        # Test parameters
        params = pattern.get_parameters([1, 3, 224, 224], AppleSiliconGeneration.M3)
        self.assertEqual(params["vector_width"], 8)
        self.assertEqual(params["tile_h"], 64)
        self.assertEqual(params["tile_w"], 64)
        self.assertTrue(params["use_tensor_cores"])
        self.assertTrue(params["use_dynamic_caching"])
    
    def test_reduction_layout_pattern(self):
        """Test reduction layout pattern"""
        # Create pattern
        pattern = ReductionLayoutPattern()
        
        # Check applicability
        reduce_op = {"type": "tt.reduce", "input_shapes": [[1024, 1024]], "args": {"axis": 1}}
        self.assertTrue(pattern.is_applicable(reduce_op, AppleSiliconGeneration.M3))
        
        # Test with non-reduction op
        matmul_op = {"type": "tt.matmul", "input_shapes": [[128, 256], [256, 512]]}
        self.assertFalse(pattern.is_applicable(matmul_op, AppleSiliconGeneration.M3))
        
        # Test optimal layout
        layout = pattern.get_optimal_layout([1024], AppleSiliconGeneration.M3)
        self.assertEqual(layout, MemoryLayout.COALESCED)
        
        # Test parameters
        params = pattern.get_parameters([1024], AppleSiliconGeneration.M3)
        self.assertEqual(params["vector_width"], 8)
        self.assertEqual(params["block_size"], 1024)
        self.assertTrue(params["hierarchical_reduction"])
        self.assertTrue(params["use_simdgroup_reduction"])
        self.assertTrue(params["two_stage_reduction"])

class TestMemoryLayoutOptimizer(unittest.TestCase):
    """Test the memory layout optimizer"""
    
    def setUp(self):
        """Set up test cases"""
        # Create a sample graph
        self.sample_graph = {
            "ops": [
                {
                    "type": "tt.matmul",
                    "id": "op1",
                    "input_shapes": [[128, 256], [256, 512]]
                },
                {
                    "type": "tt.add",
                    "id": "op2",
                    "input_shapes": [[128, 512], [128, 512]]
                },
                {
                    "type": "tt.reduce",
                    "id": "op3",
                    "input_shapes": [[128, 512]],
                    "args": {"axis": 1}
                }
            ]
        }
        
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        # Create optimizer
        optimizer = MemoryLayoutOptimizer(LayoutOptimizationLevel.HARDWARE_SPECIFIC)
        
        # Check that patterns are created
        self.assertGreaterEqual(len(optimizer.patterns), 3)
        
        # Check stats initialization
        self.assertEqual(optimizer.stats["optimized_ops"], 0)
        self.assertEqual(optimizer.stats["memory_layout_changes"], 0)
    
    def test_optimize_matmul(self):
        """Test optimization of matrix multiplication"""
        # Create optimizer
        optimizer = MemoryLayoutOptimizer(LayoutOptimizationLevel.HARDWARE_SPECIFIC)
        
        # Create matmul op
        matmul_op = {
            "type": "tt.matmul",
            "id": "op1",
            "input_shapes": [[128, 256], [256, 512]]
        }
        
        # Optimize operation
        optimized_op = optimizer._optimize_operation(matmul_op)
        
        # Check that layout hints are added
        self.assertIn("layout_hints", optimized_op)
        self.assertIn("layout", optimized_op["layout_hints"])
        
        # Check that tensor type is set
        self.assertEqual(optimized_op["tensor_type"], TensorType.MATRIX.name)
        
        # Check stats
        self.assertEqual(optimizer.stats["optimized_ops"], 1)
    
    def test_optimize_convolution(self):
        """Test optimization of convolution"""
        # Create optimizer
        optimizer = MemoryLayoutOptimizer(LayoutOptimizationLevel.HARDWARE_SPECIFIC)
        
        # Create conv op
        conv_op = {
            "type": "tt.conv2d",
            "id": "op1",
            "input_shapes": [[1, 3, 224, 224], [64, 3, 3, 3]]
        }
        
        # Optimize operation
        optimized_op = optimizer._optimize_operation(conv_op)
        
        # Check that layout hints are added
        self.assertIn("layout_hints", optimized_op)
        self.assertIn("layout", optimized_op["layout_hints"])
        
        # Check tensor type
        self.assertEqual(optimized_op["tensor_type"], TensorType.CONVOLUTION_FILTER.name)
        
        # Check stats
        self.assertEqual(optimizer.stats["optimized_ops"], 1)
    
    def test_optimize_reduction(self):
        """Test optimization of reduction"""
        # Create optimizer
        optimizer = MemoryLayoutOptimizer(LayoutOptimizationLevel.HARDWARE_SPECIFIC)
        
        # Create reduce op
        reduce_op = {
            "type": "tt.reduce",
            "id": "op1",
            "input_shapes": [[1024, 1024]],
            "args": {"axis": 1}
        }
        
        # Optimize operation
        optimized_op = optimizer._optimize_operation(reduce_op)
        
        # Check that layout hints are added
        self.assertIn("layout_hints", optimized_op)
        self.assertIn("layout", optimized_op["layout_hints"])
        
        # Check tensor type
        self.assertEqual(optimized_op["tensor_type"], TensorType.VECTOR.name)
        
        # Check stats
        self.assertEqual(optimizer.stats["optimized_ops"], 1)
    
    def test_optimize_graph(self):
        """Test optimization of entire graph"""
        # Create optimizer
        optimizer = MemoryLayoutOptimizer(LayoutOptimizationLevel.HARDWARE_SPECIFIC)
        
        # Optimize graph
        optimized_graph, stats = optimizer.optimize(self.sample_graph)
        
        # Check that all operations are optimized
        self.assertEqual(stats["optimized_ops"], 3)
        
        # Check that all operations have layout hints
        for op in optimized_graph["ops"]:
            self.assertIn("layout_hints", op)
            
        # Check metadata
        self.assertIn("metadata", optimized_graph)
        self.assertTrue(optimized_graph["metadata"]["memory_layout_optimized"])
        self.assertEqual(optimized_graph["metadata"]["optimization_level"], "HARDWARE_SPECIFIC")
    
    def test_singleton_accessor(self):
        """Test singleton accessor pattern"""
        # Get optimizer instances
        optimizer1 = get_metal_layout_optimizer()
        optimizer2 = get_metal_layout_optimizer()
        
        # Check that they are the same instance
        self.assertIs(optimizer1, optimizer2)
        
        # Use the global optimize function
        optimized_graph, stats = optimize_memory_layout(self.sample_graph)
        
        # Check that optimization was applied
        self.assertGreater(stats["optimized_ops"], 0)
    
    def test_optimization_levels(self):
        """Test different optimization levels"""
        # Create a simple operation
        simple_op = {
            "type": "tt.add",
            "id": "op1",
            "input_shapes": [[128, 512], [128, 512]]
        }
        
        # Test with no optimization
        none_optimizer = MemoryLayoutOptimizer(LayoutOptimizationLevel.NONE)
        none_optimized = none_optimizer._optimize_operation(simple_op)
        
        # Check that no layout hints are added with NONE level
        self.assertNotIn("layout_hints", none_optimized)
        
        # Test with basic optimization
        basic_optimizer = MemoryLayoutOptimizer(LayoutOptimizationLevel.BASIC)
        basic_optimized = basic_optimizer._optimize_operation(simple_op)
        
        # Check that layout hints are added with BASIC level
        self.assertIn("layout_hints", basic_optimized)

if __name__ == "__main__":
    unittest.main() 