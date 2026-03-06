"""
Test script to verify reduction memory optimization with COALESCED layout

This script tests the correct application of the COALESCED memory layout (value 8)
to different types of reduction operations across various input shapes.
"""

import os
import sys
import unittest



# Import the metal memory manager
from metal_memory_manager import get_metal_memory_manager, MemoryLayout
from memory_layout_optimizer import (
    ReductionLayoutPattern, 
    MemoryLayoutOptimizer,
    LayoutOptimizationLevel
)

class TestReductionMemoryOptimization(unittest.TestCase):
    """Test reduction memory optimization with COALESCED layout"""
    
    def setUp(self):
        """Set up test case"""
        # Get the memory manager
        self.memory_manager = get_metal_memory_manager()
        
        # Create a reduction layout pattern
        self.pattern = ReductionLayoutPattern()
        
        # Create a memory layout optimizer
        self.optimizer = MemoryLayoutOptimizer(
            optimization_level=LayoutOptimizationLevel.AGGRESSIVE
        )
        
        # Define test operations with different shapes and axes
        self.test_operations = [
            {
                "type": "tt.reduce",
                "id": "reduce1",
                "input_shapes": [[1024]],
                "args": {"axis": 0},
                "output_shape": [1]
            },
            {
                "type": "tt.reduce",
                "id": "reduce2",
                "input_shapes": [[1024, 1024]],
                "args": {"axis": 1},
                "output_shape": [1024, 1]
            },
            {
                "type": "tt.sum",
                "id": "sum1",
                "input_shapes": [[32, 64, 128]],
                "args": {"axis": 2},
                "output_shape": [32, 64, 1]
            },
            {
                "type": "tt.mean",
                "id": "mean1",
                "input_shapes": [[1, 1024, 1024]],
                "args": {"axis": [0, 1]},
                "output_shape": [1, 1, 1024]
            }
        ]
        
        # Create a test graph with all operations
        self.test_graph = {
            "ops": self.test_operations
        }
    
    def test_memory_manager_reduction_optimization(self):
        """Test that memory manager applies COALESCED layout to reductions"""
        print("Testing memory manager reduction optimization:")
        
        # Test each operation
        for op in self.test_operations:
            op_type = op["type"]
            input_shape = op["input_shapes"][0]
            axis = op["args"]["axis"]
            
            print(f"\nOptimizing {op_type} with shape {input_shape} along axis {axis}:")
            
            # Optimize the operation
            optimized_op = self.memory_manager._optimize_reduction_memory(op.copy())
            
            # Check if execution parameters were set
            self.assertIn("execution_parameters", optimized_op)
            exec_params = optimized_op["execution_parameters"]
            
            # Check memory layout
            self.assertIn("memory_layout", exec_params)
            self.assertEqual(exec_params["memory_layout"], MemoryLayout.COALESCED.value)
            
            print(f"  ✅ Memory layout correctly set to COALESCED ({MemoryLayout.COALESCED.value})")
    
    def test_reduction_pattern_applicability(self):
        """Test that reduction pattern correctly identifies reduction operations"""
        print("\nTesting reduction pattern applicability:")
        
        # Test each operation
        for op in self.test_operations:
            op_type = op["type"]
            
            # Check if pattern is applicable
            is_applicable = self.pattern.is_applicable(op, None)
            
            # Verify that pattern is applicable
            self.assertTrue(is_applicable)
            print(f"  ✅ Pattern applicable to {op_type}")
    
    def test_optimal_layout_for_reductions(self):
        """Test that pattern returns COALESCED as optimal layout"""
        print("\nTesting optimal layout for reductions:")
        
        # Test each operation
        for op in self.test_operations:
            shape = op["input_shapes"][0]
            
            # Get optimal layout
            optimal_layout = self.pattern.get_optimal_layout(shape, None)
            
            # Verify that COALESCED is returned
            self.assertEqual(optimal_layout, MemoryLayout.COALESCED)
            print(f"  ✅ Optimal layout for shape {shape} is COALESCED")
    
    def test_optimizer_integration(self):
        """Test that optimizer applies COALESCED layout in full graph"""
        print("\nTesting optimizer integration:")
        
        # Optimize the graph
        optimized_graph, stats = self.optimizer.optimize(self.test_graph.copy())
        
        print(f"  Optimization statistics: {stats}")
        
        # Check each operation in the optimized graph
        for op in optimized_graph["ops"]:
            # Check for layout hints
            self.assertIn("layout_hints", op)
            layout_hints = op["layout_hints"]
            
            # Check that layout is COALESCED
            self.assertIn("layout", layout_hints)
            self.assertEqual(layout_hints["layout"], "COALESCED")
            
            print(f"  ✅ Layout hints for {op['id']} correctly set to COALESCED")
    
    def test_different_reduction_types(self):
        """Test that different reduction types are correctly optimized"""
        print("\nTesting different reduction types:")
        
        # Define different reduction types
        reduction_ops = [
            {"type": "tt.reduce", "args": {"axis": 0}},
            {"type": "tt.sum", "args": {"axis": 0}},
            {"type": "tt.mean", "args": {"axis": 0}},
            {"type": "tt.max", "args": {"axis": 0}},
            {"type": "tt.min", "args": {"axis": 0}},
            {"type": "mlx.reduce", "args": {"axis": 0}},
            {"type": "mlx.sum", "args": {"axis": 0}}
        ]
        
        # Test each operation type
        for op in reduction_ops:
            op_type = op["type"]
            
            # Check if pattern is applicable
            is_applicable = self.pattern.is_applicable(op, None)
            
            if "tt." in op_type or "mlx." in op_type and any(x in op_type for x in ["reduce", "sum", "mean", "max", "min"]):
                # Verify that pattern is applicable
                self.assertTrue(is_applicable)
                print(f"  ✅ Pattern applicable to {op_type}")
            else:
                # Verify that pattern is not applicable to non-reduction ops
                self.assertFalse(is_applicable)
                print(f"  ✅ Pattern correctly not applicable to {op_type}")

if __name__ == "__main__":
    # Run the tests
    unittest.main() 