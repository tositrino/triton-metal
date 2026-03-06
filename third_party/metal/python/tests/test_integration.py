"""
Integration test for Metal optimization components.

This test verifies that the memory layout optimizer and metal memory manager
work correctly together, especially for reduction operations using the
COALESCED layout.
"""

import os
import sys
import unittest
from enum import Enum
from typing import Dict, Any


# Import the memory layout optimizer and metal memory manager
from MLX.memory_layout_optimizer import (
    MemoryLayoutOptimizer,
    LayoutOptimizationLevel,
    TensorType,
    MemoryLayout as OptimizerMemoryLayout,
    MemoryAccessPattern,
    MatrixLayoutPattern,
    ConvolutionLayoutPattern,
    ReductionLayoutPattern,
    optimize_memory_layout
)

from MLX.metal_memory_manager import (
    MetalMemoryManager,
    get_metal_memory_manager,
    MemoryLayout as ManagerMemoryLayout
)

# Try to import metal hardware detection
try:
    from MLX.metal_hardware_optimizer import AppleSiliconGeneration
except ImportError:
    # Define dummy enum for hardware generation
    class AppleSiliconGeneration(Enum):
        UNKNOWN = 0
        M1 = 1
        M2 = 2
        M3 = 3

# Try to import the metal optimizing compiler
try:
    from MLX.metal_optimizing_compiler import (
        MetalOptimizingCompiler,
        OptimizationLevel
    )
    COMPILER_AVAILABLE = True
except ImportError:
    COMPILER_AVAILABLE = False
    print("Warning: Metal optimization components could not be imported: optimizing compiler not available")

class TestMetalOptimizationIntegration(unittest.TestCase):
    """Integration tests for Metal optimization components"""

    def setUp(self):
        """Set up test cases"""
        # Create a memory layout optimizer
        self.layout_optimizer = MemoryLayoutOptimizer(
            optimization_level=LayoutOptimizationLevel.AGGRESSIVE
        )

        # Get memory manager instance
        self.memory_manager = get_metal_memory_manager()

        # Set hardware generation for tests
        self.hardware_gen = AppleSiliconGeneration.M3

        # Create test operations
        self.reduction_ops = [
            # 1D reduction
            {
                "type": "tt.reduce",
                "id": "reduce1d",
                "input_shapes": [[1024]],
                "args": {"axis": 0},
                "output_shape": [1]
            },
            # 2D reduction along axis 1
            {
                "type": "tt.reduce",
                "id": "reduce2d_axis1",
                "input_shapes": [[1024, 1024]],
                "args": {"axis": 1},
                "output_shape": [1024, 1]
            },
            # 2D reduction along axis 0
            {
                "type": "tt.reduce",
                "id": "reduce2d_axis0",
                "input_shapes": [[1024, 1024]],
                "args": {"axis": 0},
                "output_shape": [1, 1024]
            },
            # 3D reduction along single axis
            {
                "type": "tt.sum",
                "id": "sum3d",
                "input_shapes": [[32, 64, 128]],
                "args": {"axis": 2},
                "output_shape": [32, 64, 1]
            },
            # 3D reduction along multiple axes
            {
                "type": "tt.mean",
                "id": "mean3d",
                "input_shapes": [[32, 64, 128]],
                "args": {"axis": [0, 1]},
                "output_shape": [1, 1, 128]
            }
        ]

        # Other operation types for comparison
        self.matmul_op = {
            "type": "tt.matmul",
            "id": "matmul1",
            "input_shapes": [[128, 256], [256, 512]],
            "output_shape": [128, 512]
        }

        self.elementwise_op = {
            "type": "tt.add",
            "id": "add1",
            "input_shapes": [[128, 128], [128, 128]],
            "output_shape": [128, 128]
        }

        # Create test graph with all operations
        self.test_graph = {
            "ops": [
                self.matmul_op,
                self.elementwise_op,
                *self.reduction_ops
            ]
        }

    def test_coalesced_enum_consistency(self):
        """Test that COALESCED enum is consistent across components"""
        # Check that COALESCED is defined in both modules
        optimizer_coalesced = OptimizerMemoryLayout.COALESCED.value

        # Get COALESCED value from memory manager using a simple reduction
        test_reduce_op = self.reduction_ops[0].copy()
        memory_manager_coalesced = self.memory_manager._optimize_reduction_memory(
            test_reduce_op
        )["execution_parameters"]["memory_layout"]

        # Check that the values are the same
        self.assertEqual(optimizer_coalesced, memory_manager_coalesced)

        # Check the actual value is 8
        self.assertEqual(optimizer_coalesced, 8)
        self.assertEqual(memory_manager_coalesced, 8)

    def test_reduction_pattern(self):
        """Test that ReductionLayoutPattern correctly identifies reduction operations"""
        # Create a reduction layout pattern
        reduction_pattern = ReductionLayoutPattern()

        # Test each reduction operation
        for op in self.reduction_ops:
            # Check if the pattern is applicable to a reduction operation
            is_applicable = reduction_pattern.is_applicable(
                op,
                self.hardware_gen
            )

            # Verify that pattern applies to all reduction operations
            self.assertTrue(is_applicable, f"Pattern should apply to {op['type']}")

            # Get optimal layout and verify it's COALESCED
            shape = op["input_shapes"][0]
            optimal_layout = reduction_pattern.get_optimal_layout(shape, self.hardware_gen)
            self.assertEqual(optimal_layout, OptimizerMemoryLayout.COALESCED)

        # Verify pattern does not apply to non-reduction operations
        self.assertFalse(
            reduction_pattern.is_applicable(self.matmul_op, self.hardware_gen),
            "Pattern should not apply to matmul"
        )

        self.assertFalse(
            reduction_pattern.is_applicable(self.elementwise_op, self.hardware_gen),
            "Pattern should not apply to elementwise operations"
        )

    def test_memory_manager_reduction_optimization(self):
        """Test memory manager's reduction optimization"""
        # Test each reduction operation
        for op in self.reduction_ops:
            optimized_op = self.memory_manager._optimize_reduction_memory(op.copy())

            # Check that memory layout is set to COALESCED
            self.assertIn("execution_parameters", optimized_op)
            exec_params = optimized_op["execution_parameters"]
            self.assertEqual(
                exec_params["memory_layout"],
                ManagerMemoryLayout.COALESCED.value
            )

            # Check for hierarchical reduction flag
            self.assertIn("use_hierarchical_reduction", exec_params)

            # Check for M3-specific parameters
            # We're simulating M3 hardware in the memory manager
            self.assertIn("two_stage_reduction", exec_params)
            self.assertIn("use_simdgroup_reduction", exec_params)
            self.assertEqual(exec_params["vector_width"], 8)  # 8-wide for M3

    def test_layout_optimizer_integration(self):
        """Test that the layout optimizer applies the correct patterns"""
        # Run the layout optimizer on our operations
        optimized_result = self.layout_optimizer.optimize(self.test_graph)

        # The optimize function returns a tuple (optimized_graph, stats)
        optimized_graph = optimized_result[0]

        # Check each reduction operation in the optimized graph
        for op_id in [op["id"] for op in self.reduction_ops]:
            # Find the reduction operation in the optimized graph
            reduction_found = False
            for op in optimized_graph["ops"]:
                if op["id"] == op_id:
                    reduction_found = True

                    # Check that layout hints are set
                    self.assertIn("layout_hints", op)

                    # Check that memory layout is specified
                    self.assertIn("layout", op["layout_hints"])

                    # Check that the layout is COALESCED
                    self.assertEqual(
                        op["layout_hints"]["layout"],
                        "COALESCED"
                    )

                    # Check for other layout hints
                    self.assertIn("hierarchical_reduction", op["layout_hints"])
                    break

            # Make sure we found the operation
            self.assertTrue(reduction_found, f"Reduction operation {op_id} not found in optimized graph")

    def test_optimize_memory_layout_function(self):
        """Test the optimize_memory_layout function"""
        # Use the direct function
        optimized_result = optimize_memory_layout(
            self.test_graph,
            optimization_level=LayoutOptimizationLevel.AGGRESSIVE
        )

        # The optimize_memory_layout function returns a tuple (optimized_graph, stats)
        optimized_graph = optimized_result[0]

        # Check each reduction operation in the optimized graph
        for op_id in [op["id"] for op in self.reduction_ops]:
            # Find the reduction operation in the optimized graph
            reduction_found = False
            for op in optimized_graph["ops"]:
                if op["id"] == op_id:
                    reduction_found = True

                    # Check that layout hints are set
                    self.assertIn("layout_hints", op)

                    # Check that memory layout is specified
                    self.assertIn("layout", op["layout_hints"])

                    # Check that the layout is COALESCED
                    self.assertEqual(
                        op["layout_hints"]["layout"],
                        "COALESCED"
                    )
                    break

            # Make sure we found the operation
            self.assertTrue(reduction_found, f"Reduction operation {op_id} not found in optimized graph")

    def test_hardware_specific_optimization(self):
        """Test hardware-specific optimization for reduction operations"""
        # Run the layout optimizer with hardware-specific optimization
        hardware_optimizer = MemoryLayoutOptimizer(
            optimization_level=LayoutOptimizationLevel.HARDWARE_SPECIFIC
        )

        # Optimize the graph
        optimized_result = hardware_optimizer.optimize(self.test_graph)
        optimized_graph = optimized_result[0]

        # Check each reduction operation in the optimized graph
        for op_id in [op["id"] for op in self.reduction_ops]:
            # Find the reduction operation in the optimized graph
            for op in optimized_graph["ops"]:
                if op["id"] == op_id:
                    # Check that layout hints are set
                    self.assertIn("layout_hints", op)

                    # Check for hardware-specific parameters
                    self.assertIn("vector_width", op["layout_hints"])

                    # Verify that M3-specific optimizations are applied
                    # On M3 hardware (real or simulated), vector_width should be 8
                    self.assertEqual(op["layout_hints"]["vector_width"], 8)

    @unittest.skipIf(not COMPILER_AVAILABLE, "Optimizing compiler not available")
    def test_optimizing_compiler_integration(self):
        """Test integration with the optimizing compiler if available"""
        # Create a MetalOptimizingCompiler
        compiler = MetalOptimizingCompiler(
            optimization_level=OptimizationLevel.AGGRESSIVE
        )

        # Compile the test graph
        compiled_result = compiler.compile(self.test_graph)

        # Check that we got a result
        self.assertIsNotNone(compiled_result)

        # Get optimization summary
        summary = compiler.get_optimization_summary()

        # Check that the summary contains pass statistics
        self.assertIn("pass_statistics", summary)

        # Check that pass statistics includes memory layout optimization
        self.assertIn("memory_layout_standard", summary["pass_statistics"])

if __name__ == "__main__":
    unittest.main()