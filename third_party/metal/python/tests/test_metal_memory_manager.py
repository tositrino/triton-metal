"""
Test suite for the Metal memory manager.

This test suite verifies that the Metal memory manager properly handles various
memory layout optimizations, with a particular focus on the COALESCED layout
for reduction operations.
"""


import unittest


# Import the metal memory manager
from MLX.metal_memory_manager import (
    MetalMemoryManager,
    get_metal_memory_manager,
    MemoryLayout
)

class TestMetalMemoryManager(unittest.TestCase):
    """Test case for Metal memory manager"""

    def setUp(self):
        """Set up test case"""
        # Get the memory manager
        self.memory_manager = get_metal_memory_manager()

        # Create sample operations for testing
        self.reduction_op = {
            "type": "tt.reduce",
            "id": "reduction_op",
            "input_shapes": [[1024, 1024]],
            "args": {"axis": 1},
            "output_shape": [1024, 1]
        }

        self.matmul_op = {
            "type": "tt.matmul",
            "id": "matmul_op",
            "input_shapes": [[128, 256], [256, 512]],
            "output_shape": [128, 512]
        }

        self.elementwise_op = {
            "type": "tt.add",
            "id": "elementwise_op",
            "input_shapes": [[512, 512], [512, 512]],
            "output_shape": [512, 512]
        }

        # Create a sample graph with all operations
        self.sample_graph = {
            "ops": [
                self.reduction_op,
                self.matmul_op,
                self.elementwise_op
            ]
        }

    def test_memory_layout_enum(self):
        """Test that MemoryLayout enum includes COALESCED value"""
        # Verify COALESCED is defined in MemoryLayout enum
        self.assertTrue(hasattr(MemoryLayout, "COALESCED"))

        # Verify COALESCED has the expected value (8)
        self.assertEqual(MemoryLayout.COALESCED.value, 8)

        # Print all memory layouts for debugging
        print("Available memory layouts:")
        for layout in MemoryLayout:
            print(f"  {layout.name}: {layout.value}")

    def test_reduction_optimization(self):
        """Test reduction memory optimization with COALESCED layout"""
        # Apply reduction optimization
        optimized_op = self.memory_manager._optimize_reduction_memory(self.reduction_op.copy())

        # Verify execution parameters are set
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]

        # Verify memory layout is set to COALESCED
        self.assertIn("memory_layout", exec_params)
        self.assertEqual(exec_params["memory_layout"], MemoryLayout.COALESCED.value)

        # Verify other reduction-specific parameters
        self.assertIn("use_hierarchical_reduction", exec_params)
        self.assertIn("vector_width", exec_params)

        # Print execution parameters for debugging
        print("\nReduction execution parameters:")
        for param, value in exec_params.items():
            print(f"  {param}: {value}")

    def test_matmul_optimization(self):
        """Test matrix multiplication memory optimization"""
        # Apply matmul optimization
        optimized_op = self.memory_manager._optimize_matmul_memory(self.matmul_op.copy())

        # Verify execution parameters are set
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]

        # Verify memory layout is set
        self.assertIn("memory_layout", exec_params)

        # Print execution parameters for debugging
        print("\nMatmul execution parameters:")
        for param, value in exec_params.items():
            print(f"  {param}: {value}")

    def test_elementwise_optimization(self):
        """Test elementwise operation memory optimization"""
        # Apply elementwise optimization
        optimized_op = self.memory_manager._optimize_elementwise_memory(self.elementwise_op.copy())

        # Verify execution parameters are set
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]

        # Verify memory layout is set
        self.assertIn("memory_layout", exec_params)

        # Print execution parameters for debugging
        print("\nElementwise execution parameters:")
        for param, value in exec_params.items():
            print(f"  {param}: {value}")

    def test_operation_optimization(self):
        """Test the _optimize_operation_memory method which routes to specific optimizers"""
        # Optimize each type of operation
        operations = [
            self.reduction_op.copy(),
            self.matmul_op.copy(),
            self.elementwise_op.copy()
        ]

        for op in operations:
            optimized_op = self.memory_manager._optimize_operation_memory(op)

            # Verify execution parameters are set
            self.assertIn("execution_parameters", optimized_op)

            # Check that memory layout is set
            self.assertIn("memory_layout", optimized_op["execution_parameters"])

            # For reduction operations, verify COALESCED layout
            if op["type"] == "tt.reduce":
                self.assertEqual(
                    optimized_op["execution_parameters"]["memory_layout"],
                    MemoryLayout.COALESCED.value
                )
                print(f"Operation {op['type']} optimized with COALESCED layout")
            else:
                print(f"Operation {op['type']} optimized with layout: {optimized_op['execution_parameters']['memory_layout']}")

    def test_graph_memory_optimization(self):
        """Test full graph memory optimization"""
        # Optimize the entire graph
        optimized_graph = self.memory_manager.optimize_graph_memory(self.sample_graph.copy())

        # Check each operation in the optimized graph
        for op in optimized_graph["ops"]:
            op_id = op["id"]
            self.assertIn("execution_parameters", op)

            # Check memory layout is set
            self.assertIn("memory_layout", op["execution_parameters"])

            if op_id == "reduction_op":
                # Verify reduction uses COALESCED layout
                self.assertEqual(
                    op["execution_parameters"]["memory_layout"],
                    MemoryLayout.COALESCED.value
                )
                print(f"Graph optimization: {op_id} uses COALESCED layout")
            else:
                # Print the layout used for other operations
                print(f"Graph optimization: {op_id} uses layout {op['execution_parameters']['memory_layout']}")

    def test_reduction_variations(self):
        """Test different variations of reduction operations"""
        # Create different reduction operations
        reduction_variations = [
            # 1D reduction
            {
                "type": "tt.reduce",
                "input_shapes": [[1024]],
                "args": {"axis": 0}
            },
            # 2D reduction on axis 0
            {
                "type": "tt.reduce",
                "input_shapes": [[1024, 512]],
                "args": {"axis": 0}
            },
            # 3D reduction on multiple axes
            {
                "type": "tt.reduce",
                "input_shapes": [[64, 64, 64]],
                "args": {"axis": [0, 1]}
            },
            # Different reduction types
            {
                "type": "tt.sum",
                "input_shapes": [[512, 512]],
                "args": {"axis": 1}
            },
            {
                "type": "tt.max",
                "input_shapes": [[128, 256]],
                "args": {"axis": 0}
            }
        ]

        # Test each variation
        for op in reduction_variations:
            optimized_op = self.memory_manager._optimize_reduction_memory(op.copy())

            # Verify COALESCED layout is used for all reduction variations
            self.assertEqual(
                optimized_op["execution_parameters"]["memory_layout"],
                MemoryLayout.COALESCED.value,
                f"Failed for {op['type']} with shape {op['input_shapes']} on axis {op['args']['axis']}"
            )

if __name__ == "__main__":
    unittest.main()