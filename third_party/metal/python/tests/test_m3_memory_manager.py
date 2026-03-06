"""
Tests for M3-specific memory manager in the Triton Metal backend

This module tests the memory optimization strategies in the M3 memory manager, which
leverage features specific to Apple M3 GPUs like flexible on-chip memory and dynamic caching.
"""

import unittest
from enum import Enum

from requests import patch


# Import modules to test
try:
    from M3.m3_memory_manager import (
        M3MemoryManager, M3MemoryLayout, TensorType,
        get_m3_memory_manager
    )

    from MLX.metal_hardware_optimizer import AppleSiliconGeneration
    has_modules = True
except ImportError:
    has_modules = False

    # Define mocks for testing without real modules
    class AppleSiliconGeneration(Enum):
        UNKNOWN = 0
        M1 = 1
        M2 = 2
        M3 = 3

    class M3MemoryLayout(Enum):
        ROW_MAJOR = 0
        COLUMN_MAJOR = 1
        BLOCK_BASED_64 = 2
        BLOCK_BASED_128 = 3
        TEXTURE_OPTIMIZED = 4
        SIMDGROUP_OPTIMIZED = 5
        DYNAMIC_CACHED = 6

    class TensorType(Enum):
        MATRIX = 0
        VECTOR = 1
        CONV_FILTER = 2
        FEATURE_MAP = 3
        ELEMENTWISE = 4
        REDUCTION = 5
        ATTENTION = 6
        RAY_TRACING = 7
        MESH_DATA = 8
        IMAGE = 9

class MockHardwareCapabilities:
    """Mock hardware capabilities for testing"""

    def __init__(self, chip_generation=AppleSiliconGeneration.M3):
        self.chip_generation = chip_generation
        self.shared_memory_size = 65536 if chip_generation == AppleSiliconGeneration.M3 else 32768

@unittest.skipIf(not has_modules, "Required modules not available")
class TestM3MemoryManager(unittest.TestCase):
    """Test M3-specific memory manager"""

    def setUp(self):
        """Set up test environment"""
        # Mock hardware capabilities before instantiating the manager
        patcher = patch('m3_memory_manager.hardware_capabilities',
                        MockHardwareCapabilities(AppleSiliconGeneration.M3))
        self.mock_hw_caps = patcher.start()
        self.addCleanup(patcher.stop)

        # Create memory manager
        self.memory_manager = M3MemoryManager()

    def test_m3_detection(self):
        """Test M3 hardware detection"""
        # M3 should be detected
        self.assertTrue(self.memory_manager.is_m3_available())

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            self.assertFalse(memory_manager.is_m3_available())

    def test_m3_parameters(self):
        """Test M3-specific memory parameters"""
        # Check M3 parameters
        self.assertEqual(self.memory_manager.shared_memory_size, 65536)
        self.assertEqual(self.memory_manager.preferred_tile_size, 128)
        self.assertEqual(self.memory_manager.simdgroup_width, 32)
        self.assertTrue(self.memory_manager.supports_dynamic_caching)
        self.assertTrue(self.memory_manager.supports_flexible_memory)

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            self.assertEqual(memory_manager.shared_memory_size, 32768)
            self.assertEqual(memory_manager.preferred_tile_size, 64)
            self.assertEqual(memory_manager.simdgroup_width, 16)
            self.assertFalse(memory_manager.supports_dynamic_caching)
            self.assertFalse(memory_manager.supports_flexible_memory)

    def test_optimal_layout(self):
        """Test optimal memory layout selection"""
        # Test matrix layouts
        small_matrix_shape = [64, 64]
        large_matrix_shape = [256, 256]

        # Large matrices should use 128x128 blocks
        layout = self.memory_manager.get_optimal_layout(TensorType.MATRIX, large_matrix_shape)
        self.assertEqual(layout, M3MemoryLayout.BLOCK_BASED_128)

        # Small matrices should use 64x64 blocks
        layout = self.memory_manager.get_optimal_layout(TensorType.MATRIX, small_matrix_shape)
        self.assertEqual(layout, M3MemoryLayout.BLOCK_BASED_64)

        # Convolution filters should use SIMDGROUP_OPTIMIZED
        layout = self.memory_manager.get_optimal_layout(TensorType.CONV_FILTER, [64, 3, 3, 3])
        self.assertEqual(layout, M3MemoryLayout.SIMDGROUP_OPTIMIZED)

        # Feature maps should use TEXTURE_OPTIMIZED
        layout = self.memory_manager.get_optimal_layout(TensorType.FEATURE_MAP, [1, 64, 32, 32])
        self.assertEqual(layout, M3MemoryLayout.TEXTURE_OPTIMIZED)

        # Ray tracing data should use DYNAMIC_CACHED
        layout = self.memory_manager.get_optimal_layout(TensorType.RAY_TRACING, [1024])
        self.assertEqual(layout, M3MemoryLayout.DYNAMIC_CACHED)

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            layout = memory_manager.get_optimal_layout(TensorType.MATRIX, large_matrix_shape)
            self.assertEqual(layout, M3MemoryLayout.BLOCK_BASED_64)

    def test_threadgroup_size(self):
        """Test optimal threadgroup size selection"""
        # Matrix operations should use 1024 threads on M3
        size = self.memory_manager.get_optimal_threadgroup_size(TensorType.MATRIX, [128, 128])
        self.assertEqual(size, 1024)

        # Reduction operations should use 512 threads on M3
        size = self.memory_manager.get_optimal_threadgroup_size(TensorType.REDUCTION, [1024])
        self.assertEqual(size, 512)

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            size = memory_manager.get_optimal_threadgroup_size(TensorType.MATRIX, [128, 128])
            self.assertEqual(size, 256)

    def test_tile_size(self):
        """Test optimal tile size selection"""
        # Matrix operations should use 128x128 tiles on M3
        tile_width, tile_height = self.memory_manager.get_optimal_tile_size(TensorType.MATRIX, [1024, 1024])
        self.assertEqual(tile_width, 128)
        self.assertEqual(tile_height, 128)

        # Reduction operations should use 256x32 tiles on M3
        tile_width, tile_height = self.memory_manager.get_optimal_tile_size(TensorType.REDUCTION, [1024])
        self.assertEqual(tile_width, 256)
        self.assertEqual(tile_height, 32)

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            tile_width, tile_height = memory_manager.get_optimal_tile_size(TensorType.MATRIX, [1024, 1024])
            self.assertEqual(tile_width, 64)
            self.assertEqual(tile_height, 64)

    def test_vector_width(self):
        """Test optimal vector width selection"""
        # Element-wise operations should use 8-wide vectors on M3
        width = self.memory_manager.get_optimal_vector_width(TensorType.ELEMENTWISE)
        self.assertEqual(width, 8)

        # Matrix operations should use 4-wide vectors on M3
        width = self.memory_manager.get_optimal_vector_width(TensorType.MATRIX)
        self.assertEqual(width, 4)

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            width = memory_manager.get_optimal_vector_width(TensorType.ELEMENTWISE)
            self.assertEqual(width, 4)

    def test_buffer_allocation(self):
        """Test buffer allocation"""
        # Matrix buffers should be optimized for M3
        buffer = self.memory_manager.allocate_buffer(1024, TensorType.MATRIX)
        self.assertEqual(buffer["type"], "matrix")
        self.assertEqual(buffer["layout"], M3MemoryLayout.BLOCK_BASED_128.name)
        self.assertEqual(buffer["alignment"], 256)

        # Vector buffers should use 8-wide vectorization on M3
        buffer = self.memory_manager.allocate_buffer(1024, TensorType.VECTOR)
        self.assertEqual(buffer["type"], "vector")
        self.assertEqual(buffer["vector_width"], 8)

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            buffer = memory_manager.allocate_buffer(1024, TensorType.MATRIX)
            self.assertEqual(buffer["type"], "standard")
            self.assertEqual(buffer["alignment"], 16)

    def test_memory_access_optimization(self):
        """Test memory access optimization"""
        # Create a sample operation
        op = {
            "id": 1,
            "type": "matmul",
            "name": "matmul_op",
            "shape": [128, 128]
        }

        # Optimize for matrix operations
        optimized_op = self.memory_manager.optimize_memory_access(op, TensorType.MATRIX)

        # Check optimization parameters
        self.assertIn("execution_parameters", optimized_op)
        self.assertEqual(optimized_op["execution_parameters"]["memory_layout"], M3MemoryLayout.BLOCK_BASED_128.name)
        self.assertEqual(optimized_op["threadgroup_size"], 1024)
        self.assertEqual(optimized_op["execution_parameters"]["tile_width"], 128)
        self.assertEqual(optimized_op["execution_parameters"]["tile_height"], 128)
        self.assertEqual(optimized_op["execution_parameters"]["vector_width"], 4)
        self.assertTrue(optimized_op["execution_parameters"]["use_tensor_cores"])
        self.assertTrue(optimized_op["execution_parameters"]["use_flexible_memory"])

        # Test optimization for reduction operations
        op = {
            "id": 2,
            "type": "reduce",
            "name": "reduce_op",
            "shape": [1024]
        }

        optimized_op = self.memory_manager.optimize_memory_access(op, TensorType.REDUCTION)
        self.assertTrue(optimized_op["execution_parameters"]["use_hierarchical_reduction"])

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            optimized_op = memory_manager.optimize_memory_access(op, TensorType.MATRIX)
            self.assertEqual(optimized_op, op)

    def test_matrix_multiplication_strategy(self):
        """Test matrix multiplication strategy"""
        # Large matrices
        strategy = self.memory_manager.get_matrix_multiplication_strategy(1024, 1024, 1024)
        self.assertEqual(strategy["tile_m"], 128)
        self.assertEqual(strategy["tile_n"], 128)
        self.assertEqual(strategy["tile_k"], 16)
        self.assertTrue(strategy["use_tensor_cores"])
        self.assertTrue(strategy["use_dynamic_caching"])
        self.assertEqual(strategy["simdgroup_size"], 32)
        self.assertEqual(strategy["vector_width"], 8)

        # Small matrices
        strategy = self.memory_manager.get_matrix_multiplication_strategy(128, 128, 128)
        self.assertEqual(strategy["tile_m"], 32)
        self.assertEqual(strategy["tile_n"], 32)

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            strategy = memory_manager.get_matrix_multiplication_strategy(1024, 1024, 1024)
            self.assertEqual(strategy["tile_m"], 64)
            self.assertEqual(strategy["tile_n"], 64)
            self.assertEqual(strategy["tile_k"], 8)
            self.assertNotIn("simdgroup_size", strategy)

    def test_convolution_strategy(self):
        """Test convolution strategy"""
        # Large feature maps
        strategy = self.memory_manager.get_convolution_strategy([1, 64, 128, 128], [128, 64, 3, 3])
        self.assertEqual(strategy["tile_h"], 32)
        self.assertEqual(strategy["tile_w"], 32)
        self.assertEqual(strategy["tile_k"], 64)
        self.assertTrue(strategy["use_tensor_cores"])
        self.assertTrue(strategy["use_dynamic_caching"])

        # Small feature maps
        strategy = self.memory_manager.get_convolution_strategy([1, 64, 32, 32], [128, 64, 3, 3])
        self.assertEqual(strategy["tile_h"], 16)
        self.assertEqual(strategy["tile_w"], 16)

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            strategy = memory_manager.get_convolution_strategy([1, 64, 128, 128], [128, 64, 3, 3])
            self.assertEqual(strategy["tile_size"], 64)
            self.assertTrue(strategy["vectorize"])
            self.assertNotIn("tile_h", strategy)

    def test_reduction_strategy(self):
        """Test reduction strategy"""
        # Large reduction
        strategy = self.memory_manager.get_reduction_strategy([1, 8192], 1)
        self.assertTrue(strategy["hierarchical_reduction"])
        self.assertEqual(strategy["block_size"], 1024)
        self.assertEqual(strategy["subgroup_size"], 32)

        # Small reduction
        strategy = self.memory_manager.get_reduction_strategy([1, 512], 1)
        self.assertEqual(strategy["block_size"], 256)
        self.assertFalse(strategy["hierarchical_reduction"])

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            strategy = memory_manager.get_reduction_strategy([1, 8192], 1)
            self.assertEqual(strategy["block_size"], 256)
            self.assertTrue(strategy["use_shared_memory"])
            self.assertNotIn("hierarchical_reduction", strategy)

    def test_graph_optimization(self):
        """Test graph-level memory optimization"""
        # Create a sample graph
        graph = {
            "ops": [
                {
                    "id": 1,
                    "type": "matmul",
                    "name": "matmul_op",
                    "shape": [128, 128]
                },
                {
                    "id": 2,
                    "type": "reduce",
                    "name": "reduce_op",
                    "shape": [1024]
                }
            ]
        }

        # Optimize the graph
        optimized_graph = self.memory_manager.optimize_graph_memory(graph)

        # Check that both operations were optimized
        self.assertEqual(len(optimized_graph["ops"]), 2)

        # Check that graph-level metadata was added
        self.assertIn("metadata", optimized_graph)
        self.assertTrue(optimized_graph["metadata"]["m3_memory_optimized"])
        self.assertEqual(optimized_graph["metadata"]["shared_memory_size"], 65536)
        self.assertTrue(optimized_graph["metadata"]["dynamic_caching_enabled"])
        self.assertTrue(optimized_graph["metadata"]["flexible_memory_enabled"])

        # Check that individual operations were optimized
        for op in optimized_graph["ops"]:
            self.assertIn("execution_parameters", op)
            self.assertIn("memory_layout", op["execution_parameters"])
            self.assertIn("threadgroup_size", op)

        # Test with non-M3 hardware
        with patch('m3_memory_manager.hardware_capabilities',
                  MockHardwareCapabilities(AppleSiliconGeneration.M2)):
            memory_manager = M3MemoryManager()
            optimized_graph = memory_manager.optimize_graph_memory(graph)
            self.assertEqual(optimized_graph, graph)

    def test_tensor_type_detection(self):
        """Test tensor type detection from operation type"""
        # Test various operation types
        self.assertEqual(self.memory_manager._get_tensor_type_for_op("matmul"), TensorType.MATRIX)
        self.assertEqual(self.memory_manager._get_tensor_type_for_op("conv2d"), TensorType.CONV_FILTER)
        self.assertEqual(self.memory_manager._get_tensor_type_for_op("reduce"), TensorType.REDUCTION)
        self.assertEqual(self.memory_manager._get_tensor_type_for_op("add"), TensorType.ELEMENTWISE)
        self.assertEqual(self.memory_manager._get_tensor_type_for_op("attention"), TensorType.ATTENTION)
        self.assertEqual(self.memory_manager._get_tensor_type_for_op("ray_intersect"), TensorType.RAY_TRACING)
        self.assertEqual(self.memory_manager._get_tensor_type_for_op("mesh"), TensorType.MESH_DATA)
        self.assertEqual(self.memory_manager._get_tensor_type_for_op("unknown"), TensorType.VECTOR)

@unittest.skipIf(not has_modules, "Required modules not available")
class TestSingleton(unittest.TestCase):
    """Test singleton access to M3 memory manager"""

    def test_get_m3_memory_manager(self):
        """Test get_m3_memory_manager function"""
        memory_manager = get_m3_memory_manager()
        self.assertIsNotNone(memory_manager)
        self.assertIsInstance(memory_manager, M3MemoryManager)

        # Should return the same instance on subsequent calls
        memory_manager2 = get_m3_memory_manager()
        self.assertIs(memory_manager, memory_manager2)

if __name__ == "__main__":
    unittest.main()