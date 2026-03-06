"""
Tests for M3-specific graph optimizations in the Triton Metal backend

This module tests the optimization passes in the M3 graph optimizer, which leverage
features specific to Apple M3 GPUs like Dynamic Caching, hardware-accelerated ray
tracing, and hardware-accelerated mesh shading.
"""


import sys
import unittest
from unittest.mock import patch, MagicMock

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

# Define dummy hardware capabilities to simulate M3 hardware
class DummyEnum(Enum):
    UNKNOWN = 0
    M1 = 1
    M2 = 2
    M3 = 3

class DummyCapabilities:
    def __init__(self, gen=DummyEnum.M3):
        self.chip_generation = gen
        self.shared_memory_size = 65536 if gen == DummyEnum.M3 else 32768

# Create patch for hardware_capabilities
mock_hardware_capabilities = DummyCapabilities(DummyEnum.M3)
mock_m3_generation = DummyEnum.M3

# Mock the AppleSiliconGeneration and hardware_capabilities
sys.modules['metal_hardware_optimizer'] = MagicMock()
sys.modules['metal_hardware_optimizer'].AppleSiliconGeneration = DummyEnum
sys.modules['metal_hardware_optimizer'].hardware_capabilities = mock_hardware_capabilities

# Import modules to test after patching

from M3.m3_graph_optimizer import M3DynamicCachingPass, M3RayTracingPass
from M3.m3_graph_optimizer import M3SIMDGroupOptimizationPass, M3MemoryOptimizationPass
from M3.m3_graph_optimizer import M3GraphOptimizer
from MLX.mlx_graph_optimizer import MLXGraphOptimizer, optimize
from MLX.metal_fusion_optimizer import FusionOptimizer, FusionPattern
from MLX.metal_memory_manager import get_metal_memory_manager

class TestM3GraphOptimizer(unittest.TestCase):

    def setUp(self):
        """Set up test environment"""
        # Create sample operation graph for testing
        self.sample_graph = {
            "ops": [
                {
                    "id": 1,
                    "type": "matmul",
                    "inputs": [2, 3],
                    "input_shapes": [[128, 256], [256, 512]],
                    "output_shape": [128, 512]
                },
                {
                    "id": 4,
                    "type": "add",
                    "inputs": [1, 5],
                    "input_shapes": [[128, 512], [512]],
                    "output_shape": [128, 512]
                },
                {
                    "id": 6,
                    "type": "relu",
                    "inputs": [4],
                    "input_shapes": [[128, 512]],
                    "output_shape": [128, 512]
                },
                {
                    "id": 7,
                    "type": "matmul",
                    "inputs": [6, 8],
                    "input_shapes": [[128, 512], [512, 128]],
                    "output_shape": [128, 128]
                },
                {
                    "id": 9,
                    "type": "softmax",
                    "inputs": [7],
                    "input_shapes": [[128, 128]],
                    "output_shape": [128, 128]
                },
                {
                    "id": 10,
                    "type": "matmul",
                    "inputs": [9, 11],
                    "input_shapes": [[128, 128], [128, 512]],
                    "output_shape": [128, 512]
                }
            ]
        }

        # Sample operation for M3-specific optimizations
        self.matmul_op = {
            "id": 1,
            "type": "matmul",
            "inputs": [2, 3],
            "input_shapes": [[1024, 1024], [1024, 1024]],
            "output_shape": [1024, 1024]
        }

        self.attention_pattern = [
            {
                "id": 1,
                "type": "matmul",
                "inputs": [2, 3],
                "input_shapes": [[128, 512], [512, 128]],
                "output_shape": [128, 128]
            },
            {
                "id": 4,
                "type": "softmax",
                "inputs": [1],
                "input_shapes": [[128, 128]],
                "output_shape": [128, 128]
            },
            {
                "id": 5,
                "type": "matmul",
                "inputs": [4, 6],
                "input_shapes": [[128, 128], [128, 512]],
                "output_shape": [128, 512]
            }
        ]

    def test_m3_dynamic_caching_pass(self):
        """Test M3DynamicCachingPass optimization"""
        # Create optimization pass
        dynamic_caching_pass = M3DynamicCachingPass()

        # Apply optimization
        optimized_graph, stats = dynamic_caching_pass.apply(self.sample_graph)

        # Verify that optimization was applied
        self.assertIsNotNone(optimized_graph)
        self.assertIn("dynamic_caching_optimized_ops", stats)

        # Check that applicable operations (matmul) were optimized
        matmul_ops = [op for op in optimized_graph["ops"] if op["type"] == "matmul"]
        for op in matmul_ops:
            self.assertIn("metadata", op)
            self.assertEqual(op["metadata"]["dynamic_caching_enabled"], True)
            self.assertIn("execution_parameters", op)
            self.assertEqual(op["execution_parameters"]["prefer_dynamic_register_allocation"], True)

    def test_m3_simdgroup_optimization_pass(self):
        """Test M3SIMDGroupOptimizationPass optimization"""
        # Create optimization pass
        simdgroup_pass = M3SIMDGroupOptimizationPass()

        # Apply optimization
        optimized_graph, stats = simdgroup_pass.apply(self.sample_graph)

        # Verify that optimization was applied
        self.assertIsNotNone(optimized_graph)
        self.assertIn("simdgroup_optimized_ops", stats)

        # Check that matmul operations were optimized with SIMD group parameters
        matmul_ops = [op for op in optimized_graph["ops"] if op["type"] == "matmul"]
        for op in matmul_ops:
            self.assertIn("execution_parameters", op)
            self.assertIn("use_simdgroups", op["execution_parameters"])
            self.assertTrue(op["execution_parameters"]["use_simdgroups"])
            self.assertEqual(op["execution_parameters"]["simdgroup_width"], 32)

    def test_m3_memory_optimization_pass(self):
        """Test M3MemoryOptimizationPass optimization"""
        # Create optimization pass
        memory_pass = M3MemoryOptimizationPass()

        # Apply optimization
        optimized_graph, stats = memory_pass.apply(self.sample_graph)

        # Verify that optimization was applied
        self.assertIsNotNone(optimized_graph)
        self.assertIn("memory_optimized_ops", stats)

        # Check that operations with heavy memory access were optimized
        matmul_ops = [op for op in optimized_graph["ops"] if op["type"] == "matmul"]
        for op in matmul_ops:
            self.assertIn("execution_parameters", op)
            self.assertIn("shared_memory_size", op["execution_parameters"])
            self.assertEqual(op["execution_parameters"]["shared_memory_size"], 65536)  # 64KB for M3

    def test_metal_fusion_optimizer_m3_patterns(self):
        """Test that FusionOptimizer includes M3-specific patterns"""
        # Create fusion optimizer
        fusion_optimizer = FusionOptimizer(mock_hardware_capabilities)

        # Check that M3-specific patterns are included
        m3_patterns = [p for p in fusion_optimizer.patterns if p.name.startswith("m3_")]
        self.assertGreater(len(m3_patterns), 0)

        # Test fusion of attention pattern
        optimized_ops = fusion_optimizer.optimize(self.attention_pattern)

        # Verify fusion happened
        self.assertLess(len(optimized_ops), len(self.attention_pattern))

        # Check for M3-specific execution parameters in the fused operation
        fused_op = optimized_ops[0]
        self.assertIn("execution_parameters", fused_op)
        self.assertEqual(fused_op["execution_parameters"]["use_tensor_cores"], True)
        self.assertEqual(fused_op["execution_parameters"]["simdgroup_width"], 32)

    def test_memory_manager_m3_optimizations(self):
        """Test that MetalMemoryManager applies M3-specific optimizations"""
        # Get memory manager
        memory_manager = get_metal_memory_manager()

        # Create deep copy of matmul op to not affect other tests
        matmul_op = self.matmul_op.copy()

        # Apply memory optimizations
        optimized_op = memory_manager._optimize_matmul_memory(matmul_op)

        # Check M3-specific optimizations
        self.assertIn("execution_parameters", optimized_op)
        exec_params = optimized_op["execution_parameters"]

        # Expect M3-specific parameters
        self.assertEqual(exec_params["tile_m"], 128)
        self.assertEqual(exec_params["tile_n"], 128)
        self.assertEqual(exec_params["use_tensor_cores"], True)
        self.assertEqual(exec_params["vector_width"], 8)  # M3's 8-wide vectorization
        self.assertEqual(exec_params["shared_memory_size"], 65536)  # 64KB for M3

        # Check hierarchical reduction
        self.assertTrue(exec_params["use_hierarchical_reduction"])

        # Check dynamic shared memory
        self.assertTrue(exec_params["use_dynamic_shared_memory"])

    def test_end_to_end_m3_optimization(self):
        """Test end-to-end M3 optimization pipeline"""
        # Create graph optimizer
        m3_optimizer = M3GraphOptimizer()

        # Apply optimizations
        optimized_graph, stats = m3_optimizer.optimize(self.sample_graph)

        # Verify that appropriate passes were applied
        self.assertIn("dynamic_caching", stats)
        self.assertIn("simdgroup_optimization", stats)
        self.assertIn("memory_optimization", stats)

        # Check that the graph structure is preserved
        self.assertEqual(len(optimized_graph["ops"]), len(self.sample_graph["ops"]))

        # Check for M3 optimization feature flags
        features = m3_optimizer.get_available_features()
        self.assertTrue(all(features.values()))

    def test_integration_with_mlx_optimizer(self):
        """Test integration with MLX graph optimizer"""
        # Apply MLX graph optimization
        optimized_graph, stats = optimize(self.sample_graph)

        # Verify optimization results
        self.assertIsNotNone(optimized_graph)
        self.assertIn("passes", stats)

        # Check that M3 optimizations were included in passes
        m3_passes = [p for p in stats["passes"] if p.startswith("m3_")]
        self.assertGreater(len(m3_passes), 0)

if __name__ == "__main__":
    unittest.main()