#!/usr/bin/env python
"""
Test MLX Graph Optimizer

This script tests the MLX graph optimizer functionality with a focus on
M3-specific optimizations.
"""

import os
import sys
import unittest
import json
from typing import Dict, List, Any, Optional, Tuple, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    print("MLX not found. Please install it with 'pip install mlx'")
    MLX_AVAILABLE = False

# Import our modules
try:
    from python.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
    from python.mlx_graph_optimizer import (
        optimize, MLXGraphOptimizer, 
        OptimizationPass, OperationFusionPass,
        M3SpecificFusionPass, MemoryAccessOptimizationPass,
        MetalSpecificOptimizationPass
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this from the metal module root directory")
    sys.exit(1)

class TestMLXGraphOptimizer(unittest.TestCase):
    """Test the MLX Graph Optimizer"""
    
    def setUp(self):
        """Set up test case"""
        # Check if we're running on Apple Silicon
        if not hasattr(hardware_capabilities, "chip_generation") or \
           hardware_capabilities.chip_generation == AppleSiliconGeneration.UNKNOWN:
            self.skipTest("Not running on Apple Silicon")
        
        # Record the chip generation for test adjustments
        self.chip_generation = hardware_capabilities.chip_generation
        
        # Create a test graph optimizer
        self.optimizer = MLXGraphOptimizer(hardware_capabilities)
        
        # Create test computation graphs
        self.matmul_graph = self._create_matmul_test_graph()
        self.conv_graph = self._create_conv_test_graph()
        self.attention_graph = self._create_attention_test_graph()
        self.swiglu_graph = self._create_swiglu_test_graph()
        
    def _create_matmul_test_graph(self) -> Dict:
        """Create a test graph with matrix multiplication"""
        return {
            "ops": [
                {
                    "id": "op1",
                    "type": "tt.matmul",
                    "a_id": "input1",
                    "b_id": "input2",
                    "a_shape": [128, 64],
                    "b_shape": [64, 128],
                    "output_shape": [128, 128]
                },
                {
                    "id": "op2",
                    "type": "tt.binary.add",
                    "lhs_id": "op1",
                    "rhs_id": "input3",
                    "lhs_shape": [128, 128],
                    "rhs_shape": [128, 128],
                    "output_shape": [128, 128]
                }
            ],
            "inputs": ["input1", "input2", "input3"],
            "outputs": ["op2"]
        }
        
    def _create_conv_test_graph(self) -> Dict:
        """Create a test graph with convolution"""
        return {
            "ops": [
                {
                    "id": "op1",
                    "type": "tt.conv",
                    "input_id": "input1",
                    "filter_id": "input2",
                    "input_shape": [1, 64, 32, 32],
                    "filter_shape": [64, 64, 3, 3],
                    "output_shape": [1, 64, 32, 32],
                    "stride": [1, 1],
                    "padding": [1, 1]
                },
                {
                    "id": "op2",
                    "type": "tt.unary.relu",
                    "operand_id": "op1",
                    "operand_shape": [1, 64, 32, 32],
                    "output_shape": [1, 64, 32, 32]
                }
            ],
            "inputs": ["input1", "input2"],
            "outputs": ["op2"]
        }
        
    def _create_attention_test_graph(self) -> Dict:
        """Create a test graph with attention mechanism"""
        return {
            "ops": [
                {
                    "id": "op1",
                    "type": "tt.matmul",
                    "a_id": "query",
                    "b_id": "key",
                    "a_shape": [16, 512, 64],
                    "b_shape": [16, 64, 512],
                    "output_shape": [16, 512, 512]
                },
                {
                    "id": "op2",
                    "type": "tt.binary.div",
                    "lhs_id": "op1",
                    "rhs_id": "scale",
                    "lhs_shape": [16, 512, 512],
                    "rhs_shape": [1],
                    "output_shape": [16, 512, 512]
                },
                {
                    "id": "op3",
                    "type": "tt.softmax",
                    "operand_id": "op2",
                    "operand_shape": [16, 512, 512],
                    "output_shape": [16, 512, 512],
                    "dim": -1
                },
                {
                    "id": "op4",
                    "type": "tt.matmul",
                    "a_id": "op3",
                    "b_id": "value",
                    "a_shape": [16, 512, 512],
                    "b_shape": [16, 512, 64],
                    "output_shape": [16, 512, 64]
                }
            ],
            "inputs": ["query", "key", "value", "scale"],
            "outputs": ["op4"]
        }
        
    def _create_swiglu_test_graph(self) -> Dict:
        """Create a test graph with SwiGLU activation"""
        return {
            "ops": [
                {
                    "id": "op1",
                    "type": "tt.binary.mul",
                    "lhs_id": "input1",
                    "rhs_id": "input2",
                    "lhs_shape": [64, 128],
                    "rhs_shape": [64, 128],
                    "output_shape": [64, 128]
                },
                {
                    "id": "op2",
                    "type": "tt.unary.sigmoid",
                    "operand_id": "input3",
                    "operand_shape": [64, 128],
                    "output_shape": [64, 128]
                },
                {
                    "id": "op3",
                    "type": "tt.binary.mul",
                    "lhs_id": "op1",
                    "rhs_id": "op2",
                    "lhs_shape": [64, 128],
                    "rhs_shape": [64, 128],
                    "output_shape": [64, 128]
                }
            ],
            "inputs": ["input1", "input2", "input3"],
            "outputs": ["op3"]
        }
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        # Check that passes were created
        self.assertGreater(len(self.optimizer.passes), 0)
        
        # Check that passes are properly initialized
        for optimization_pass in self.optimizer.passes:
            self.assertIsInstance(optimization_pass, OptimizationPass)
            
        # Print detected passes
        print(f"\nOptimization passes: {[p.name for p in self.optimizer.passes]}")
        
    def test_operation_fusion_pass(self):
        """Test the operation fusion pass"""
        # Create a dedicated fusion pass
        fusion_pass = OperationFusionPass()
        
        # Apply to matmul graph
        optimized_graph, stats = fusion_pass.apply(self.matmul_graph)
        
        # Should fuse matmul+add into FMA on any hardware
        self.assertIn("fused_ops", stats)
        
        print(f"\nFusion pass stats for matmul graph: {stats}")
    
    def test_memory_access_optimization_pass(self):
        """Test memory access optimization pass"""
        # Create a dedicated memory access optimization pass
        memory_pass = MemoryAccessOptimizationPass()
        
        # Apply to matmul graph
        optimized_graph, stats = memory_pass.apply(self.matmul_graph)
        
        # Should perform memory optimizations
        self.assertIn("memory_opts", stats)
        
        # Check that layout hints were added
        op = optimized_graph["ops"][0]
        self.assertIn("layout_hints", op)
        
        # Print the memory optimization stats
        print(f"\nMemory optimization stats: {stats}")
        
        # Check hardware-specific optimizations
        if self.chip_generation == AppleSiliconGeneration.M3:
            # Check M3-specific memory optimizations
            self.assertEqual(op["layout_hints"]["block_size"], 128)
            self.assertTrue(op["layout_hints"]["use_tensor_cores"])
        
    def test_metal_specific_optimization_pass(self):
        """Test Metal-specific optimization pass"""
        # Create a dedicated Metal optimization pass
        metal_pass = MetalSpecificOptimizationPass()
        
        # Apply to matmul graph
        optimized_graph, stats = metal_pass.apply(self.matmul_graph)
        
        # Should perform Metal-specific optimizations
        self.assertIn("metal_opts", stats)
        
        # Check that Metal hints were added
        self.assertIn("metadata", optimized_graph)
        self.assertTrue(optimized_graph["metadata"]["metal_optimized"])
        
        # Check hardware-specific optimizations
        if self.chip_generation == AppleSiliconGeneration.M3:
            # Check M3-specific Metal optimizations
            self.assertTrue(optimized_graph["metadata"]["use_simdgroup_matrix"])
            self.assertTrue(optimized_graph["metadata"]["use_tensor_cores"])
            
            # Check operation-specific optimizations
            op = optimized_graph["ops"][0]
            self.assertIn("metal_hints", op)
            self.assertEqual(op["metal_hints"]["preferred_workgroup_size"], (8, 8, 1))
            
        # Print the Metal optimization stats
        print(f"\nMetal optimization stats: {stats}")
    
    def test_m3_specific_fusion_pass(self):
        """Test M3-specific fusion pass"""
        # Create a dedicated M3 fusion pass
        m3_pass = M3SpecificFusionPass()
        
        # Check if this pass is applicable on current hardware
        is_applicable = m3_pass.is_applicable()
        print(f"\nM3-specific fusion pass applicable: {is_applicable}")
        
        # Skip detailed tests if not on M3
        if not is_applicable:
            self.skipTest("M3-specific optimizations only applicable on M3 hardware")
            
        # Apply to attention graph
        optimized_graph, stats = m3_pass.apply(self.attention_graph)
        
        # Should perform M3-specific fusions
        self.assertIn("m3_fused_ops", stats)
        
        # Print the M3 fusion stats
        print(f"\nM3-specific fusion stats: {stats}")
        
        # Check for M3-specific metadata
        self.assertIn("metadata", optimized_graph)
        self.assertTrue(optimized_graph["metadata"]["m3_optimized"])
    
    def test_full_optimization_pipeline(self):
        """Test the full optimization pipeline"""
        # Apply all optimizations to the attention graph
        optimized_graph, stats = self.optimizer.optimize(self.attention_graph)
        
        # Print the full optimization stats
        print(f"\nFull optimization pipeline stats: {stats}")
        
        # Check that optimizations were applied
        if self.chip_generation == AppleSiliconGeneration.M3:
            # On M3, we expect more optimizations
            self.assertGreater(sum(stats.values()), 0, "No optimizations were applied on M3")
            
            # Check for Metal-specific optimizations
            if "metal_opts" in stats:
                self.assertGreater(stats["metal_opts"], 0, "No Metal optimizations applied on M3")
        
    def test_optimization_on_all_graphs(self):
        """Test optimization on all test graphs"""
        test_graphs = {
            "matmul": self.matmul_graph,
            "conv": self.conv_graph,
            "attention": self.attention_graph,
            "swiglu": self.swiglu_graph
        }
        
        results = {}
        
        # Optimize each graph
        for name, graph in test_graphs.items():
            optimized_graph, stats = self.optimizer.optimize(graph)
            results[name] = {
                "original_ops": len(graph["ops"]),
                "optimized_ops": len(optimized_graph["ops"]),
                "stats": stats
            }
        
        # Print results for all graphs
        print("\nOptimization results for all test graphs:")
        for name, result in results.items():
            print(f"  {name}: {result['original_ops']} ops -> {result['optimized_ops']} ops")
            print(f"    Stats: {result['stats']}")
    
    def test_global_optimize_function(self):
        """Test the global optimize function"""
        # Apply global optimize function to matmul graph
        optimized_graph, stats = optimize(self.matmul_graph)
        
        # Check that optimizations were applied
        self.assertIsInstance(optimized_graph, dict)
        self.assertIsInstance(stats, dict)
        
        # Print the optimization stats
        print(f"\nGlobal optimize function stats: {stats}")

def main():
    """Run tests"""
    # Print some information
    print("Testing MLX Graph Optimizer")
    print(f"MLX Available: {MLX_AVAILABLE}")
    print(f"Hardware: {hardware_capabilities.chip_generation.name if hasattr(hardware_capabilities, 'chip_generation') else 'Unknown'}")
    
    # Run tests
    unittest.main()

if __name__ == "__main__":
    main() 