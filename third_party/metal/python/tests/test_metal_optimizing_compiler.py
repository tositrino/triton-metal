"""
Tests for Metal-specific Graph Optimizing Compiler
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
from MLX.metal_optimizing_compiler import (
    MetalOptimizingCompiler,
    OptimizationLevel,
    get_metal_optimizing_compiler,
    optimize_for_metal
)

# Mock hardware capabilities if needed
try:
    from metal_hardware_optimizer import AppleSiliconGeneration
except ImportError:
    # Create mock enum
    class DummyEnum(Enum):
        UNKNOWN = 0
        M1 = 1
        M2 = 2
        M3 = 3
    
    AppleSiliconGeneration = DummyEnum

# Create mock hardware capabilities
class MockHardwareCapabilities:
    """Mock hardware capabilities for testing"""
    
    def __init__(self, chip_generation):
        """Initialize with specified chip generation"""
        self.chip_generation = chip_generation

# Mock optimization functions to test pipeline
class MockOptimizationFunctions:
    """Mock optimization functions for testing"""
    
    @staticmethod
    def mock_memory_layout(graph, level=None):
        """Mock memory layout optimization"""
        # Add tracking metadata to verify this pass was called
        if "metadata" not in graph:
            graph["metadata"] = {}
        graph["metadata"]["memory_layout_optimized"] = True
        graph["metadata"]["memory_layout_level"] = str(level) if level else "None"
        
        # Return mock stats
        stats = {
            "optimized_ops": 10,
            "memory_layout_changes": 5
        }
        
        return graph, stats
    
    @staticmethod
    def mock_operation_fusion(graph):
        """Mock operation fusion optimization"""
        # Add tracking metadata to verify this pass was called
        if "metadata" not in graph:
            graph["metadata"] = {}
        graph["metadata"]["operation_fusion_optimized"] = True
        
        # Return mock stats
        stats = {
            "fused_ops": 8,
            "fusion_patterns": {"elementwise": 3, "matmul_add": 2},
            "hardware_specific_fusions": 2
        }
        
        return graph, stats
    
    @staticmethod
    def mock_mlx_graph(graph, hardware_gen=None, aggressive=False):
        """Mock MLX graph optimization"""
        # Add tracking metadata to verify this pass was called
        if "metadata" not in graph:
            graph["metadata"] = {}
        graph["metadata"]["mlx_graph_optimized"] = True
        graph["metadata"]["mlx_aggressive"] = aggressive
        
        # Return mock stats
        stats = {
            "optimized_ops": 15 if aggressive else 10,
            "optimizations": {"constant_folding": 5, "dead_code_elimination": 3}
        }
        
        return graph, stats
    
    @staticmethod
    def mock_m3_optimization(graph):
        """Mock M3-specific optimization"""
        # Add tracking metadata to verify this pass was called
        if "metadata" not in graph:
            graph["metadata"] = {}
        graph["metadata"]["m3_optimized"] = True
        
        # Return mock stats (empty for this mock)
        stats = {}
        
        return graph, stats

class TestCompilerInitialization(unittest.TestCase):
    """Test compiler initialization with different optimization levels"""
    
    def test_none_level(self):
        """Test initialization with NONE optimization level"""
        compiler = MetalOptimizingCompiler(OptimizationLevel.NONE)
        
        # Should have empty pipeline
        self.assertEqual(len(compiler.optimization_pipeline), 0)
    
    def test_basic_level(self):
        """Test initialization with BASIC optimization level"""
        compiler = MetalOptimizingCompiler(OptimizationLevel.BASIC)
        
        # Should have basic passes
        pass_names = [name for name, _ in compiler.optimization_pipeline]
        self.assertIn("mlx_graph_basic", pass_names)
        self.assertIn("memory_layout_basic", pass_names)
        self.assertEqual(len(compiler.optimization_pipeline), 2)
    
    def test_standard_level(self):
        """Test initialization with STANDARD optimization level"""
        compiler = MetalOptimizingCompiler(OptimizationLevel.STANDARD)
        
        # Should have standard passes
        pass_names = [name for name, _ in compiler.optimization_pipeline]
        self.assertIn("mlx_graph_basic", pass_names)
        self.assertIn("memory_layout_standard", pass_names)  # Upgraded from basic
        self.assertIn("operation_fusion", pass_names)
        # M3 specific pass may or may not be present depending on hardware
    
    def test_aggressive_level(self):
        """Test initialization with AGGRESSIVE optimization level"""
        compiler = MetalOptimizingCompiler(OptimizationLevel.AGGRESSIVE)
        
        # Should have aggressive passes
        pass_names = [name for name, _ in compiler.optimization_pipeline]
        self.assertIn("mlx_graph_basic", pass_names)
        self.assertIn("memory_layout_standard", pass_names)
        self.assertIn("operation_fusion", pass_names)
        self.assertIn("mlx_graph_aggressive", pass_names)

class TestCompilerWithM3(unittest.TestCase):
    """Test compiler with M3 hardware"""
    
    def setUp(self):
        """Set up test cases"""
        # Mock M3 hardware
        global hardware_capabilities
        hardware_capabilities = MockHardwareCapabilities(AppleSiliconGeneration.M3)
        
        # Create a sample graph for testing
        self.sample_graph = {
            "ops": [
                {"type": "tt.matmul", "id": "op1"},
                {"type": "tt.add", "id": "op2"},
                {"type": "tt.relu", "id": "op3"}
            ]
        }
        
        # Replace optimization functions with mocks
        self._original_optimize_memory_layout = None
        self._original_optimize_operation_fusion = None
        self._original_optimize_mlx_graph = None
        self._setup_mock_functions()
        
    def _setup_mock_functions(self):
        """Replace real optimization functions with mocks"""
        import metal_optimizing_compiler as compiler_module
        
        # Save original functions
        self._original_optimize_memory_layout = compiler_module.optimize_memory_layout
        self._original_optimize_operation_fusion = compiler_module.optimize_operation_fusion
        self._original_optimize_mlx_graph = compiler_module.optimize_mlx_graph
        
        # Replace with mocks
        compiler_module.optimize_memory_layout = MockOptimizationFunctions.mock_memory_layout
        compiler_module.optimize_operation_fusion = MockOptimizationFunctions.mock_operation_fusion
        compiler_module.optimize_mlx_graph = MockOptimizationFunctions.mock_mlx_graph
        
        # Mock M3 optimization
        compiler_module.has_m3_optimizations = True
        compiler_module.get_m3_memory_manager = lambda: type('obj', (object,), {
            'optimize_graph_memory': MockOptimizationFunctions.mock_m3_optimization
        })
    
    def tearDown(self):
        """Restore original functions after tests"""
        if self._original_optimize_memory_layout:
            import metal_optimizing_compiler as compiler_module
            compiler_module.optimize_memory_layout = self._original_optimize_memory_layout
            compiler_module.optimize_operation_fusion = self._original_optimize_operation_fusion
            compiler_module.optimize_mlx_graph = self._original_optimize_mlx_graph
    
    def test_m3_standard_optimization(self):
        """Test standard optimization with M3 hardware"""
        # Create compiler with standard optimization level
        compiler = MetalOptimizingCompiler(OptimizationLevel.STANDARD)
        
        # Check M3 optimization is enabled
        self.assertTrue(compiler.use_m3_optimizations)
        
        # Check pipeline includes M3-specific pass
        pass_names = [name for name, _ in compiler.optimization_pipeline]
        self.assertIn("m3_memory_optimization", pass_names)
        
        # Test compilation
        optimized_graph, stats = compiler.compile(self.sample_graph)
        
        # Check all passes were applied
        self.assertIn("memory_layout_optimized", optimized_graph["metadata"])
        self.assertIn("operation_fusion_optimized", optimized_graph["metadata"])
        self.assertIn("mlx_graph_optimized", optimized_graph["metadata"])
        self.assertIn("m3_optimized", optimized_graph["metadata"])
        
        # Check stats were collected
        self.assertGreater(stats["compile_time"], 0)
        self.assertEqual(len(stats["passes"]), 4)  # 4 passes should have run
    
    def test_optimization_summary(self):
        """Test getting optimization summary"""
        # Create compiler with aggressive optimization level
        compiler = MetalOptimizingCompiler(OptimizationLevel.AGGRESSIVE)
        
        # Compile graph
        _, _ = compiler.compile(self.sample_graph)
        
        # Get summary
        summary = compiler.get_optimization_summary()
        
        # Check summary contains expected information
        self.assertEqual(summary["optimization_level"], "AGGRESSIVE")
        self.assertEqual(summary["hardware_generation"], "M3")
        self.assertGreater(summary["compile_time"], 0)
        self.assertEqual(len(summary["passes_applied"]), 5)  # 5 passes with aggressive level
        
        # Check pass statistics
        self.assertIn("pass_statistics", summary)
        for name in summary["passes_applied"]:
            self.assertIn(name, summary["pass_statistics"])
            self.assertEqual(summary["pass_statistics"][name]["status"], "success")

class TestCompilerWithM1(unittest.TestCase):
    """Test compiler with M1 hardware"""
    
    def setUp(self):
        """Set up test cases"""
        # Mock M1 hardware
        global hardware_capabilities
        hardware_capabilities = MockHardwareCapabilities(AppleSiliconGeneration.M1)
        
        # Create a sample graph for testing
        self.sample_graph = {
            "ops": [
                {"type": "tt.matmul", "id": "op1"},
                {"type": "tt.add", "id": "op2"},
                {"type": "tt.relu", "id": "op3"}
            ]
        }
        
        # Replace optimization functions with mocks
        self._original_optimize_memory_layout = None
        self._original_optimize_operation_fusion = None
        self._original_optimize_mlx_graph = None
        self._setup_mock_functions()
        
    def _setup_mock_functions(self):
        """Replace real optimization functions with mocks"""
        import metal_optimizing_compiler as compiler_module
        
        # Save original functions
        self._original_optimize_memory_layout = compiler_module.optimize_memory_layout
        self._original_optimize_operation_fusion = compiler_module.optimize_operation_fusion
        self._original_optimize_mlx_graph = compiler_module.optimize_mlx_graph
        
        # Replace with mocks
        compiler_module.optimize_memory_layout = MockOptimizationFunctions.mock_memory_layout
        compiler_module.optimize_operation_fusion = MockOptimizationFunctions.mock_operation_fusion
        compiler_module.optimize_mlx_graph = MockOptimizationFunctions.mock_mlx_graph
        
        # Mock M3 optimization availability
        compiler_module.has_m3_optimizations = True
    
    def tearDown(self):
        """Restore original functions after tests"""
        if self._original_optimize_memory_layout:
            import metal_optimizing_compiler as compiler_module
            compiler_module.optimize_memory_layout = self._original_optimize_memory_layout
            compiler_module.optimize_operation_fusion = self._original_optimize_operation_fusion
            compiler_module.optimize_mlx_graph = self._original_optimize_mlx_graph
    
    def test_m1_standard_optimization(self):
        """Test standard optimization with M1 hardware"""
        # Create compiler with standard optimization level
        compiler = MetalOptimizingCompiler(OptimizationLevel.STANDARD)
        
        # Check M3 optimization is disabled on M1
        self.assertFalse(compiler.use_m3_optimizations)
        
        # Check pipeline does not include M3-specific pass
        pass_names = [name for name, _ in compiler.optimization_pipeline]
        self.assertNotIn("m3_memory_optimization", pass_names)
        
        # Test compilation
        optimized_graph, stats = compiler.compile(self.sample_graph)
        
        # Check standard passes were applied but not M3-specific ones
        self.assertIn("memory_layout_optimized", optimized_graph["metadata"])
        self.assertIn("operation_fusion_optimized", optimized_graph["metadata"])
        self.assertIn("mlx_graph_optimized", optimized_graph["metadata"])
        self.assertNotIn("m3_optimized", optimized_graph["metadata"])
        
        # Check stats were collected
        self.assertGreater(stats["compile_time"], 0)
        self.assertEqual(len(stats["passes"]), 3)  # 3 passes should have run (no M3-specific pass)

class TestSingletonAccessor(unittest.TestCase):
    """Test singleton accessor pattern"""
    
    def setUp(self):
        """Reset singleton before each test"""
        import metal_optimizing_compiler as compiler_module
        compiler_module._metal_optimizing_compiler = None
    
    def test_singleton_accessor(self):
        """Test singleton accessor creates a single instance"""
        # Get compiler instances with same optimization level
        compiler1 = get_metal_optimizing_compiler(OptimizationLevel.STANDARD)
        compiler2 = get_metal_optimizing_compiler(OptimizationLevel.STANDARD)
        
        # Should be the same instance
        self.assertIs(compiler1, compiler2)
    
    def test_singleton_different_levels(self):
        """Test singleton is recreated with different optimization levels"""
        # Get compiler with standard level
        compiler1 = get_metal_optimizing_compiler(OptimizationLevel.STANDARD)
        
        # Get compiler with different level
        compiler2 = get_metal_optimizing_compiler(OptimizationLevel.AGGRESSIVE)
        
        # Should be different instances
        self.assertIsNot(compiler1, compiler2)
        
        # Different optimization levels
        self.assertEqual(compiler1.optimization_level, OptimizationLevel.STANDARD)
        self.assertEqual(compiler2.optimization_level, OptimizationLevel.AGGRESSIVE)
    
    def test_helper_function(self):
        """Test the optimize_for_metal helper function"""
        # Replace optimization functions with mocks
        import metal_optimizing_compiler as compiler_module
        original_optimize_memory_layout = compiler_module.optimize_memory_layout
        original_optimize_operation_fusion = compiler_module.optimize_operation_fusion
        original_optimize_mlx_graph = compiler_module.optimize_mlx_graph
        
        try:
            # Replace with mocks
            compiler_module.optimize_memory_layout = MockOptimizationFunctions.mock_memory_layout
            compiler_module.optimize_operation_fusion = MockOptimizationFunctions.mock_operation_fusion
            compiler_module.optimize_mlx_graph = MockOptimizationFunctions.mock_mlx_graph
            
            # Create a sample graph
            sample_graph = {
                "ops": [
                    {"type": "tt.matmul", "id": "op1"},
                    {"type": "tt.add", "id": "op2"}
                ]
            }
            
            # Use helper function
            optimized_graph, stats = optimize_for_metal(sample_graph, OptimizationLevel.STANDARD)
            
            # Check optimizations were applied
            self.assertIn("memory_layout_optimized", optimized_graph["metadata"])
            self.assertIn("operation_fusion_optimized", optimized_graph["metadata"])
            self.assertIn("mlx_graph_optimized", optimized_graph["metadata"])
            
            # Check stats were collected
            self.assertGreater(stats["compile_time"], 0)
            self.assertGreaterEqual(len(stats["passes"]), 3)  # At least 3 passes
            
            # Check that the helper uses the singleton
            compiler = get_metal_optimizing_compiler(OptimizationLevel.STANDARD)
            self.assertEqual(compiler.stats["compile_time"], stats["compile_time"])
            
        finally:
            # Restore original functions
            compiler_module.optimize_memory_layout = original_optimize_memory_layout
            compiler_module.optimize_operation_fusion = original_optimize_operation_fusion
            compiler_module.optimize_mlx_graph = original_optimize_mlx_graph

if __name__ == "__main__":
    unittest.main() 