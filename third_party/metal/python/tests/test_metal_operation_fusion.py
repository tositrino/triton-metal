"""
Tests for Metal-specific Operation Fusion Optimizer
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
from MLX.metal_operation_fusion import (
    MetalOperationFusionOptimizer,
    ElementwiseFusionPattern,
    MatMulAddFusionPattern,
    SoftmaxFusionPattern,
    FlashAttentionFusionPattern,
    GeluFusionPattern,
    SwiGLUFusionPattern,
    optimize_operation_fusion,
    get_metal_fusion_optimizer
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

# Create mock hardware capabilities
class MockHardwareCapabilities:
    """Mock hardware capabilities for testing"""
    
    def __init__(self, chip_generation):
        """Initialize with specified chip generation"""
        self.chip_generation = chip_generation

class TestFusionPatterns(unittest.TestCase):
    """Test individual fusion patterns"""
    
    def test_elementwise_fusion_pattern(self):
        """Test elementwise fusion pattern"""
        # Create pattern
        pattern = ElementwiseFusionPattern()
        
        # Create operations
        ops = [
            {"type": "tt.add", "id": "op1", "input_shapes": [[128, 128], [128, 128]]},
            {"type": "tt.mul", "id": "op2", "input_shapes": [[128, 128], [128, 128]]}
        ]
        
        # Check if pattern matches
        self.assertTrue(pattern.matches(ops, 0))
        
        # Check with non-matching operations
        non_matching_ops = [
            {"type": "tt.add", "id": "op1", "input_shapes": [[128, 128], [128, 128]]},
            {"type": "tt.matmul", "id": "op2", "input_shapes": [[128, 128], [128, 128]]}
        ]
        self.assertFalse(pattern.matches(non_matching_ops, 0))
        
        # Test fused operation creation
        fused_op = pattern.get_fused_op(ops, 0)
        self.assertEqual(fused_op["type"], "fusion.elementwise_fusion")
        self.assertEqual(len(fused_op["fused_ops"]), 2)
        self.assertEqual(fused_op["fusion_strategy"], "elementwise")
        self.assertTrue(fused_op["vectorize"])
    
    def test_matmul_add_fusion_pattern(self):
        """Test matmul-add fusion pattern"""
        # Create pattern
        pattern = MatMulAddFusionPattern()
        
        # Create operations
        ops = [
            {"type": "tt.matmul", "id": "op1", "input_shapes": [[128, 256], [256, 512]]},
            {"type": "tt.add", "id": "op2", "input_shapes": [[128, 512], [128, 512]]}
        ]
        
        # Check if pattern matches
        self.assertTrue(pattern.matches(ops, 0))
        
        # Test fused operation creation
        fused_op = pattern.get_fused_op(ops, 0)
        self.assertEqual(fused_op["type"], "fusion.matmul_add_fusion")
        self.assertEqual(len(fused_op["fused_ops"]), 2)
        self.assertEqual(fused_op["fusion_strategy"], "matmul_add")
        self.assertEqual(fused_op["matmul_dims"]["m"], 128)
        self.assertEqual(fused_op["matmul_dims"]["n"], 512)
        self.assertEqual(fused_op["matmul_dims"]["k"], 256)
    
    def test_softmax_fusion_pattern(self):
        """Test softmax fusion pattern"""
        # Create pattern
        pattern = SoftmaxFusionPattern()
        
        # Create operations
        ops = [
            {"type": "tt.sub", "id": "op1", "input_shapes": [[128, 128], [1, 128]]},
            {"type": "tt.exp", "id": "op2", "input_shapes": [[128, 128]]},
            {"type": "tt.sum", "id": "op3", "input_shapes": [[128, 128]]},
            {"type": "tt.div", "id": "op4", "input_shapes": [[128, 128], [128, 1]]}
        ]
        
        # Check if pattern matches
        self.assertTrue(pattern.matches(ops, 0))
        
        # Test fused operation creation
        fused_op = pattern.get_fused_op(ops, 0)
        self.assertEqual(fused_op["type"], "fusion.softmax_fusion")
        self.assertEqual(len(fused_op["fused_ops"]), 4)
        self.assertEqual(fused_op["fusion_strategy"], "softmax")
        self.assertTrue(fused_op["use_fast_math"])
    
    def test_gelu_fusion_pattern(self):
        """Test GELU fusion pattern"""
        # Create pattern
        pattern = GeluFusionPattern()
        
        # Create operations for GELU sequence
        ops = [
            {"type": "tt.mul", "id": "op1"},
            {"type": "tt.pow", "id": "op2"},
            {"type": "tt.mul", "id": "op3"},
            {"type": "tt.add", "id": "op4"},
            {"type": "tt.mul", "id": "op5"},
            {"type": "tt.tanh", "id": "op6"},
            {"type": "tt.add", "id": "op7"},
            {"type": "tt.mul", "id": "op8"}
        ]
        
        # Check if pattern matches
        self.assertTrue(pattern.matches(ops, 0))
        
        # Test fused operation creation
        fused_op = pattern.get_fused_op(ops, 0)
        self.assertEqual(fused_op["type"], "fusion.gelu_fusion")
        self.assertEqual(len(fused_op["fused_ops"]), 8)
        self.assertEqual(fused_op["fusion_strategy"], "gelu")
        self.assertTrue(fused_op["vectorize"])
        self.assertTrue(fused_op["use_fast_math"])
        self.assertTrue(fused_op["use_approximation"])
    
    def test_flash_attention_pattern(self):
        """Test flash attention pattern"""
        # Create pattern
        pattern = FlashAttentionFusionPattern()
        
        # Create operations
        ops = [
            {"type": "tt.matmul", "id": "op1"},
            {"type": "tt.div", "id": "op2"},
            {"type": "tt.softmax", "id": "op3"},
            {"type": "tt.matmul", "id": "op4"}
        ]
        
        # Check if pattern matches
        self.assertTrue(pattern.matches(ops, 0))
        
        # Test fused operation creation
        fused_op = pattern.get_fused_op(ops, 0)
        self.assertEqual(fused_op["type"], "fusion.flash_attention_fusion")
        self.assertEqual(len(fused_op["fused_ops"]), 4)
        self.assertEqual(fused_op["fusion_strategy"], "flash_attention")
        self.assertTrue(fused_op["use_tensor_cores"])
        self.assertTrue(fused_op["use_hierarchical_softmax"])
        self.assertEqual(fused_op["implementation"], "flash_attention_v2")
    
    def test_swiglu_pattern(self):
        """Test SwiGLU pattern"""
        # Create pattern
        pattern = SwiGLUFusionPattern()
        
        # Create operations
        ops = [
            {"type": "tt.mul", "id": "op1"},
            {"type": "tt.sigmoid", "id": "op2"},
            {"type": "tt.mul", "id": "op3"}
        ]
        
        # Check if pattern matches
        self.assertTrue(pattern.matches(ops, 0))
        
        # Test fused operation creation
        fused_op = pattern.get_fused_op(ops, 0)
        self.assertEqual(fused_op["type"], "fusion.swiglu_fusion")
        self.assertEqual(len(fused_op["fused_ops"]), 3)
        self.assertEqual(fused_op["fusion_strategy"], "swiglu")
        self.assertTrue(fused_op["vectorize"])
        self.assertTrue(fused_op["use_fast_sigmoid"])
        self.assertTrue(fused_op["use_tensor_cores"])
        self.assertEqual(fused_op["implementation"], "native_swiglu")

class TestOptimizerWithM3(unittest.TestCase):
    """Test optimizer with M3 hardware"""
    
    def setUp(self):
        """Set up test cases"""
        # Mock M3 hardware
        global hardware_capabilities
        hardware_capabilities = MockHardwareCapabilities(AppleSiliconGeneration.M3)
        
        # Create a sample graph with fusible operations
        self.sample_graph = {
            "ops": [
                # MatMul + Add
                {"type": "tt.matmul", "id": "op1", "input_shapes": [[128, 256], [256, 512]]},
                {"type": "tt.add", "id": "op2", "input_shapes": [[128, 512], [1, 512]]},
                
                # Elementwise ops
                {"type": "tt.add", "id": "op3", "input_shapes": [[128, 512], [128, 512]]},
                {"type": "tt.mul", "id": "op4", "input_shapes": [[128, 512], [128, 512]]},
                
                # Softmax sequence
                {"type": "tt.sub", "id": "op5", "input_shapes": [[128, 128], [1, 128]]},
                {"type": "tt.exp", "id": "op6", "input_shapes": [[128, 128]]},
                {"type": "tt.sum", "id": "op7", "input_shapes": [[128, 128]]},
                {"type": "tt.div", "id": "op8", "input_shapes": [[128, 128], [128, 1]]},
                
                # Flash attention sequence
                {"type": "tt.matmul", "id": "op9", "input_shapes": [[128, 128], [128, 128]]},
                {"type": "tt.div", "id": "op10", "input_shapes": [[128, 128], [1, 1]]},
                {"type": "tt.softmax", "id": "op11", "input_shapes": [[128, 128]]},
                {"type": "tt.matmul", "id": "op12", "input_shapes": [[128, 128], [128, 64]]}
            ]
        }
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization with M3 hardware"""
        # Create optimizer
        optimizer = MetalOperationFusionOptimizer()
        
        # Check hardware detection
        self.assertEqual(optimizer.hardware_gen, AppleSiliconGeneration.M3)
        
        # Check that patterns are created
        self.assertGreaterEqual(len(optimizer.patterns), 6)  # All patterns including M3-specific ones
        
        # Check stats initialization
        self.assertEqual(optimizer.stats["fused_ops"], 0)
        self.assertEqual(len(optimizer.stats["fusion_patterns"]), 0)
    
    def test_optimize_graph(self):
        """Test optimization of entire graph with M3 hardware"""
        # Create optimizer
        optimizer = MetalOperationFusionOptimizer()
        
        # Optimize graph
        optimized_graph, stats = optimizer.optimize(self.sample_graph)
        
        # Check that operations were fused
        self.assertGreater(stats["fused_ops"], 0)
        
        # Check that operations count decreased due to fusion
        self.assertLess(len(optimized_graph["ops"]), len(self.sample_graph["ops"]))
        
        # Check metadata
        self.assertIn("metadata", optimized_graph)
        self.assertTrue(optimized_graph["metadata"]["operation_fusion_optimized"])
        self.assertTrue(optimized_graph["metadata"]["m3_fusion_optimized"])
        
        # Check that M3-specific optimizations were applied
        self.assertGreater(stats["hardware_specific_fusions"], 0)
    
    def test_m3_specific_patterns(self):
        """Test M3-specific patterns are applied"""
        # Create optimizer
        optimizer = MetalOperationFusionOptimizer()
        
        # Create a sample graph with only M3-specific patterns
        flash_attention_graph = {
            "ops": [
                {"type": "tt.matmul", "id": "op1"},
                {"type": "tt.div", "id": "op2"},
                {"type": "tt.softmax", "id": "op3"},
                {"type": "tt.matmul", "id": "op4"}
            ]
        }
        
        # Optimize graph
        optimized_graph, stats = optimizer.optimize(flash_attention_graph)
        
        # Check that flash attention pattern was applied
        self.assertEqual(len(optimized_graph["ops"]), 1)
        self.assertEqual(optimized_graph["ops"][0]["type"], "fusion.flash_attention_fusion")
        self.assertEqual(stats["hardware_specific_fusions"], 1)

class TestOptimizerWithM1(unittest.TestCase):
    """Test optimizer with M1 hardware"""
    
    def setUp(self):
        """Set up test cases"""
        # Mock M1 hardware
        global hardware_capabilities
        hardware_capabilities = MockHardwareCapabilities(AppleSiliconGeneration.M1)
        
        # Create a sample graph with fusible operations
        self.sample_graph = {
            "ops": [
                # MatMul + Add
                {"type": "tt.matmul", "id": "op1", "input_shapes": [[128, 256], [256, 512]]},
                {"type": "tt.add", "id": "op2", "input_shapes": [[128, 512], [1, 512]]},
                
                # Elementwise ops
                {"type": "tt.add", "id": "op3", "input_shapes": [[128, 512], [128, 512]]},
                {"type": "tt.mul", "id": "op4", "input_shapes": [[128, 512], [128, 512]]},
                
                # Softmax sequence
                {"type": "tt.sub", "id": "op5", "input_shapes": [[128, 128], [1, 128]]},
                {"type": "tt.exp", "id": "op6", "input_shapes": [[128, 128]]},
                {"type": "tt.sum", "id": "op7", "input_shapes": [[128, 128]]},
                {"type": "tt.div", "id": "op8", "input_shapes": [[128, 128], [128, 1]]},
                
                # Flash attention sequence (would only be fused on M3)
                {"type": "tt.matmul", "id": "op9", "input_shapes": [[128, 128], [128, 128]]},
                {"type": "tt.div", "id": "op10", "input_shapes": [[128, 128], [1, 1]]},
                {"type": "tt.softmax", "id": "op11", "input_shapes": [[128, 128]]},
                {"type": "tt.matmul", "id": "op12", "input_shapes": [[128, 128], [128, 64]]}
            ]
        }
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization with M1 hardware"""
        # Create optimizer
        optimizer = MetalOperationFusionOptimizer()
        
        # Check hardware detection
        self.assertEqual(optimizer.hardware_gen, AppleSiliconGeneration.M1)
        
        # Check that patterns are created (should not include M3-specific patterns)
        patterns_with_m3_requirement = [p for p in optimizer.patterns if p.min_hardware_gen == AppleSiliconGeneration.M3]
        for pattern in patterns_with_m3_requirement:
            self.assertFalse(pattern.is_applicable_to_hardware(optimizer.hardware_gen))
    
    def test_optimize_graph(self):
        """Test optimization of entire graph with M1 hardware"""
        # Create optimizer
        optimizer = MetalOperationFusionOptimizer()
        
        # Optimize graph
        optimized_graph, stats = optimizer.optimize(self.sample_graph)
        
        # Check that operations were fused
        self.assertGreater(stats["fused_ops"], 0)
        
        # Check that operations count decreased due to fusion
        self.assertLess(len(optimized_graph["ops"]), len(self.sample_graph["ops"]))
        
        # Check metadata
        self.assertIn("metadata", optimized_graph)
        self.assertTrue(optimized_graph["metadata"]["operation_fusion_optimized"])
        self.assertNotIn("m3_fusion_optimized", optimized_graph["metadata"])
        
        # Check that no M3-specific optimizations were applied
        self.assertEqual(stats["hardware_specific_fusions"], 0)
    
    def test_m3_specific_patterns_not_applied(self):
        """Test M3-specific patterns are not applied on M1"""
        # Create optimizer
        optimizer = MetalOperationFusionOptimizer()
        
        # Create a sample graph with only M3-specific patterns
        flash_attention_graph = {
            "ops": [
                {"type": "tt.matmul", "id": "op1"},
                {"type": "tt.div", "id": "op2"},
                {"type": "tt.softmax", "id": "op3"},
                {"type": "tt.matmul", "id": "op4"}
            ]
        }
        
        # Optimize graph
        optimized_graph, stats = optimizer.optimize(flash_attention_graph)
        
        # Check that flash attention pattern was not applied (M3-specific)
        self.assertEqual(len(optimized_graph["ops"]), 4)  # No fusion applied
        self.assertEqual(stats["hardware_specific_fusions"], 0)

class TestSingletonAccessor(unittest.TestCase):
    """Test singleton accessor pattern"""
    
    def test_singleton_accessor(self):
        """Test singleton accessor pattern"""
        # Get optimizer instances
        optimizer1 = get_metal_fusion_optimizer()
        optimizer2 = get_metal_fusion_optimizer()
        
        # Check that they are the same instance
        self.assertIs(optimizer1, optimizer2)
        
        # Use the global optimize function
        optimized_graph, stats = optimize_operation_fusion({"ops": []})
        
        # Check that optimization was called
        self.assertEqual(len(stats["fusion_patterns"]), 0)

if __name__ == "__main__":
    unittest.main() 