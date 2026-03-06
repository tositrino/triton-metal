"""
MLX Graph Optimizer for Triton Metal Backend

This module provides optimization passes for MLX computation graphs,
with specific optimizations for Apple Silicon, particularly M3 chips.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from enum import Enum



# Import hardware detection safely to handle cases where it might fail
try:
    from MLX.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
except ImportError:
    # Define dummy hardware capabilities for testing
    class DummyEnum(Enum):
        UNKNOWN = 0
        M1 = 1
        M2 = 2
        M3 = 3

    class DummyCapabilities:
        def __init__(self):
            self.chip_generation = DummyEnum.UNKNOWN

    AppleSiliconGeneration = DummyEnum
    hardware_capabilities = DummyCapabilities()

# Try to import operation mapping and fusion optimizer
try:
    from MLX.operation_mapping import MLXDispatcher, OpCategory
    from MLX.metal_fusion_optimizer import FusionOptimizer, FusionPattern
    has_fusion_optimizer = True
except ImportError:
    has_fusion_optimizer = False
    print("Warning: operation_mapping or metal_fusion_optimizer module not found. Fusion optimizations will be disabled.")

# Import memory manager for memory optimizations
try:
    from  MLX.metal_memory_manager import get_metal_memory_manager, MetalMemoryManager, TensorType
    has_memory_manager = True
except ImportError:
    has_memory_manager = False
    print("Warning: metal_memory_manager module not found. Memory optimizations will be disabled.")

class OptimizationPass:
    """Base class for MLX computation graph optimization passes"""

    def __init__(self, name: str, min_hardware_gen=None):
        """
        Initialize optimization pass

        Args:
            name: Pass name
            min_hardware_gen: Minimum hardware generation required
        """
        self.name = name
        self.min_hardware_gen = min_hardware_gen

    def is_applicable(self) -> bool:
        """
        Check if this optimization pass is applicable to current hardware

        Returns:
            True if applicable, False otherwise
        """
        if self.min_hardware_gen is None:
            return True

        if not hasattr(hardware_capabilities, "chip_generation") or \
           hardware_capabilities.chip_generation == AppleSiliconGeneration.UNKNOWN:
            return False

        return hardware_capabilities.chip_generation.value >= self.min_hardware_gen.value

    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply optimization pass to the computation graph

        Args:
            graph: MLX computation graph representation

        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        raise NotImplementedError("Subclasses must implement this method")

class OperationFusionPass(OptimizationPass):
    """Optimization pass that fuses compatible operations"""

    def __init__(self, fusion_optimizer=None):
        """
        Initialize operation fusion pass

        Args:
            fusion_optimizer: Optional fusion optimizer instance
        """
        super().__init__("operation_fusion")
        if has_fusion_optimizer:
            self.fusion_optimizer = fusion_optimizer or FusionOptimizer(hardware_capabilities)
            self.has_fusion = True
        else:
            self.has_fusion = False

    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply operation fusion optimizations to the computation graph

        Args:
            graph: MLX computation graph representation

        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not graph or "ops" not in graph or not self.has_fusion:
            return graph, {"fused_ops": 0}

        # Convert graph to list of ops for the fusion optimizer
        ops = graph["ops"]

        # Apply fusion optimizations
        optimized_ops = self.fusion_optimizer.optimize(ops)

        # Count the number of fused operations
        fused_count = len(ops) - len(optimized_ops)

        # Update the graph
        optimized_graph = graph.copy()
        optimized_graph["ops"] = optimized_ops

        return optimized_graph, {"fused_ops": fused_count}

class M3SpecificFusionPass(OptimizationPass):
    """M3-specific operation fusion optimizations"""

    def __init__(self):
        """Initialize M3-specific fusion pass"""
        super().__init__("m3_specific_fusion", min_hardware_gen=AppleSiliconGeneration.M3)
        self.m3_patterns = self._create_m3_fusion_patterns()

    def _create_m3_fusion_patterns(self) -> List:
        """
        Create M3-specific fusion patterns

        Returns:
            List of fusion patterns
        """
        if not has_fusion_optimizer:
            return []

        patterns = []

        # Flash Attention for M3 (optimized attention mechanism)
        patterns.append(FusionPattern(
            "flash_attention",
            ["matmul", "div", "softmax", "matmul"],
            lambda ops: ops[0].get("type") == "tt.matmul" and
                        ops[1].get("type") == "tt.binary.div" and
                        ops[2].get("type") == "tt.softmax" and
                        ops[3].get("type") == "tt.matmul",
            min_hardware_gen=AppleSiliconGeneration.M3
        ))

        # Fused SwiGLU with additional optimizations for M3
        patterns.append(FusionPattern(
            "optimized_swiglu",
            ["mul", "sigmoid", "mul"],
            lambda ops: ops[0].get("type") == "tt.binary.mul" and
                        ops[1].get("type") == "tt.unary.sigmoid" and
                        ops[2].get("type") == "tt.binary.mul",
            min_hardware_gen=AppleSiliconGeneration.M3
        ))

        # Multi-head attention fusion specifically for M3
        patterns.append(FusionPattern(
            "multihead_attention",
            ["reshape", "transpose", "matmul", "div", "softmax", "matmul", "transpose", "reshape"],
            lambda ops: ops[0].get("type") == "tt.reshape" and
                        ops[1].get("type") == "tt.transpose" and
                        ops[2].get("type") == "tt.matmul" and
                        ops[3].get("type") == "tt.binary.div" and
                        ops[4].get("type") == "tt.softmax" and
                        ops[5].get("type") == "tt.matmul" and
                        ops[6].get("type") == "tt.transpose" and
                        ops[7].get("type") == "tt.reshape",
            min_hardware_gen=AppleSiliconGeneration.M3
        ))

        # Fused layer normalization for M3
        patterns.append(FusionPattern(
            "fused_layernorm",
            ["sub", "pow", "mean", "add", "sqrt", "div", "mul", "add"],
            lambda ops: ops[0].get("type") == "tt.binary.sub" and
                        ops[1].get("type") == "tt.pow" and
                        ops[2].get("type") == "tt.reduce" and
                        ops[3].get("type") == "tt.binary.add" and
                        ops[4].get("type") == "tt.unary.sqrt" and
                        ops[5].get("type") == "tt.binary.div" and
                        ops[6].get("type") == "tt.binary.mul" and
                        ops[7].get("type") == "tt.binary.add",
            min_hardware_gen=AppleSiliconGeneration.M3
        ))

        # Fused matrix multiply with GELU activation optimized for M3 tensor cores
        patterns.append(FusionPattern(
            "matmul_gelu",
            ["matmul", "mul", "pow", "mul", "add", "mul", "tanh", "add", "mul"],
            lambda ops: ops[0].get("type") == "tt.matmul" and
                       ops[1].get("type") == "tt.binary.mul" and
                       ops[2].get("type") == "tt.pow" and
                       ops[3].get("type") == "tt.binary.mul" and
                       ops[4].get("type") == "tt.binary.add" and
                       ops[5].get("type") == "tt.binary.mul" and
                       ops[6].get("type") == "tt.unary.tanh" and
                       ops[7].get("type") == "tt.binary.add" and
                       ops[8].get("type") == "tt.binary.mul",
            min_hardware_gen=AppleSiliconGeneration.M3
        ))

        # Fused convolution with batch norm for M3
        patterns.append(FusionPattern(
            "conv_batchnorm",
            ["conv2d", "sub", "mul", "add"],
            lambda ops: ops[0].get("type") == "tt.conv2d" and
                       ops[1].get("type") == "tt.binary.sub" and
                       ops[2].get("type") == "tt.binary.mul" and
                       ops[3].get("type") == "tt.binary.add",
            min_hardware_gen=AppleSiliconGeneration.M3
        ))

        # Fused matrix multiply with bias and activation for M3 tensor cores
        patterns.append(FusionPattern(
            "matmul_bias_act",
            ["matmul", "add", "relu"],
            lambda ops: ops[0].get("type") == "tt.matmul" and
                       ops[1].get("type") == "tt.binary.add" and
                       ops[2].get("type") in ["tt.relu", "tt.unary.relu"],
            min_hardware_gen=AppleSiliconGeneration.M3
        ))

        # Fused dot product attention for M3 (Scaled Dot-Product Attention)
        patterns.append(FusionPattern(
            "scaled_dot_product_attention",
            ["matmul", "div", "softmax", "matmul"],
            lambda ops: ops[0].get("type") == "tt.matmul" and
                       ops[1].get("type") == "tt.binary.div" and
                       ops[2].get("type") == "tt.softmax" and
                       ops[3].get("type") == "tt.matmul",
            min_hardware_gen=AppleSiliconGeneration.M3
        ))

        # Optimized reduction operations for M3
        patterns.append(FusionPattern(
            "fused_reduction",
            ["reduce", "add", "div"],
            lambda ops: ops[0].get("type") == "tt.reduce" and
                       ops[1].get("type") == "tt.binary.add" and
                       ops[2].get("type") == "tt.binary.div",
            min_hardware_gen=AppleSiliconGeneration.M3
        ))

        return patterns

    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply M3-specific fusion optimizations

        Args:
            graph: MLX computation graph representation

        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not self.is_applicable() or not graph or "ops" not in graph or not has_fusion_optimizer:
            return graph, {"m3_fused_ops": 0}

        # Create a specialized fusion optimizer with M3 patterns
        fusion_optimizer = FusionOptimizer(hardware_capabilities)

        # Add M3-specific patterns to the optimizer
        for pattern in self.m3_patterns:
            fusion_optimizer.patterns.append(pattern)

        # Apply fusion optimizations
        ops = graph["ops"]
        optimized_ops = fusion_optimizer.optimize(ops)

        # Count the number of fused operations
        fused_count = len(ops) - len(optimized_ops)

        # Update the graph
        optimized_graph = graph.copy()
        optimized_graph["ops"] = optimized_ops

        # Add M3-specific metadata
        if "metadata" not in optimized_graph:
            optimized_graph["metadata"] = {}
        optimized_graph["metadata"]["m3_optimized"] = True
        optimized_graph["metadata"]["m3_fusion_applied"] = fused_count > 0
        optimized_graph["metadata"]["m3_fusion_count"] = fused_count
        # Add details on M3-specific features
        optimized_graph["metadata"]["m3_features"] = {
            "tensor_cores_used": True,
            "simdgroup_width": 32,
            "shared_memory_size": 65536,  # 64KB for M3
            "vector_width": 8,
            "dynamic_caching": True
        }

        return optimized_graph, {"m3_fused_ops": fused_count}

class MemoryAccessOptimizationPass(OptimizationPass):
    """Optimizes memory access patterns for Metal"""

    def __init__(self):
        """Initialize memory access optimization pass"""
        super().__init__("memory_access_optimization")

    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply memory access pattern optimizations

        Args:
            graph: MLX computation graph representation

        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not graph or "ops" not in graph:
            return graph, {"memory_opts": 0}

        optimized_graph = graph.copy()
        ops = optimized_graph["ops"]

        # Track optimizations
        reordered_ops = 0
        data_layout_changes = 0

        # Analyze and optimize memory access patterns
        for i, op in enumerate(ops):
            op_type = op.get("type", "")

            # Optimize matrix operations for Metal's memory layout
            if "matmul" in op_type or "gemm" in op_type:
                ops[i] = self._optimize_matrix_memory_layout(op)
                data_layout_changes += 1

            # Optimize convolution operations for Metal's memory layout
            elif "conv" in op_type:
                ops[i] = self._optimize_conv_memory_layout(op)
                data_layout_changes += 1

            # Optimize element-wise operations for Metal's memory layout
            elif any(elem_op in op_type for elem_op in ["add", "sub", "mul", "div", "relu", "sigmoid", "tanh"]):
                ops[i] = self._optimize_elementwise_memory_access(op)
                data_layout_changes += 1

        # Reorder operations for better memory locality
        reordered_ops = self._reorder_operations_for_locality(ops)

        return optimized_graph, {"memory_opts": data_layout_changes + reordered_ops}

    def _optimize_matrix_memory_layout(self, op: Dict) -> Dict:
        """
        Optimize memory layout for matrix operations

        Args:
            op: Matrix operation

        Returns:
            Optimized operation
        """
        optimized_op = op.copy()

        # Add layout hints
        if "layout_hints" not in optimized_op:
            optimized_op["layout_hints"] = {}

        # Add matrix-specific layout hints
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
            # Optimized for M3 hardware
            optimized_op["layout_hints"]["layout"] = "block_based"
            optimized_op["layout_hints"]["block_size"] = 128  # Larger blocks for M3
            optimized_op["layout_hints"]["use_tensor_cores"] = True
            optimized_op["layout_hints"]["vectorize_width"] = 8
            optimized_op["layout_hints"]["shared_memory_size"] = 65536  # 64KB
        else:
            # Default for other hardware
            optimized_op["layout_hints"]["layout"] = "block_based"
            optimized_op["layout_hints"]["block_size"] = 64  # Standard block size
            optimized_op["layout_hints"]["use_tensor_cores"] = False
            optimized_op["layout_hints"]["vectorize_width"] = 4
            optimized_op["layout_hints"]["shared_memory_size"] = 32768  # 32KB

        return optimized_op

    def _optimize_conv_memory_layout(self, op: Dict) -> Dict:
        """
        Optimize memory layout for convolution operations

        Args:
            op: Convolution operation

        Returns:
            Optimized operation
        """
        optimized_op = op.copy()

        # Add layout hints
        if "layout_hints" not in optimized_op:
            optimized_op["layout_hints"] = {}

        # Add convolution-specific layout hints
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
            # Optimized for M3 hardware
            optimized_op["layout_hints"]["layout"] = "texture_optimized"
            optimized_op["layout_hints"]["tile_size"] = [128, 128]
            optimized_op["layout_hints"]["use_texture_memory"] = True
            optimized_op["layout_hints"]["filter_layout"] = "simdgroup_optimized"
            optimized_op["layout_hints"]["vectorize_width"] = 8
        else:
            # Default for other hardware
            optimized_op["layout_hints"]["layout"] = "texture_optimized"
            optimized_op["layout_hints"]["tile_size"] = [64, 64]
            optimized_op["layout_hints"]["use_texture_memory"] = True
            optimized_op["layout_hints"]["filter_layout"] = "simdgroup_optimized"
            optimized_op["layout_hints"]["vectorize_width"] = 4

        return optimized_op

    def _optimize_elementwise_memory_access(self, op: Dict) -> Dict:
        """
        Optimize memory access for element-wise operations

        Args:
            op: Element-wise operation

        Returns:
            Optimized operation
        """
        optimized_op = op.copy()

        # Add memory access hints
        if "access_hints" not in optimized_op:
            optimized_op["access_hints"] = {}

        # Add element-wise specific access hints
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
            # Optimized for M3 hardware
            optimized_op["access_hints"]["vectorize"] = True
            optimized_op["access_hints"]["vectorize_width"] = 8
            optimized_op["access_hints"]["coalesce_memory_access"] = True
            optimized_op["access_hints"]["unroll_factor"] = 4
        else:
            # Default for other hardware
            optimized_op["access_hints"]["vectorize"] = True
            optimized_op["access_hints"]["vectorize_width"] = 4
            optimized_op["access_hints"]["coalesce_memory_access"] = True
            optimized_op["access_hints"]["unroll_factor"] = 2

        return optimized_op

    def _reorder_operations_for_locality(self, ops: List[Dict]) -> int:
        """
        Reorder operations for better memory locality

        Args:
            ops: List of operations

        Returns:
            Number of reordered operations
        """
        # This implementation is simplified; a real implementation would analyze
        # the data flow graph and reorder operations to maximize data locality

        # For now, we'll just return 0 as a placeholder
        return 0

class MetalSpecificOptimizationPass(OptimizationPass):
    """Metal-specific optimizations"""

    def __init__(self):
        """Initialize Metal-specific optimization pass"""
        super().__init__("metal_specific_optimization")

    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply Metal-specific optimizations

        Args:
            graph: MLX computation graph representation

        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not graph or "ops" not in graph:
            return graph, {"metal_opts": 0}

        optimized_graph = graph.copy()
        ops = optimized_graph["ops"]

        # Add Metal-specific metadata
        if "metadata" not in optimized_graph:
            optimized_graph["metadata"] = {}

        optimized_graph["metadata"]["metal_optimized"] = True

        # Add hardware-specific optimizations
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
            optimized_graph["metadata"]["use_simdgroup_matrix"] = True
            optimized_graph["metadata"]["use_tensor_cores"] = True
            optimized_graph["metadata"]["simdgroup_width"] = 32
            optimized_graph["metadata"]["threadgroup_size"] = 1024
            optimized_graph["metadata"]["vector_width"] = 8
            optimized_graph["metadata"]["shared_memory_size"] = 65536  # 64KB
        else:
            optimized_graph["metadata"]["use_simdgroup_matrix"] = True
            optimized_graph["metadata"]["use_tensor_cores"] = False
            optimized_graph["metadata"]["simdgroup_width"] = 16
            optimized_graph["metadata"]["threadgroup_size"] = 512
            optimized_graph["metadata"]["vector_width"] = 4
            optimized_graph["metadata"]["shared_memory_size"] = 32768  # 32KB

        # Track optimizations
        matmul_opts = 0
        reduction_opts = 0
        softmax_opts = 0

        # Optimize specific operations
        for i, op in enumerate(ops):
            op_type = op.get("type", "")

            # Optimize matrix multiplication for Metal
            if "matmul" in op_type:
                ops[i] = self._optimize_matmul_for_metal(op)
                matmul_opts += 1

            # Optimize reduction operations for Metal
            elif "reduce" in op_type:
                ops[i] = self._optimize_reduction_for_metal(op)
                reduction_opts += 1

            # Optimize softmax for Metal
            elif "softmax" in op_type:
                ops[i] = self._optimize_softmax_for_metal(op)
                softmax_opts += 1

        total_opts = matmul_opts + reduction_opts + softmax_opts

        return optimized_graph, {"metal_opts": total_opts}

    def _optimize_matmul_for_metal(self, op: Dict) -> Dict:
        """
        Optimize matrix multiplication for Metal

        Args:
            op: Matrix multiplication operation

        Returns:
            Optimized operation
        """
        optimized_op = op.copy()

        # Add Metal-specific hints
        if "metal_hints" not in optimized_op:
            optimized_op["metal_hints"] = {}

        # Set matrix multiplication hints
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
            # Optimized for M3 hardware
            optimized_op["metal_hints"]["use_simdgroup_matrix"] = True
            optimized_op["metal_hints"]["use_tensor_cores"] = True
            optimized_op["metal_hints"]["matrix_tile_size"] = 16
            optimized_op["metal_hints"]["preferred_workgroup_size"] = (8, 8, 1)
            optimized_op["metal_hints"]["prefetch_depth"] = 2
        else:
            # Default for other hardware
            optimized_op["metal_hints"]["use_simdgroup_matrix"] = True
            optimized_op["metal_hints"]["use_tensor_cores"] = False
            optimized_op["metal_hints"]["matrix_tile_size"] = 8
            optimized_op["metal_hints"]["preferred_workgroup_size"] = (8, 8, 1)
            optimized_op["metal_hints"]["prefetch_depth"] = 1

        return optimized_op

    def _optimize_reduction_for_metal(self, op: Dict) -> Dict:
        """
        Optimize reduction operations for Metal

        Args:
            op: Reduction operation

        Returns:
            Optimized operation
        """
        optimized_op = op.copy()

        # Add Metal-specific hints
        if "metal_hints" not in optimized_op:
            optimized_op["metal_hints"] = {}

        # Set reduction hints
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
            # Optimized for M3 hardware
            optimized_op["metal_hints"]["use_hierarchical_reduction"] = True
            optimized_op["metal_hints"]["threadgroup_size"] = 1024
            optimized_op["metal_hints"]["vectorize_width"] = 8
        else:
            # Default for other hardware
            optimized_op["metal_hints"]["use_hierarchical_reduction"] = True
            optimized_op["metal_hints"]["threadgroup_size"] = 512
            optimized_op["metal_hints"]["vectorize_width"] = 4

        return optimized_op

    def _optimize_softmax_for_metal(self, op: Dict) -> Dict:
        """
        Optimize softmax for Metal

        Args:
            op: Softmax operation

        Returns:
            Optimized operation
        """
        optimized_op = op.copy()

        # Add Metal-specific hints
        if "metal_hints" not in optimized_op:
            optimized_op["metal_hints"] = {}

        # Set softmax hints
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
            # Optimized for M3 hardware with improved reduction
            optimized_op["metal_hints"]["use_fast_softmax"] = True
            optimized_op["metal_hints"]["use_simdgroup_reduction"] = True
            optimized_op["metal_hints"]["use_shared_memory"] = True
            optimized_op["metal_hints"]["shared_memory_size"] = 65536  # 64KB
        else:
            # Default for other hardware
            optimized_op["metal_hints"]["use_fast_softmax"] = True
            optimized_op["metal_hints"]["use_simdgroup_reduction"] = True
            optimized_op["metal_hints"]["use_shared_memory"] = True
            optimized_op["metal_hints"]["shared_memory_size"] = 32768  # 32KB

        return optimized_op

class MemoryOptimizationPass(OptimizationPass):
    """Optimize memory allocation and usage"""

    def __init__(self):
        """Initialize memory optimization pass"""
        super().__init__("memory_optimization")

        # Get memory manager instance
        if has_memory_manager:
            self.memory_manager = get_metal_memory_manager()
            self.has_memory_manager = True
        else:
            self.has_memory_manager = False

    def is_applicable(self) -> bool:
        """
        Check if this optimization pass is applicable

        Returns:
            True if applicable, False otherwise
        """
        return has_memory_manager

    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply memory optimizations

        Args:
            graph: MLX computation graph representation

        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not self.is_applicable() or not graph or "ops" not in graph:
            return graph, {"memory_optimizations": 0}

        # Apply memory optimizations using the memory manager
        optimized_graph = self.memory_manager.optimize_graph_memory(graph)

        # Count the number of operations with memory optimizations
        optimized_count = sum(1 for op in optimized_graph["ops"] if "tensors" in op)

        return optimized_graph, {"memory_optimizations": optimized_count}

class MLXGraphOptimizer:
    """Optimizer for MLX computation graphs"""

    def __init__(self, hardware_capabilities=None):
        """
        Initialize MLX graph optimizer

        Args:
            hardware_capabilities: Hardware capabilities object
        """
        self.hardware_capabilities = hardware_capabilities or globals().get("hardware_capabilities")
        self.passes = self._create_optimization_passes()

    def _create_optimization_passes(self) -> List[OptimizationPass]:
        """
        Create optimization passes

        Returns:
            List of optimization passes
        """
        passes = []

        # Add operation fusion pass
        passes.append(OperationFusionPass())

        # Add memory access optimization pass
        passes.append(MemoryAccessOptimizationPass())

        # Add Metal-specific optimization pass
        passes.append(MetalSpecificOptimizationPass())

        # Add M3-specific fusion pass
        passes.append(M3SpecificFusionPass())

        # Add memory optimization pass if memory manager is available
        if has_memory_manager:
            passes.append(MemoryOptimizationPass())

        return passes

    def optimize(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Optimize MLX computation graph

        Args:
            graph: MLX computation graph representation

        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not graph or "ops" not in graph:
            return graph, {}

        # Start with the original graph
        optimized_graph = graph.copy()

        # Track overall statistics
        stats = {}

        # Apply optimization passes
        for optimization_pass in self.passes:
            # Check if the pass is applicable
            if optimization_pass.is_applicable():
                # Apply the pass
                optimized_graph, pass_stats = optimization_pass.apply(optimized_graph)

                # Update statistics
                for key, value in pass_stats.items():
                    stats[key] = value

        return optimized_graph, stats

# Global instance
_mlx_graph_optimizer = None

def optimize(graph: Dict) -> Tuple[Dict, Dict]:
    """
    Optimize MLX computation graph using global optimizer instance

    Args:
        graph: MLX computation graph representation

    Returns:
        Tuple of (optimized graph, optimization stats)
    """
    global _mlx_graph_optimizer
    if _mlx_graph_optimizer is None:
        _mlx_graph_optimizer = MLXGraphOptimizer(hardware_capabilities)
    return _mlx_graph_optimizer.optimize(graph)