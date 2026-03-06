"""
M3-Specific Graph Optimizer for Triton Metal Backend

This module provides optimization passes specifically designed for Apple M3 GPUs,
leveraging their unique features like Dynamic Caching, hardware-accelerated ray
tracing, and hardware-accelerated mesh shading.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from enum import Enum



# Import existing optimizers safely
try:
    from mlx.mlx_graph_optimizer import OptimizationPass, MLXGraphOptimizer
    from MLX.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
    from MLX.metal_fusion_optimizer import FusionOptimizer, FusionPattern
except ImportError:
    print("Warning: Could not import required modules for M3 optimization.")
    
    # Define fallback classes for testing
    class OptimizationPass:
        def __init__(self, name: str, min_hardware_gen=None):
            self.name = name
            self.min_hardware_gen = min_hardware_gen
            
        def is_applicable(self) -> bool:
            return True
            
        def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
            return graph, {}
    
    class DummyEnum(Enum):
        UNKNOWN = 0
        M1 = 1
        M2 = 2
        M3 = 3
        
    class DummyCapabilities:
        def __init__(self):
            self.chip_generation = DummyEnum.M3
    
    AppleSiliconGeneration = DummyEnum
    hardware_capabilities = DummyCapabilities()

class M3Features(Enum):
    """Enum for M3-specific hardware features"""
    DYNAMIC_CACHING = 0
    HARDWARE_RAY_TRACING = 1
    HARDWARE_MESH_SHADING = 2
    HIGH_PERFORMANCE_ALU = 3
    FLEXIBLE_MEMORY = 4
    LARGER_THREAD_OCCUPANCY = 5
    SIMDGROUP_ENHANCEMENTS = 6

class M3DynamicCachingPass(OptimizationPass):
    """Optimization pass that leverages M3's Dynamic Caching"""
    
    def __init__(self):
        """Initialize Dynamic Caching optimization pass"""
        super().__init__("m3_dynamic_caching", min_hardware_gen=AppleSiliconGeneration.M3)
        
    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply Dynamic Caching optimizations to the computation graph
        
        Args:
            graph: MLX computation graph representation
            
        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not self.is_applicable() or not graph or "ops" not in graph:
            return graph, {"dynamic_caching_optimized_ops": 0}
            
        optimized_ops = []
        optimized_count = 0
        
        for op in graph["ops"]:
            # Analyze op to determine if it can benefit from dynamic caching
            op_type = op.get("type", "")
            
            if op_type in ["matmul", "conv2d", "reduce", "scan"]:
                # These operations typically have variable register usage patterns
                # and can benefit from dynamic register allocation
                optimized_op = self._optimize_for_dynamic_caching(op)
                optimized_ops.append(optimized_op)
                optimized_count += 1
            else:
                optimized_ops.append(op)
        
        # Update the graph
        optimized_graph = graph.copy()
        optimized_graph["ops"] = optimized_ops
        
        return optimized_graph, {"dynamic_caching_optimized_ops": optimized_count}
    
    def _optimize_for_dynamic_caching(self, op: Dict) -> Dict:
        """
        Apply dynamic caching optimizations to a specific operation
        
        Args:
            op: Operation to optimize
            
        Returns:
            Optimized operation
        """
        optimized_op = op.copy()
        
        # Add metadata to indicate dynamic caching optimizations
        if "metadata" not in optimized_op:
            optimized_op["metadata"] = {}
            
        optimized_op["metadata"]["dynamic_caching_enabled"] = True
        
        # Adjust threadgroup sizes for optimal register usage with dynamic caching
        if "threadgroup_size" in optimized_op:
            # For operations with high register pressure, we can increase the threadgroup size
            # since M3 can dynamically allocate registers more efficiently
            op_type = op.get("type", "")
            current_size = optimized_op["threadgroup_size"]
            
            if op_type == "matmul":
                # MatMul operations can benefit from larger threadgroups on M3
                optimized_op["threadgroup_size"] = min(current_size * 2, 1024)
            elif op_type == "conv2d":
                # Conv2D operations can also benefit from larger threadgroups
                optimized_op["threadgroup_size"] = min(current_size * 2, 1024)
                
        # Adjust dispatch parameters to better utilize dynamic caching
        if "execution_parameters" not in optimized_op:
            optimized_op["execution_parameters"] = {}
            
        optimized_op["execution_parameters"]["prefer_dynamic_register_allocation"] = True
        
        return optimized_op

class M3RayTracingPass(OptimizationPass):
    """Optimization pass that leverages M3's hardware-accelerated ray tracing"""
    
    def __init__(self):
        """Initialize Ray Tracing optimization pass"""
        super().__init__("m3_ray_tracing", min_hardware_gen=AppleSiliconGeneration.M3)
        
    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply Ray Tracing optimizations to the computation graph
        
        Args:
            graph: MLX computation graph representation
            
        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not self.is_applicable() or not graph or "ops" not in graph:
            return graph, {"ray_tracing_optimized_ops": 0}
            
        optimized_ops = []
        optimized_count = 0
        
        # Detect ray tracing operations in the graph
        for op in graph["ops"]:
            # Check if operation is related to ray tracing
            is_ray_tracing_op = self._is_ray_tracing_operation(op)
            
            if is_ray_tracing_op:
                optimized_op = self._optimize_for_hardware_ray_tracing(op)
                optimized_ops.append(optimized_op)
                optimized_count += 1
            else:
                optimized_ops.append(op)
        
        # Update the graph
        optimized_graph = graph.copy()
        optimized_graph["ops"] = optimized_ops
        
        return optimized_graph, {"ray_tracing_optimized_ops": optimized_count}
    
    def _is_ray_tracing_operation(self, op: Dict) -> bool:
        """
        Check if an operation is related to ray tracing
        
        Args:
            op: Operation to check
            
        Returns:
            True if operation is related to ray tracing, False otherwise
        """
        op_type = op.get("type", "")
        op_name = op.get("name", "")
        
        # Check operation type or name for ray tracing indicators
        rt_op_types = ["ray_intersect", "ray_trace", "ray_query", "ray_cast"]
        rt_name_indicators = ["ray", "intersect", "trace", "bvh", "acceleration_structure"]
        
        if op_type in rt_op_types:
            return True
            
        # Check if operation name contains ray tracing indicators
        for indicator in rt_name_indicators:
            if indicator in op_name.lower():
                return True
                
        # Check for ray tracing in attributes or parameters
        attributes = op.get("attributes", {})
        for attr_name, attr_value in attributes.items():
            if isinstance(attr_value, str) and "ray" in attr_value.lower():
                return True
                
        return False
    
    def _optimize_for_hardware_ray_tracing(self, op: Dict) -> Dict:
        """
        Apply hardware ray tracing optimizations to a specific operation
        
        Args:
            op: Operation to optimize
            
        Returns:
            Optimized operation
        """
        optimized_op = op.copy()
        
        # Add metadata to indicate hardware ray tracing optimizations
        if "metadata" not in optimized_op:
            optimized_op["metadata"] = {}
            
        optimized_op["metadata"]["hardware_ray_tracing_enabled"] = True
        
        # Configure to use the hardware-accelerated intersector API rather than query API
        if "execution_parameters" not in optimized_op:
            optimized_op["execution_parameters"] = {}
            
        optimized_op["execution_parameters"]["use_hardware_ray_tracing"] = True
        optimized_op["execution_parameters"]["prefer_intersector_api"] = True
        optimized_op["execution_parameters"]["enable_reorder_stage"] = True
        
        # Optimize ray payload size to minimize on-chip memory usage
        optimized_op["execution_parameters"]["optimize_ray_payload"] = True
        
        return optimized_op

class M3MeshShadingPass(OptimizationPass):
    """Optimization pass that leverages M3's hardware-accelerated mesh shading"""
    
    def __init__(self):
        """Initialize Mesh Shading optimization pass"""
        super().__init__("m3_mesh_shading", min_hardware_gen=AppleSiliconGeneration.M3)
        
    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply Mesh Shading optimizations to the computation graph
        
        Args:
            graph: MLX computation graph representation
            
        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not self.is_applicable() or not graph or "ops" not in graph:
            return graph, {"mesh_shading_optimized_ops": 0}
            
        optimized_ops = []
        optimized_count = 0
        
        # Identify mesh or geometry shader operations in the graph
        for op in graph["ops"]:
            # Check if operation is related to mesh shading or geometry processing
            is_mesh_op = self._is_mesh_operation(op)
            
            if is_mesh_op:
                optimized_op = self._optimize_for_hardware_mesh_shading(op)
                optimized_ops.append(optimized_op)
                optimized_count += 1
            else:
                optimized_ops.append(op)
        
        # Update the graph
        optimized_graph = graph.copy()
        optimized_graph["ops"] = optimized_ops
        
        return optimized_graph, {"mesh_shading_optimized_ops": optimized_count}
    
    def _is_mesh_operation(self, op: Dict) -> bool:
        """
        Check if an operation is related to mesh shading or geometry processing
        
        Args:
            op: Operation to check
            
        Returns:
            True if operation is related to mesh shading, False otherwise
        """
        op_type = op.get("type", "")
        op_name = op.get("name", "")
        
        # Check operation type or name for mesh shading indicators
        mesh_op_types = ["mesh", "geometry", "tessellation", "vertex_processing", "primitive_assembly"]
        mesh_name_indicators = ["mesh", "geom", "tess", "vertex", "primitive"]
        
        if op_type in mesh_op_types:
            return True
            
        # Check if operation name contains mesh shading indicators
        for indicator in mesh_name_indicators:
            if indicator in op_name.lower():
                return True
                
        # Check for mesh shading in attributes
        attributes = op.get("attributes", {})
        if "pipeline_type" in attributes and attributes["pipeline_type"] == "mesh":
            return True
            
        return False
    
    def _optimize_for_hardware_mesh_shading(self, op: Dict) -> Dict:
        """
        Apply hardware mesh shading optimizations to a specific operation
        
        Args:
            op: Operation to optimize
            
        Returns:
            Optimized operation
        """
        optimized_op = op.copy()
        
        # Add metadata to indicate hardware mesh shading optimizations
        if "metadata" not in optimized_op:
            optimized_op["metadata"] = {}
            
        optimized_op["metadata"]["hardware_mesh_shading_enabled"] = True
        
        # Configure to use hardware-accelerated mesh shading
        if "execution_parameters" not in optimized_op:
            optimized_op["execution_parameters"] = {}
            
        optimized_op["execution_parameters"]["use_hardware_mesh_shading"] = True
        
        # Support for larger threadgroup count (over 1 million vs 1024)
        optimized_op["execution_parameters"]["expanded_mesh_grid"] = True
        
        # Configure for optimal mesh data sizes
        optimized_op["execution_parameters"]["optimize_mesh_data_types"] = True
        optimized_op["execution_parameters"]["optimize_primitive_culling"] = True
        
        return optimized_op

class M3SIMDGroupOptimizationPass(OptimizationPass):
    """Optimization pass that leverages M3's improved SIMD group execution model"""
    
    def __init__(self):
        """Initialize SIMD group optimization pass"""
        super().__init__("m3_simdgroup_optimization", min_hardware_gen=AppleSiliconGeneration.M3)
        
    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply SIMD group optimizations to the computation graph
        
        Args:
            graph: MLX computation graph representation
            
        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not self.is_applicable() or not graph or "ops" not in graph:
            return graph, {"simdgroup_optimized_ops": 0}
            
        optimized_ops = []
        optimized_count = 0
        
        # Optimize operations that can benefit from improved SIMD group execution
        for op in graph["ops"]:
            # Most compute-heavy operations can benefit from SIMD group optimizations
            can_optimize = self._can_optimize_simdgroups(op)
            
            if can_optimize:
                optimized_op = self._optimize_for_simdgroups(op)
                optimized_ops.append(optimized_op)
                optimized_count += 1
            else:
                optimized_ops.append(op)
        
        # Update the graph
        optimized_graph = graph.copy()
        optimized_graph["ops"] = optimized_ops
        
        return optimized_graph, {"simdgroup_optimized_ops": optimized_count}
    
    def _can_optimize_simdgroups(self, op: Dict) -> bool:
        """
        Check if an operation can benefit from SIMD group optimizations
        
        Args:
            op: Operation to check
            
        Returns:
            True if operation can benefit from SIMD group optimizations, False otherwise
        """
        op_type = op.get("type", "")
        
        # Operations that typically benefit from SIMD group optimizations
        simd_optimizable_ops = [
            "matmul", "conv2d", "reduce", "scan", "sort", "elementwise", 
            "transform", "broadcast", "fft", "gemm", "batch_norm"
        ]
        
        return op_type in simd_optimizable_ops
    
    def _optimize_for_simdgroups(self, op: Dict) -> Dict:
        """
        Apply SIMD group optimizations to a specific operation
        
        Args:
            op: Operation to optimize
            
        Returns:
            Optimized operation
        """
        optimized_op = op.copy()
        
        # Add metadata to indicate SIMD group optimizations
        if "metadata" not in optimized_op:
            optimized_op["metadata"] = {}
            
        optimized_op["metadata"]["simdgroup_optimized"] = True
        
        # Configure for improved SIMD group execution
        if "execution_parameters" not in optimized_op:
            optimized_op["execution_parameters"] = {}
            
        # Enable FP16/FP32/Int parallel execution
        optimized_op["execution_parameters"]["enable_mixed_precision_parallelism"] = True
        
        # Configure for higher thread occupancy
        optimized_op["execution_parameters"]["target_higher_occupancy"] = True
        
        # Adjust simdgroup size for the M3 architecture
        op_type = op.get("type", "")
        if op_type == "matmul":
            # M3 can handle larger SIMD groups for matrix operations efficiently
            optimized_op["execution_parameters"]["simdgroup_matrix_size"] = 16  # Up from 8
        elif op_type in ["conv2d", "reduce"]:
            # Optimize reduction and convolution operations for M3's wider SIMD width
            optimized_op["execution_parameters"]["simd_width"] = 32  # M3-optimized SIMD width
            
        return optimized_op

class M3MemoryOptimizationPass(OptimizationPass):
    """Optimization pass that leverages M3's flexible on-chip memory"""
    
    def __init__(self):
        """Initialize Memory optimization pass"""
        super().__init__("m3_memory_optimization", min_hardware_gen=AppleSiliconGeneration.M3)
        
    def apply(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply Memory optimizations to the computation graph
        
        Args:
            graph: MLX computation graph representation
            
        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not self.is_applicable() or not graph or "ops" not in graph:
            return graph, {"memory_optimized_ops": 0}
            
        optimized_ops = []
        optimized_count = 0
        
        # Find operations with heavy memory access patterns
        for op in graph["ops"]:
            # Check memory access patterns
            has_heavy_memory_access = self._has_heavy_memory_access(op)
            
            if has_heavy_memory_access:
                optimized_op = self._optimize_for_flexible_memory(op)
                optimized_ops.append(optimized_op)
                optimized_count += 1
            else:
                optimized_ops.append(op)
        
        # Update the graph
        optimized_graph = graph.copy()
        optimized_graph["ops"] = optimized_ops
        
        return optimized_graph, {"memory_optimized_ops": optimized_count}
    
    def _has_heavy_memory_access(self, op: Dict) -> bool:
        """
        Check if an operation has heavy memory access patterns
        
        Args:
            op: Operation to check
            
        Returns:
            True if operation has heavy memory access, False otherwise
        """
        op_type = op.get("type", "")
        
        # Operations that typically have heavy memory access
        memory_heavy_ops = [
            "gather", "scatter", "transpose", "reshape", "slice", "concat",
            "pool", "embedding", "attention", "layer_norm", "batch_norm"
        ]
        
        # Check memory access patterns in attributes
        attributes = op.get("attributes", {})
        has_memory_attributes = False
        
        memory_attribute_indicators = ["buffer", "threadgroup", "tile", "shared", "stack"]
        for attr_name in attributes:
            for indicator in memory_attribute_indicators:
                if indicator in attr_name.lower():
                    has_memory_attributes = True
                    break
            if has_memory_attributes:
                break
        
        return op_type in memory_heavy_ops or has_memory_attributes
    
    def _optimize_for_flexible_memory(self, op: Dict) -> Dict:
        """
        Apply flexible memory optimizations to a specific operation
        
        Args:
            op: Operation to optimize
            
        Returns:
            Optimized operation
        """
        optimized_op = op.copy()
        
        # Add metadata to indicate flexible memory optimizations
        if "metadata" not in optimized_op:
            optimized_op["metadata"] = {}
            
        optimized_op["metadata"]["flexible_memory_optimized"] = True
        
        # Configure for flexible on-chip memory usage
        if "execution_parameters" not in optimized_op:
            optimized_op["execution_parameters"] = {}
            
        # Enable flexible on-chip memory
        optimized_op["execution_parameters"]["use_flexible_memory"] = True
        
        # Optimize for specific memory types based on operation
        op_type = op.get("type", "")
        if op_type in ["gather", "scatter", "embedding"]:
            # These operations benefit from buffer memory optimization
            optimized_op["execution_parameters"]["optimize_buffer_access"] = True
        elif op_type in ["layer_norm", "batch_norm", "attention"]:
            # These operations benefit from threadgroup memory optimization
            optimized_op["execution_parameters"]["optimize_threadgroup_memory"] = True
        elif op_type in ["pool", "conv2d"]:
            # These operations benefit from tile memory optimization
            optimized_op["execution_parameters"]["optimize_tile_memory"] = True
            
        return optimized_op

class M3GraphOptimizer:
    """Graph optimizer specifically for M3 hardware"""
    
    def __init__(self, hardware_capabilities=None):
        """
        Initialize M3 graph optimizer
        
        Args:
            hardware_capabilities: Optional hardware capabilities instance
        """
        self.hardware = hardware_capabilities
        self.optimization_passes = self._create_optimization_passes()
        
    def _create_optimization_passes(self) -> List[OptimizationPass]:
        """
        Create optimization passes specifically for M3 hardware
        
        Returns:
            List of optimization passes
        """
        passes = []
        
        # Add M3-specific optimization passes
        passes.append(M3DynamicCachingPass())
        passes.append(M3RayTracingPass())
        passes.append(M3MeshShadingPass())
        passes.append(M3SIMDGroupOptimizationPass())
        passes.append(M3MemoryOptimizationPass())
        
        return passes
        
    def optimize(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Apply M3-specific optimizations to the computation graph
        
        Args:
            graph: MLX computation graph representation
            
        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not graph:
            return graph, {}
            
        optimized_graph = graph
        all_stats = {}
        
        # Apply each optimization pass
        for opt_pass in self.optimization_passes:
            if opt_pass.is_applicable():
                optimized_graph, stats = opt_pass.apply(optimized_graph)
                all_stats.update(stats)
        
        return optimized_graph, all_stats
        
    def get_available_features(self) -> Dict[M3Features, bool]:
        """
        Get available M3 features
        
        Returns:
            Dictionary mapping features to availability
        """
        # Check if we're on M3 hardware
        is_m3 = (
            hasattr(self.hardware, 'chip_generation') and 
            self.hardware.chip_generation == AppleSiliconGeneration.M3
        )
        
        features = {feature: is_m3 for feature in M3Features}
        
        return features

# Create global instance for direct access
m3_graph_optimizer = M3GraphOptimizer(hardware_capabilities)

# Function to get singleton instance
def get_m3_graph_optimizer():
    """Get singleton M3 graph optimizer instance"""
    return m3_graph_optimizer
