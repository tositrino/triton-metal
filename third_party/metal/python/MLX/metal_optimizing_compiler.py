"""
Metal-specific Graph Optimizing Compiler for Triton

This module combines Metal-specific optimizations into a comprehensive 
compiler pipeline, integrating memory layout optimizations, operation
fusion, and hardware-specific enhancements.
"""

import os
import sys

import time
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable

# Import Metal optimization components
try:
    from MLX.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
    from MLX.memory_layout_optimizer import (
        optimize_memory_layout, 
        LayoutOptimizationLevel,
        get_metal_layout_optimizer
    )
    from MLX.metal_operation_fusion import (
        optimize_operation_fusion,
        get_metal_fusion_optimizer
    )
    from MLX.mlx_graph_optimizer import optimize_mlx_graph
    
    # Import M3-specific optimizations if available
    try:
        from M3.m3_memory_manager import get_m3_memory_manager
        has_m3_optimizations = True
    except ImportError:
        has_m3_optimizations = False
        
except ImportError as e:
    print(f"Warning: Some Metal optimization components could not be imported: {e}")
    # Define dummy optimization functions
    def optimize_memory_layout(graph, level=None):
        return graph, {}
        
    def optimize_operation_fusion(graph):
        return graph, {}
        
    def optimize_mlx_graph(graph, hardware_gen=None):
        return graph, {}
        
    has_m3_optimizations = False
    
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

class OptimizationLevel(Enum):
    """Optimization level for Metal compiler"""
    NONE = 0        # No optimizations
    BASIC = 1       # Basic optimizations for compatibility
    STANDARD = 2    # Standard optimizations for good performance
    AGGRESSIVE = 3  # Aggressive optimizations for max performance
    EXPERIMENTAL = 4  # Experimental optimizations (may affect stability)

class MetalOptimizingCompiler:
    """
    Comprehensive Metal-specific optimizing compiler that integrates multiple
    optimization passes for improved performance on Apple Silicon GPUs.
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        """
        Initialize Metal optimizing compiler
        
        Args:
            optimization_level: Level of optimization to apply
        """
        self.optimization_level = optimization_level
        
        # Detect hardware generation
        self.hardware_gen = self._detect_hardware_generation()
        
        # Enable hardware-specific optimizations
        self.use_m3_optimizations = (
            has_m3_optimizations and 
            self.hardware_gen == AppleSiliconGeneration.M3 and
            self.optimization_level.value >= OptimizationLevel.STANDARD.value
        )
        
        # Set up optimization pipeline
        self.optimization_pipeline = self._create_optimization_pipeline()
        
        # Track optimization statistics
        self.stats = {
            "compile_time": 0,
            "optimization_level": optimization_level.name,
            "hardware_generation": self.hardware_gen.name if self.hardware_gen else "UNKNOWN",
            "passes": {}
        }
    
    def _detect_hardware_generation(self) -> Optional[Any]:
        """
        Detect hardware generation
        
        Returns:
            Hardware generation enum value
        """
        if hasattr(hardware_capabilities, "chip_generation"):
            return hardware_capabilities.chip_generation
        return None
    
    def _create_optimization_pipeline(self) -> List[Tuple[str, Callable]]:
        """
        Create optimization pipeline based on optimization level
        
        Returns:
            List of (name, function) tuples for optimization passes
        """
        pipeline = []
        
        # NONE level: No optimizations
        if self.optimization_level == OptimizationLevel.NONE:
            return []
            
        # BASIC level: Essential optimizations
        if self.optimization_level.value >= OptimizationLevel.BASIC.value:
            # Add basic MLX graph optimizations
            pipeline.append(("mlx_graph_basic", 
                lambda graph: optimize_mlx_graph(graph, self.hardware_gen)))
                
            # Add basic memory layout optimizations with BASIC level
            pipeline.append(("memory_layout_basic", 
                lambda graph: optimize_memory_layout(graph, LayoutOptimizationLevel.BASIC)))
        
        # STANDARD level: Good performance optimizations
        if self.optimization_level.value >= OptimizationLevel.STANDARD.value:
            # Update memory layout optimization to HARDWARE_SPECIFIC level
            # Replace the basic memory layout pass
            if pipeline and pipeline[-1][0] == "memory_layout_basic":
                pipeline.pop()
                
            pipeline.append(("memory_layout_standard", 
                lambda graph: optimize_memory_layout(graph, LayoutOptimizationLevel.HARDWARE_SPECIFIC)))
                
            # Add operation fusion
            pipeline.append(("operation_fusion", optimize_operation_fusion))
            
            # Add M3-specific memory optimizations if applicable
            if self.use_m3_optimizations:
                pipeline.append(("m3_memory_optimization", 
                    lambda graph: (get_m3_memory_manager().optimize_graph_memory(graph), {})))
        
        # AGGRESSIVE level: Maximum performance optimizations
        if self.optimization_level.value >= OptimizationLevel.AGGRESSIVE.value:
            # Add aggressive MLX optimizations
            pipeline.append(("mlx_graph_aggressive", 
                lambda graph: optimize_mlx_graph(graph, self.hardware_gen, aggressive=True)))
        
        # EXPERIMENTAL level: Experimental optimizations
        if self.optimization_level.value >= OptimizationLevel.EXPERIMENTAL.value:
            # Add experimental passes
            pass
            
        return pipeline
    
    def compile(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Compile and optimize computation graph for Metal
        
        Args:
            graph: Computation graph
            
        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not graph:
            return graph, self.stats
            
        # Reset statistics
        self.stats = {
            "compile_time": 0,
            "optimization_level": self.optimization_level.name,
            "hardware_generation": self.hardware_gen.name if self.hardware_gen else "UNKNOWN",
            "passes": {}
        }
        
        # Measure compilation time
        start_time = time.time()
        
        # Make a copy of the input graph
        optimized_graph = graph.copy()
        
        # Run optimization pipeline
        for name, optimize_fn in self.optimization_pipeline:
            pass_start_time = time.time()
            
            # Apply optimization
            try:
                result, pass_stats = optimize_fn(optimized_graph)
                optimized_graph = result
                
                # Record statistics
                pass_end_time = time.time()
                self.stats["passes"][name] = {
                    "time": pass_end_time - pass_start_time,
                    "stats": pass_stats
                }
            except Exception as e:
                # Log error and continue with next pass
                print(f"Warning: Optimization pass '{name}' failed: {e}")
                self.stats["passes"][name] = {
                    "error": str(e)
                }
        
        # Record total compilation time
        end_time = time.time()
        self.stats["compile_time"] = end_time - start_time
        
        # Add metadata to graph
        if "metadata" not in optimized_graph:
            optimized_graph["metadata"] = {}
            
        optimized_graph["metadata"]["metal_optimized"] = True
        optimized_graph["metadata"]["optimization_level"] = self.optimization_level.name
        
        if self.hardware_gen:
            optimized_graph["metadata"]["optimized_for"] = self.hardware_gen.name
            
        if self.use_m3_optimizations:
            optimized_graph["metadata"]["m3_optimized"] = True
        
        return optimized_graph, self.stats
    
    def get_optimization_summary(self) -> Dict:
        """
        Get summary of applied optimizations
        
        Returns:
            Summary dictionary
        """
        summary = {
            "optimization_level": self.optimization_level.name,
            "hardware_generation": self.hardware_gen.name if self.hardware_gen else "UNKNOWN",
            "compile_time": self.stats["compile_time"],
            "passes_applied": list(self.stats["passes"].keys())
        }
        
        # Add key statistics from each pass
        pass_stats = {}
        for name, stats in self.stats["passes"].items():
            if "error" in stats:
                pass_stats[name] = {"status": "failed", "error": stats["error"]}
            else:
                # Extract key statistics based on pass type
                key_stats = {}
                
                if name.startswith("memory_layout"):
                    pass_data = stats.get("stats", {})
                    key_stats["optimized_ops"] = pass_data.get("optimized_ops", 0)
                    key_stats["memory_layout_changes"] = pass_data.get("memory_layout_changes", 0)
                    
                elif name.startswith("operation_fusion"):
                    pass_data = stats.get("stats", {})
                    key_stats["fused_ops"] = pass_data.get("fused_ops", 0)
                    key_stats["fusion_patterns"] = pass_data.get("fusion_patterns", {})
                    key_stats["hardware_specific_fusions"] = pass_data.get("hardware_specific_fusions", 0)
                    
                elif name.startswith("mlx_graph"):
                    pass_data = stats.get("stats", {})
                    key_stats["optimized_ops"] = pass_data.get("optimized_ops", 0)
                    key_stats["optimizations_applied"] = pass_data.get("optimizations", {})
                
                pass_stats[name] = {
                    "status": "success",
                    "time": stats.get("time", 0),
                    "stats": key_stats
                }
                
        summary["pass_statistics"] = pass_stats
        
        return summary

# Singleton instance
_metal_optimizing_compiler = None

def get_metal_optimizing_compiler(optimization_level: OptimizationLevel = OptimizationLevel.STANDARD) -> MetalOptimizingCompiler:
    """
    Get Metal optimizing compiler
    
    Args:
        optimization_level: Level of optimization to apply
        
    Returns:
        Metal optimizing compiler
    """
    global _metal_optimizing_compiler
    if _metal_optimizing_compiler is None or _metal_optimizing_compiler.optimization_level != optimization_level:
        _metal_optimizing_compiler = MetalOptimizingCompiler(optimization_level)
    return _metal_optimizing_compiler

def optimize_for_metal(graph: Dict, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD) -> Tuple[Dict, Dict]:
    """
    Optimize computation graph for Metal
    
    Args:
        graph: Computation graph
        optimization_level: Level of optimization to apply
        
    Returns:
        Tuple of (optimized graph, optimization stats)
    """
    compiler = get_metal_optimizing_compiler(optimization_level)
    return compiler.compile(graph) 