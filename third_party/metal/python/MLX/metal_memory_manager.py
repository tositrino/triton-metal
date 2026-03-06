"""
Metal Memory Manager for Triton Metal Backend

This module provides memory management and optimization strategies specifically
tailored for Metal backend execution on Apple Silicon GPUs.
"""

from enum import Enum
from typing import Dict, Any



# Try to import hardware detection
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
            self.shared_memory_size = 32768  # Default to 32KB

    AppleSiliconGeneration = DummyEnum
    hardware_capabilities = DummyCapabilities()

class TensorType(Enum):
    """Enum for tensor types"""
    UNKNOWN = 0
    MATRIX = 1
    VECTOR = 2
    SCALAR = 3
    CONVOLUTION_FILTER = 4
    FEATURE_MAP = 5
    ATTENTION = 6

class MemoryLayout(Enum):
    """Enum for memory layouts"""
    DEFAULT = 0
    ROW_MAJOR = 1
    COLUMN_MAJOR = 2
    BLOCK_BASED = 3
    TILED = 4
    INTERLEAVED = 5
    SIMD_ALIGNED = 6
    TEXTURE_OPTIMIZED = 7
    COALESCED = 8

class MemoryAccessPattern(Enum):
    """Enum for memory access patterns"""
    SEQUENTIAL = 0
    STRIDED = 1
    RANDOM = 2
    COALESCED = 3
    BROADCAST = 4
    GATHER = 5
    SCATTER = 6

class MetalMemoryManager:
    """Memory management for Metal backend"""

    def __init__(self):
        """Initialize memory manager"""
        self.is_m3 = self._is_m3_hardware()

        # Set shared memory size based on hardware
        if self.is_m3:
            self.shared_memory_size = 65536  # 64KB for M3
        else:
            # Default to 32KB for M1/M2
            self.shared_memory_size = getattr(hardware_capabilities, "shared_memory_size", 32768)

        # Configure optimal tile sizes based on hardware
        self.tile_sizes = self._configure_tile_sizes()

        # Configure vector widths based on hardware
        self.vector_width = 8 if self.is_m3 else 4

        # SIMD group width
        self.simd_width = 32

    def _is_m3_hardware(self) -> bool:
        """
        Check if we're running on M3 hardware

        Returns:
            True if running on M3, False otherwise
        """
        return hasattr(hardware_capabilities, "chip_generation") and \
               hardware_capabilities.chip_generation == AppleSiliconGeneration.M3

    def _configure_tile_sizes(self) -> Dict[TensorType, Dict[str, Any]]:
        """
        Configure optimal tile sizes based on hardware

        Returns:
            Dictionary of tile sizes for different tensor types
        """
        if self.is_m3:
            # M3-specific tile sizes (larger due to 64KB shared memory)
            return {
                TensorType.MATRIX: {
                    "default": (128, 128),
                    "large": (128, 256),
                    "small": (64, 64)
                },
                TensorType.CONVOLUTION_FILTER: {
                    "default": (128, 64, 3),
                    "large": (128, 128, 3),
                    "small": (64, 64, 3)
                },
                TensorType.FEATURE_MAP: {
                    "default": (64, 64, 4),
                    "large": (128, 128, 4),
                    "small": (32, 32, 4)
                },
                TensorType.ATTENTION: {
                    "default": (64, 64),
                    "large": (128, 128),
                    "small": (32, 32)
                }
            }
        else:
            # M1/M2 tile sizes (smaller due to 32KB shared memory)
            return {
                TensorType.MATRIX: {
                    "default": (64, 64),
                    "large": (128, 128),
                    "small": (32, 32)
                },
                TensorType.CONVOLUTION_FILTER: {
                    "default": (64, 32, 3),
                    "large": (64, 64, 3),
                    "small": (32, 32, 3)
                },
                TensorType.FEATURE_MAP: {
                    "default": (32, 32, 4),
                    "large": (64, 64, 4),
                    "small": (16, 16, 4)
                },
                TensorType.ATTENTION: {
                    "default": (32, 32),
                    "large": (64, 64),
                    "small": (16, 16)
                }
            }

    def optimize_graph_memory(self, graph: Dict) -> Dict:
        """
        Optimize memory layout and access patterns for all operations in the graph

        Args:
            graph: MLX computation graph

        Returns:
            Optimized graph
        """
        if not graph or "ops" not in graph:
            return graph

        optimized_ops = []

        for op in graph["ops"]:
            optimized_op = self._optimize_operation_memory(op)
            optimized_ops.append(optimized_op)

        # Update graph with optimized operations
        optimized_graph = graph.copy()
        optimized_graph["ops"] = optimized_ops

        return optimized_graph

    def _optimize_operation_memory(self, op: Dict) -> Dict:
        """
        Optimize memory layout and access patterns for a specific operation

        Args:
            op: Operation to optimize

        Returns:
            Optimized operation
        """
        # Start with a copy of the operation
        optimized_op = op.copy()

        # Identify operation type
        op_type = op.get("type", "").split(".")[-1]

        # Dispatch to specific optimization functions based on operation type
        if op_type in ["matmul", "mm"]:
            return self._optimize_matmul_memory(optimized_op)
        elif op_type in ["conv1d", "conv2d", "conv3d"]:
            return self._optimize_convolution_memory(optimized_op)
        elif op_type in ["reduce", "sum", "mean", "max", "min"]:
            return self._optimize_reduction_memory(optimized_op)
        elif op_type in ["transpose", "permute"]:
            return self._optimize_transpose_memory(optimized_op)
        elif op_type in ["softmax", "attention"]:
            return self._optimize_attention_memory(optimized_op)
        elif op_type in ["add", "sub", "mul", "div", "pow"]:
            return self._optimize_elementwise_memory(optimized_op)
        else:
            # For other operations, apply general memory optimizations
            return self._apply_general_memory_optimizations(optimized_op)

    def _optimize_matmul_memory(self, op: Dict) -> Dict:
        """
        Optimize memory layout for matrix multiplication

        Args:
            op: Matrix multiplication operation

        Returns:
            Optimized operation
        """
        # Identify input shapes
        input_shapes = op.get("input_shapes", [])

        if len(input_shapes) >= 2:
            m, k = input_shapes[0]
            k2, n = input_shapes[1]

            # Ensure k dimensions match
            if k != k2:
                return op

            # Determine tile size based on matrix dimensions
            tile_size_key = "large" if m >= 1024 and n >= 1024 else \
                "small" if m <= 128 and n <= 128 else \
                "default"

            tile_m, tile_n = self.tile_sizes[TensorType.MATRIX][tile_size_key]

            # For M3, enable tensor core usage if applicable
            use_tensor_cores = self.is_m3 and min(m, n, k) >= 16

            # Calculate optimal threadgroup size
            optimal_threadgroup_size = min(tile_m * tile_n // 8, 1024)

            # Set execution parameters
            if "execution_parameters" not in op:
                op["execution_parameters"] = {}

            op["execution_parameters"].update({
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": 32,  # Typical k tile size
                "memory_layout": MemoryLayout.BLOCK_BASED.value,
                "vector_width": self.vector_width,
                "simd_width": self.simd_width,
                "threadgroup_size": optimal_threadgroup_size,
                "use_tensor_cores": use_tensor_cores,
                "shared_memory_size": self.shared_memory_size
            })

            # M3-specific optimizations
            if self.is_m3:
                op["execution_parameters"].update({
                    "use_hierarchical_reduction": True,
                    "use_dynamic_shared_memory": True,
                    "simdgroup_matrix_size": 16,  # Optimal for M3 tensor cores
                    "prefetch_distance": 2  # Prefetch 2 tiles ahead for M3's higher bandwidth
                })

        return op

    def _optimize_convolution_memory(self, op: Dict) -> Dict:
        """
        Optimize memory layout for convolution operations

        Args:
            op: Convolution operation

        Returns:
            Optimized operation
        """
        # Determine convolution dimensions from operation type
        op_type = op.get("type", "").split(".")[-1]

        # Get input shapes
        input_shapes = op.get("input_shapes", [])

        if len(input_shapes) >= 2:
            # Input shape for convolution is typically [N, C, H, W] or [N, C, D, H, W]
            input_shape = input_shapes[0]
            filter_shape = input_shapes[1]

            # Determine tensor type for tile size lookup
            tensor_type = TensorType.CONVOLUTION_FILTER

            # Determine tile size based on feature map dimensions
            if len(input_shape) >= 4:
                feature_map_size = input_shape[2] * input_shape[3]  # H * W
                tile_size_key = "large" if feature_map_size >= 16384 else \
                    "small" if feature_map_size <= 1024 else \
                    "default"
            else:
                tile_size_key = "default"

            # Get optimal tile size
            if op_type == "conv1d":
                tile_size = (self.tile_sizes[tensor_type][tile_size_key][0],)
            elif op_type == "conv2d":
                tile_size = self.tile_sizes[tensor_type][tile_size_key][:2]
            else:  # conv3d
                tile_size = self.tile_sizes[tensor_type][tile_size_key]

            # Set execution parameters
            if "execution_parameters" not in op:
                op["execution_parameters"] = {}

            op["execution_parameters"].update({
                "tile_size": tile_size,
                "memory_layout": MemoryLayout.TEXTURE_OPTIMIZED.value,
                "vector_width": self.vector_width,
                "simd_width": self.simd_width,
                "threadgroup_size": 256,  # Typical value for convolutions
                "shared_memory_size": self.shared_memory_size
            })

            # M3-specific optimizations
            if self.is_m3:
                op["execution_parameters"].update({
                    "use_texture_for_weights": True,  # M3 has improved texture support
                    "use_warp_specialization": True,  # M3 supports specialized warps
                    "prefetch_mode": "double_buffer",  # Double buffering for M3
                    "use_simdgroup_reduction": True    # Use SIMD groups for reduction
                })

        return op

    def _optimize_reduction_memory(self, op: Dict) -> Dict:
        """
        Optimize memory layout for reduction operations

        Args:
            op: Reduction operation

        Returns:
            Optimized operation
        """
        # Get input shapes
        input_shapes = op.get("input_shapes", [])

        if input_shapes:
            input_shape = input_shapes[0]

            # Get reduction axes
            reduction_axes = op.get("args", {}).get("axis", [])
            if not isinstance(reduction_axes, list):
                reduction_axes = [reduction_axes]

            # Calculate reduction size
            reduction_size = 1
            for axis in reduction_axes:
                if axis < len(input_shape):
                    reduction_size *= input_shape[axis]

            # Determine optimal threadgroup size based on reduction size
            if reduction_size <= 256:
                threadgroup_size = 256
                use_hierarchical = False
            elif reduction_size <= 1024:
                threadgroup_size = 256
                use_hierarchical = True
            else:
                threadgroup_size = 1024
                use_hierarchical = True

            # Set execution parameters
            if "execution_parameters" not in op:
                op["execution_parameters"] = {}

            op["execution_parameters"].update({
                "memory_layout": MemoryLayout.COALESCED.value,
                "vector_width": self.vector_width,
                "simd_width": self.simd_width,
                "threadgroup_size": threadgroup_size,
                "use_hierarchical_reduction": use_hierarchical,
                "shared_memory_size": self.shared_memory_size
            })

            # M3-specific optimizations
            if self.is_m3:
                op["execution_parameters"].update({
                    "two_stage_reduction": True,  # Two-stage reduction for M3
                    "use_simdgroup_reduction": True,  # SIMD group reduction for M3
                    "vector_width": 8,  # 8-wide vectors for M3
                })

        return op

    def _optimize_transpose_memory(self, op: Dict) -> Dict:
        """
        Optimize memory layout for transpose operations

        Args:
            op: Transpose operation

        Returns:
            Optimized operation
        """
        # Get input shapes
        input_shapes = op.get("input_shapes", [])

        if input_shapes:
            input_shape = input_shapes[0]

            # Determine if this is a matrix transpose
            is_matrix = len(input_shape) == 2

            # Determine memory layout based on operation
            memory_layout = MemoryLayout.TILED.value if is_matrix else MemoryLayout.DEFAULT.value

            # Set execution parameters
            if "execution_parameters" not in op:
                op["execution_parameters"] = {}

            op["execution_parameters"].update({
                "memory_layout": memory_layout,
                "vector_width": self.vector_width,
                "simd_width": self.simd_width,
                "threadgroup_size": 256,
                "shared_memory_size": self.shared_memory_size
            })

            # M3-specific optimizations
            if self.is_m3 and is_matrix:
                # For matrix transpose on M3, use 32x32 tiles with 8-wide vectors
                tile_size = 32
                op["execution_parameters"].update({
                    "tile_size": tile_size,
                    "vector_width": 8,
                    "use_simdgroup_matrix": True,  # Use SIMD group for matrix transpose
                })

        return op

    def _optimize_attention_memory(self, op: Dict) -> Dict:
        """
        Optimize memory layout for attention operations

        Args:
            op: Attention operation

        Returns:
            Optimized operation
        """
        # Get input shapes
        input_shapes = op.get("input_shapes", [])

        if input_shapes:
            # Determine attention dimensions
            batch_size = input_shapes[0][0] if len(input_shapes[0]) > 0 else 1
            seq_length = input_shapes[0][1] if len(input_shapes[0]) > 1 else 512  # Default

            # Determine tile size key based on sequence length
            tile_size_key = "large" if seq_length >= 1024 else \
                          "small" if seq_length <= 128 else \
                          "default"

            # Get optimal tile size
            tile_size = self.tile_sizes[TensorType.ATTENTION][tile_size_key]

            # Calculate optimal threadgroup size
            optimal_threadgroup_size = min(tile_size[0] * tile_size[1] // 4, 1024)

            # Set execution parameters
            if "execution_parameters" not in op:
                op["execution_parameters"] = {}

            op["execution_parameters"].update({
                "tile_size": tile_size,
                "memory_layout": MemoryLayout.BLOCK_BASED.value,
                "vector_width": self.vector_width,
                "simd_width": self.simd_width,
                "threadgroup_size": optimal_threadgroup_size,
                "shared_memory_size": self.shared_memory_size
            })

            # M3-specific optimizations
            if self.is_m3:
                op["execution_parameters"].update({
                    "use_flash_attention": True,  # Flash attention for M3
                    "use_tensor_cores": True,     # Tensor cores for matrix multiplications
                    "block_size": 128,            # Larger blocks for M3
                    "use_causal_mask_optimization": True,  # Causal mask optimization
                })

        return op

    def _optimize_elementwise_memory(self, op: Dict) -> Dict:
        """
        Optimize memory layout for elementwise operations

        Args:
            op: Elementwise operation

        Returns:
            Optimized operation
        """
        # Get input shapes to determine vector width
        input_shapes = op.get("input_shapes", [])

        if input_shapes:
            # Check if any dimension is divisible by vector width
            can_vectorize = False
            for shape in input_shapes:
                if any(dim % self.vector_width == 0 for dim in shape):
                    can_vectorize = True
                    break

            # Set execution parameters
            if "execution_parameters" not in op:
                op["execution_parameters"] = {}

            op["execution_parameters"].update({
                "memory_layout": MemoryLayout.COALESCED.value,
                "vector_width": self.vector_width if can_vectorize else 1,
                "simd_width": self.simd_width,
                "threadgroup_size": 256,
                "shared_memory_size": self.shared_memory_size
            })

            # M3-specific optimizations
            if self.is_m3 and can_vectorize:
                op["execution_parameters"].update({
                    "vector_width": 8,  # 8-wide vectors for M3
                    "unroll_factor": 4   # Unroll by 4 for M3
                })

        return op

    def _apply_general_memory_optimizations(self, op: Dict) -> Dict:
        """
        Apply general memory optimizations for any operation

        Args:
            op: Operation to optimize

        Returns:
            Optimized operation
        """
        # Set basic execution parameters
        if "execution_parameters" not in op:
            op["execution_parameters"] = {}

        op["execution_parameters"].update({
            "vector_width": self.vector_width,
            "simd_width": self.simd_width,
            "threadgroup_size": 256,
            "shared_memory_size": self.shared_memory_size
        })

        # M3-specific optimizations
        if self.is_m3:
            op["execution_parameters"].update({
                "memory_optimization_level": 2,  # Higher level for M3
                "use_dynamic_shared_memory": True  # Dynamic shared memory for M3
            })

        return op

# Singleton instance
_metal_memory_manager = None

def get_metal_memory_manager() -> MetalMemoryManager:
    """
    Get the Metal memory manager instance

    Returns:
        MetalMemoryManager instance
    """
    global _metal_memory_manager

    if _metal_memory_manager is None:
        _metal_memory_manager = MetalMemoryManager()

    return _metal_memory_manager