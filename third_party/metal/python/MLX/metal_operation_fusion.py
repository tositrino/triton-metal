"""
Metal-specific Operation Fusion Optimizer for Triton Backend

This module implements specialized operation fusion optimizations for Metal GPUs,
focusing on identifying patterns that can be fused for improved performance on
Apple Silicon, particularly M3 chips with tensor cores.
"""


from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable, Pattern


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

    AppleSiliconGeneration = DummyEnum
    hardware_capabilities = DummyCapabilities()

# Try to import fusion optimizer components
try:
    from MLX.metal_fusion_optimizer import FusionPattern, FusionOptimizer
    has_fusion_optimizer = True
except ImportError:
    has_fusion_optimizer = False

    # Define minimal FusionPattern for standalone usage
    class FusionPattern:
        """Pattern of operations that can be fused"""

        def __init__(self, name: str, op_sequence: List[str], matcher_fn=None,
                    min_hardware_gen=None):
            """
            Initialize fusion pattern

            Args:
                name: Pattern name
                op_sequence: Sequence of operation types
                matcher_fn: Optional function to match operations
                min_hardware_gen: Minimum hardware generation required
            """
            self.name = name
            self.op_sequence = op_sequence
            self.matcher_fn = matcher_fn
            self.min_hardware_gen = min_hardware_gen

        def matches(self, ops: List[Dict], start_idx: int = 0) -> bool:
            """Check if operations match this pattern"""
            if start_idx + len(self.op_sequence) > len(ops):
                return False

            # Check operation sequence
            for i, op_type in enumerate(self.op_sequence):
                op_idx = start_idx + i
                op_name = ops[op_idx].get("type", "").split(".")[-1]

                if op_name != op_type:
                    return False

            # Apply custom matcher if provided
            if self.matcher_fn and not self.matcher_fn(ops[start_idx:start_idx + len(self.op_sequence)]):
                return False

            return True

class FusionRole(Enum):
    """Role of operation in fusion pattern"""
    UNKNOWN = 0
    INPUT = 1
    INTERMEDIATE = 2
    OUTPUT = 3
    EXECUTOR = 4

class MetalFusionPattern:
    """
    Metal-specific fusion pattern with additional optimizations
    """

    def __init__(self, name: str, op_types: List[str],
                min_hardware_gen: Optional[AppleSiliconGeneration] = None):
        """
        Initialize Metal fusion pattern

        Args:
            name: Pattern name
            op_types: List of operation types in the pattern
            min_hardware_gen: Minimum hardware generation required
        """
        self.name = name
        self.op_types = op_types
        self.min_hardware_gen = min_hardware_gen
        self.roles = self._assign_roles()

    def _assign_roles(self) -> Dict[int, FusionRole]:
        """
        Assign roles to operations in the pattern

        Returns:
            Dictionary mapping operation index to role
        """
        roles = {}

        # Default role assignment
        roles[0] = FusionRole.INPUT
        for i in range(1, len(self.op_types) - 1):
            roles[i] = FusionRole.INTERMEDIATE
        if len(self.op_types) > 1:
            roles[len(self.op_types) - 1] = FusionRole.OUTPUT

        return roles

    def matches(self, ops: List[Dict], start_idx: int) -> bool:
        """
        Check if operations match this pattern

        Args:
            ops: List of operations
            start_idx: Starting index to check from

        Returns:
            True if pattern matches, False otherwise
        """
        # Check if enough operations remain
        if start_idx + len(self.op_types) > len(ops):
            return False

        # Check each operation type
        for i, pattern_op_type in enumerate(self.op_types):
            op_idx = start_idx + i
            op_type = ops[op_idx].get("type", "").split(".")[-1]

            if op_type != pattern_op_type:
                return False

        return True

    def get_fused_op(self, ops: List[Dict], start_idx: int) -> Dict:
        """
        Create fused operation from matched operations

        Args:
            ops: List of operations
            start_idx: Starting index of matched pattern

        Returns:
            Fused operation
        """
        # Extract matched operations
        matched_ops = ops[start_idx:start_idx + len(self.op_types)]

        # Create base fused operation
        fused_op = {
            "type": f"fusion.{self.name}",
            "fused_ops": [op.copy() for op in matched_ops],
            "pattern_name": self.name,
            "input_indices": self._get_input_indices(matched_ops),
            "output_indices": self._get_output_indices(matched_ops)
        }

        # Add pattern-specific optimizations
        self._add_pattern_specific_optimizations(fused_op, matched_ops)

        return fused_op

    def _get_input_indices(self, ops: List[Dict]) -> List[int]:
        """
        Get indices of input operations

        Args:
            ops: List of operations

        Returns:
            List of input indices
        """
        input_indices = []

        # Find operations with INPUT role
        for i, role in self.roles.items():
            if role == FusionRole.INPUT and i < len(ops):
                input_indices.append(i)

        # If no inputs were explicitly defined, use the first operation
        if not input_indices and ops:
            input_indices.append(0)

        return input_indices

    def _get_output_indices(self, ops: List[Dict]) -> List[int]:
        """
        Get indices of output operations

        Args:
            ops: List of operations

        Returns:
            List of output indices
        """
        output_indices = []

        # Find operations with OUTPUT role
        for i, role in self.roles.items():
            if role == FusionRole.OUTPUT and i < len(ops):
                output_indices.append(i)

        # If no outputs were explicitly defined, use the last operation
        if not output_indices and ops:
            output_indices.append(len(ops) - 1)

        return output_indices

    def _add_pattern_specific_optimizations(self, fused_op: Dict, ops: List[Dict]) -> None:
        """
        Add pattern-specific optimizations to fused operation

        Args:
            fused_op: Fused operation to modify
            ops: Original operations
        """
        # Base implementation does nothing, to be overridden by subclasses
        pass

    def is_applicable_to_hardware(self, hardware_gen: Optional[Any] = None) -> bool:
        """
        Check if this pattern is applicable to the given hardware

        Args:
            hardware_gen: Hardware generation to check against

        Returns:
            True if applicable, False otherwise
        """
        if self.min_hardware_gen is None:
            return True

        if hardware_gen is None:
            return False

        return hardware_gen.value >= self.min_hardware_gen.value

class ElementwiseFusionPattern(MetalFusionPattern):
    """Pattern for fusing elementwise operations"""

    def __init__(self):
        """Initialize elementwise fusion pattern"""
        super().__init__("elementwise_fusion", ["add", "mul"], None)

    def matches(self, ops: List[Dict], start_idx: int) -> bool:
        """Check if operations can be fused as elementwise"""
        if not super().matches(ops, start_idx):
            return False

        # Check that operations are elementwise
        elementwise_types = {"add", "sub", "mul", "div", "max", "min", "pow"}

        for i in range(len(self.op_types)):
            op_idx = start_idx + i
            op_type = ops[op_idx].get("type", "").split(".")[-1]

            if op_type not in elementwise_types:
                return False

        return True

    def _add_pattern_specific_optimizations(self, fused_op: Dict, ops: List[Dict]) -> None:
        """Add elementwise-specific optimizations"""
        fused_op["fusion_strategy"] = "elementwise"
        fused_op["vectorize"] = True

        # Set optimal vector width based on hardware
        hardware_gen = getattr(hardware_capabilities, "chip_generation", None)
        if hardware_gen == AppleSiliconGeneration.M3:
            fused_op["vector_width"] = 8
        else:
            fused_op["vector_width"] = 4

class MatMulAddFusionPattern(MetalFusionPattern):
    """Pattern for fusing matrix multiplication with add (GEMM)"""

    def __init__(self):
        """Initialize matmul-add fusion pattern"""
        super().__init__("matmul_add_fusion", ["matmul", "add"], None)

    def _add_pattern_specific_optimizations(self, fused_op: Dict, ops: List[Dict]) -> None:
        """Add matmul-specific optimizations"""
        fused_op["fusion_strategy"] = "matmul_add"

        # Extract shapes from matmul operation if available
        if ops and "input_shapes" in ops[0] and len(ops[0]["input_shapes"]) >= 2:
            m, k = ops[0]["input_shapes"][0]
            k2, n = ops[0]["input_shapes"][1]

            # Ensure k dimensions match
            if k == k2:
                fused_op["matmul_dims"] = {"m": m, "n": n, "k": k}

        # Enable tensor cores for M3 if applicable
        hardware_gen = getattr(hardware_capabilities, "chip_generation", None)
        if hardware_gen == AppleSiliconGeneration.M3:
            fused_op["use_tensor_cores"] = True
            fused_op["block_size"] = 128
        else:
            fused_op["use_tensor_cores"] = False
            fused_op["block_size"] = 64

class SoftmaxFusionPattern(MetalFusionPattern):
    """Pattern for fusing softmax-related operations"""

    def __init__(self):
        """Initialize softmax fusion pattern"""
        super().__init__("softmax_fusion", ["sub", "exp", "sum", "div"], None)

    def _add_pattern_specific_optimizations(self, fused_op: Dict, ops: List[Dict]) -> None:
        """Add softmax-specific optimizations"""
        fused_op["fusion_strategy"] = "softmax"

        # Enable softmax-specific optimizations for M3
        hardware_gen = getattr(hardware_capabilities, "chip_generation", None)
        if hardware_gen == AppleSiliconGeneration.M3:
            fused_op["use_fast_math"] = True
            fused_op["use_simdgroup_reduction"] = True
            fused_op["shared_memory_size"] = 65536  # 64KB
        else:
            fused_op["use_fast_math"] = True
            fused_op["use_simdgroup_reduction"] = False
            fused_op["shared_memory_size"] = 32768  # 32KB

class FlashAttentionFusionPattern(MetalFusionPattern):
    """Pattern for fusing flash attention operations (M3-specific)"""

    def __init__(self):
        """Initialize flash attention fusion pattern"""
        super().__init__(
            "flash_attention_fusion",
            ["matmul", "div", "softmax", "matmul"],
            AppleSiliconGeneration.M3  # M3-specific optimization
        )

    def _add_pattern_specific_optimizations(self, fused_op: Dict, ops: List[Dict]) -> None:
        """Add flash attention optimizations"""
        fused_op["fusion_strategy"] = "flash_attention"
        fused_op["use_tensor_cores"] = True
        fused_op["block_size"] = 128
        fused_op["use_hierarchical_softmax"] = True
        fused_op["shared_memory_size"] = 65536  # 64KB
        fused_op["use_causal_mask_optimization"] = True
        fused_op["implementation"] = "flash_attention_v2"

class GeluFusionPattern(MetalFusionPattern):
    """Pattern for fusing GELU activation operations"""

    def __init__(self):
        """Initialize GELU fusion pattern"""
        super().__init__("gelu_fusion", ["mul", "pow", "mul", "add", "mul", "tanh", "add", "mul"], None)

    def _add_pattern_specific_optimizations(self, fused_op: Dict, ops: List[Dict]) -> None:
        """Add GELU-specific optimizations"""
        fused_op["fusion_strategy"] = "gelu"
        fused_op["vectorize"] = True

        # Set optimal vector width based on hardware
        hardware_gen = getattr(hardware_capabilities, "chip_generation", None)
        if hardware_gen == AppleSiliconGeneration.M3:
            fused_op["vector_width"] = 8
            fused_op["use_fast_math"] = True
            fused_op["use_approximation"] = True
        else:
            fused_op["vector_width"] = 4
            fused_op["use_fast_math"] = True
            fused_op["use_approximation"] = True

class SwiGLUFusionPattern(MetalFusionPattern):
    """Pattern for fusing SwiGLU activation operations (M3-specific)"""

    def __init__(self):
        """Initialize SwiGLU fusion pattern"""
        super().__init__(
            "swiglu_fusion",
            ["mul", "sigmoid", "mul"],
            AppleSiliconGeneration.M3  # M3-specific optimization
        )

    def _add_pattern_specific_optimizations(self, fused_op: Dict, ops: List[Dict]) -> None:
        """Add SwiGLU-specific optimizations"""
        fused_op["fusion_strategy"] = "swiglu"
        fused_op["vectorize"] = True
        fused_op["vector_width"] = 8
        fused_op["use_fast_sigmoid"] = True
        fused_op["use_tensor_cores"] = True
        fused_op["implementation"] = "native_swiglu"

class MetalOperationFusionOptimizer:
    """
    Optimizer for fusing operations on Metal GPUs
    """

    def __init__(self):
        """Initialize Metal operation fusion optimizer"""
        # Detect hardware generation
        self.hardware_gen = self._detect_hardware_generation()

        # Create fusion patterns
        self.patterns = self._create_fusion_patterns()

        # Track statistics
        self.stats = {
            "fused_ops": 0,
            "fusion_patterns": {},
            "hardware_specific_fusions": 0
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

    def _create_fusion_patterns(self) -> List[MetalFusionPattern]:
        """
        Create fusion patterns

        Returns:
            List of fusion patterns
        """
        patterns = []

        # Add basic patterns
        patterns.append(ElementwiseFusionPattern())
        patterns.append(MatMulAddFusionPattern())
        patterns.append(SoftmaxFusionPattern())
        patterns.append(GeluFusionPattern())

        # Add M3-specific patterns if applicable
        if self.hardware_gen == AppleSiliconGeneration.M3:
            patterns.append(FlashAttentionFusionPattern())
            patterns.append(SwiGLUFusionPattern())

        return patterns

    def optimize(self, graph: Dict) -> Tuple[Dict, Dict]:
        """
        Optimize graph by fusing operations

        Args:
            graph: Computation graph

        Returns:
            Tuple of (optimized graph, optimization stats)
        """
        if not graph or "ops" not in graph:
            return graph, self.stats

        # Reset statistics
        self.stats = {
            "fused_ops": 0,
            "fusion_patterns": {},
            "hardware_specific_fusions": 0
        }

        # Get operations from graph
        ops = graph["ops"]

        # Find and apply fusion optimizations
        optimized_ops = self._optimize_operations(ops)

        # Create optimized graph
        optimized_graph = graph.copy()
        optimized_graph["ops"] = optimized_ops

        # Add metadata
        if "metadata" not in optimized_graph:
            optimized_graph["metadata"] = {}

        optimized_graph["metadata"]["operation_fusion_optimized"] = True

        if self.hardware_gen == AppleSiliconGeneration.M3:
            optimized_graph["metadata"]["m3_fusion_optimized"] = True

        # Add fusion statistics to metadata
        optimized_graph["metadata"]["fusion_stats"] = self.stats.copy()

        return optimized_graph, self.stats

    def _optimize_operations(self, ops: List[Dict]) -> List[Dict]:
        """
        Optimize operations by applying fusion patterns

        Args:
            ops: List of operations

        Returns:
            Optimized operations
        """
        if not ops:
            return []

        optimized_ops = []
        i = 0

        while i < len(ops):
            # Try to match each pattern
            matched = False

            for pattern in self.patterns:
                # Skip patterns that aren't applicable to this hardware
                if not pattern.is_applicable_to_hardware(self.hardware_gen):
                    continue

                # Check if pattern matches
                if pattern.matches(ops, i):
                    # Pattern matched, create fused operation
                    fused_op = pattern.get_fused_op(ops, i)

                    # Add to optimized operations
                    optimized_ops.append(fused_op)

                    # Update statistics
                    self.stats["fused_ops"] += len(pattern.op_types) - 1
                    self.stats["fusion_patterns"][pattern.name] = self.stats["fusion_patterns"].get(pattern.name, 0) + 1

                    if pattern.min_hardware_gen == AppleSiliconGeneration.M3:
                        self.stats["hardware_specific_fusions"] += 1

                    # Skip fused operations
                    i += len(pattern.op_types)
                    matched = True
                    break

            # If no pattern matched, keep the original operation
            if not matched:
                optimized_ops.append(ops[i])
                i += 1

        return optimized_ops

    def apply_to_mlx_fusion_optimizer(self, fusion_optimizer: Any) -> Any:
        """
        Apply Metal-specific patterns to an existing MLX fusion optimizer

        Args:
            fusion_optimizer: MLX fusion optimizer to extend

        Returns:
            Extended fusion optimizer
        """
        if not has_fusion_optimizer:
            return fusion_optimizer

        # Convert Metal patterns to MLX fusion patterns
        for pattern in self.patterns:
            # Skip patterns that aren't applicable to this hardware
            if not pattern.is_applicable_to_hardware(self.hardware_gen):
                continue

            # Create matcher function
            def create_matcher_fn(metal_pattern):
                def matcher_fn(ops):
                    return metal_pattern.matches(ops, 0)
                return matcher_fn

            # Create MLX fusion pattern
            mlx_pattern = FusionPattern(
                name=pattern.name,
                op_pattern=pattern.op_types,
                min_hardware_gen=pattern.min_hardware_gen,
                pattern_matcher=create_matcher_fn(pattern)
            )

            # Add to fusion optimizer
            fusion_optimizer.patterns.append(mlx_pattern)

        return fusion_optimizer

# Singleton instance
_metal_fusion_optimizer = None

def get_metal_fusion_optimizer() -> MetalOperationFusionOptimizer:
    """
    Get Metal operation fusion optimizer

    Returns:
        Metal operation fusion optimizer
    """
    global _metal_fusion_optimizer
    if _metal_fusion_optimizer is None:
        _metal_fusion_optimizer = MetalOperationFusionOptimizer()
    return _metal_fusion_optimizer

def optimize_operation_fusion(graph: Dict) -> Tuple[Dict, Dict]:
    """
    Optimize operation fusion in graph

    Args:
        graph: Computation graph

    Returns:
        Tuple of (optimized graph, optimization stats)
    """
    optimizer = get_metal_fusion_optimizer()
    return optimizer.optimize(graph)