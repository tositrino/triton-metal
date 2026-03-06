"""
Operation mapping for Triton to MLX conversion

This module provides detailed mapping of Triton operations to their MLX equivalents,
with special optimizations for Apple Silicon GPUs.
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from enum import Enum

# Define missing operations that may not be in the current MLX version
def _polyfill_dot(a, b):
    """Polyfill for dot operation if not available in MLX"""
    return mx.matmul(a, b)

class OpCategory(Enum):
    """Categories of operations for organization"""
    ELEMENTWISE = 1
    REDUCTION = 2
    MEMORY = 3
    MATH = 4
    MATRIX = 5
    TENSOR = 6
    CONTROL = 7
    SYNCHRONIZATION = 8

class MLXDispatcher:
    """
    MLX operation dispatcher with Apple Silicon optimizations
    """

    def __init__(self, hardware_capabilities=None):
        """
        Initialize MLX dispatcher with hardware capabilities

        Args:
            hardware_capabilities: Optional hardware capabilities instance
        """
        self.hardware = hardware_capabilities or self._get_default_hardware_capabilities()
        self.cached_ops = {}
        self.init_op_mappings()

    def _get_default_hardware_capabilities(self):
        """Get default hardware capabilities"""
        from .metal_hardware_optimizer import hardware_capabilities
        return hardware_capabilities

    def init_op_mappings(self):
        """Initialize operation mappings for all categories"""
        # Check for available operations in MLX
        has_dot = hasattr(mx, "dot")
        has_prod = hasattr(mx, "prod")
        has_var = hasattr(mx, "var")
        has_std = hasattr(mx, "std")
        has_all = hasattr(mx, "all")
        has_any = hasattr(mx, "any")

        # Base MLX operations
        self.base_ops = {
            # Elementwise binary operations
            "add": mx.add,
            "sub": mx.subtract,
            "mul": mx.multiply,
            "div": mx.divide,
            "mod": mx.remainder if hasattr(mx, "remainder") else lambda x, y: x % y,
            "pow": mx.power,
            "max": mx.maximum,
            "min": mx.minimum,

            # Logical operations
            "and": mx.logical_and if hasattr(mx, "logical_and") else lambda x, y: x & y,
            "or": mx.logical_or if hasattr(mx, "logical_or") else lambda x, y: x | y,
            "xor": lambda x, y: mx.logical_or(
                mx.logical_and(x, mx.logical_not(y)),
                mx.logical_and(mx.logical_not(x), y)
            ) if hasattr(mx, "logical_or") else lambda x, y: x ^ y,

            # Elementwise unary operations
            "exp": mx.exp,
            "log": mx.log,
            "sqrt": mx.sqrt,
            "rsqrt": lambda x: 1.0 / mx.sqrt(x),
            "sin": mx.sin,
            "cos": mx.cos,
            "tan": mx.tan if hasattr(mx, "tan") else lambda x: mx.sin(x) / mx.cos(x),
            "tanh": mx.tanh,
            "sigmoid": mx.sigmoid,
            "abs": mx.abs,
            "neg": lambda x: -x,
            "not": mx.logical_not if hasattr(mx, "logical_not") else lambda x: ~x,
            "bitwise_not": lambda x: ~x,

            # Comparison operations
            "eq": mx.equal,
            "ne": mx.not_equal,
            "lt": mx.less,
            "le": mx.less_equal,
            "gt": mx.greater,
            "ge": mx.greater_equal,

            # Reduction operations
            "sum": mx.sum,
            "prod": mx.prod if has_prod else lambda x, dims: np.prod(x, axis=dims),
            "mean": mx.mean,
            "var": mx.var if has_var else lambda x, dims: np.var(x, axis=dims),
            "std": mx.std if has_std else lambda x, dims: np.std(x, axis=dims),
            "max_reduce": mx.max,
            "min_reduce": mx.min,
            "all_reduce": (lambda x, dims: mx.all(mx.array(x, dtype=mx.bool_), axis=dims)) if has_all
                          else lambda x, dims: np.all(x, axis=dims),
            "any_reduce": (lambda x, dims: mx.any(mx.array(x, dtype=mx.bool_), axis=dims)) if has_any
                          else lambda x, dims: np.any(x, axis=dims),

            # Tensor operations
            "reshape": mx.reshape,
            "transpose": mx.transpose,
            "broadcast": mx.broadcast_to if hasattr(mx, "broadcast_to") else lambda x, shape: np.broadcast_to(x, shape),
            "concat": mx.concatenate,
            "split": mx.split,
            "stack": mx.stack,
            "slice": lambda x, start, end: mx.slice(x, start, end) if hasattr(mx, "slice") else x[start:end],

            # Matrix operations
            "matmul": mx.matmul,
            "dot": mx.dot if has_dot else _polyfill_dot,

            # Conversion operations
            "cast": mx.array,

            # Selection operations
            "where": mx.where,
        }

        # Specialized optimized operations based on hardware
        self.optimized_ops = self._create_optimized_ops()

        # Combined operations (optimized take precedence)
        self.ops = {**self.base_ops, **self.optimized_ops}

    def _create_optimized_ops(self) -> Dict[str, Callable]:
        """
        Create optimized operations based on hardware capabilities

        Returns:
            Dictionary of optimized operations
        """
        optimized = {}

        # Check if we have hardware information
        if not self.hardware:
            # Use safe defaults without hardware info
            optimized["fma"] = mx.addmm if hasattr(mx, "addmm") else lambda a, b, c=None, alpha=1.0, beta=1.0: a * b + (c * beta if c is not None else 0)
            optimized["softmax"] = mx.softmax
            return optimized

        # Add hardware-specific optimizations
        # For now, use the same optimizations regardless of hardware
        # Fused multiply-add (FMA) optimizations
        optimized["fma"] = mx.addmm if hasattr(mx, "addmm") else lambda a, b, c=None, alpha=1.0, beta=1.0: a * b + (c * beta if c is not None else 0)

        # Optimized softmax
        optimized["softmax"] = mx.softmax

        # Optimized attention for Apple Silicon using MLX's transformer_block
        optimized["attention"] = self._optimized_attention

        # Optimized GEMM for Apple Silicon (using MLX's implementation)
        optimized["gemm"] = self._optimized_gemm

        # Optimized conv for Apple Silicon (using MLX's implementation)
        optimized["conv"] = mx.conv if hasattr(mx, "conv") else None

        return optimized

    def _optimized_attention(self, q, k, v, scale=None, mask=None, dropout_p=0.0):
        """
        Optimized attention operation for Apple Silicon

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            scale: Scale factor (default: 1/sqrt(d_k))
            mask: Attention mask (default: None)
            dropout_p: Dropout probability (default: 0.0)

        Returns:
            Attention output
        """
        # Implement manually
        d_k = q.shape[-1]
        if scale is None:
            scale = 1.0 / (d_k ** 0.5)

        # Handle different tensor ranks
        if len(q.shape) == 3:
            # For 3D tensors (batch, seq_len, dim)
            # Compute scaled dot product
            scores = mx.matmul(q, mx.transpose(k, axes=(0, 2, 1))) * scale

            # Apply mask if provided
            if mask is not None:
                scores = mx.where(mask, scores, mx.full_like(scores, -1e9))

            # Apply softmax
            attn = mx.softmax(scores, axis=-1)

            # Apply dropout if available and necessary
            if dropout_p > 0.0 and hasattr(mx, "dropout"):
                attn = mx.dropout(attn, dropout_p)

            # Apply attention to values
            return mx.matmul(attn, v)
        else:
            # For 2D tensors (seq_len, dim)
            # Compute scaled dot product
            scores = mx.matmul(q, mx.transpose(k)) * scale

            # Apply mask if provided
            if mask is not None:
                scores = mx.where(mask, scores, mx.full_like(scores, -1e9))

            # Apply softmax
            attn = mx.softmax(scores, axis=-1)

            # Apply dropout if available and necessary
            if dropout_p > 0.0 and hasattr(mx, "dropout"):
                attn = mx.dropout(attn, dropout_p)

            # Apply attention to values
            return mx.matmul(attn, v)

    def _optimized_gemm(self, a, b, c=None, alpha=1.0, beta=0.0,
                       trans_a=False, trans_b=False):
        """
        Optimized General Matrix Multiply (GEMM) for Apple Silicon

        Args:
            a: First matrix
            b: Second matrix
            c: Optional bias matrix
            alpha: Scalar for a*b
            beta: Scalar for c
            trans_a: Transpose first matrix
            trans_b: Transpose second matrix

        Returns:
            GEMM result
        """
        # Handle transposes
        if trans_a:
            a = mx.transpose(a)
        if trans_b:
            b = mx.transpose(b)

        # Compute matrix multiply
        result = mx.matmul(a, b)

        # Apply alpha scaling
        if alpha != 1.0:
            result = result * alpha

        # Add bias if provided
        if c is not None and beta != 0.0:
            result = result + beta * c

        return result

    def get_op(self, op_name: str, category: OpCategory = None) -> Optional[Callable]:
        """
        Get operation by name and category

        Args:
            op_name: Operation name
            category: Optional operation category

        Returns:
            Operation callable or None if not found
        """
        # If we have a cached optimized version, return it
        cache_key = f"{op_name}_{category.name if category else 'None'}"
        if cache_key in self.cached_ops:
            return self.cached_ops[cache_key]

        # Look up the operation
        op = self.ops.get(op_name)

        # Cache the result for future use
        self.cached_ops[cache_key] = op

        return op

    def map_triton_op(self, triton_op: str) -> Tuple[Optional[Callable], OpCategory]:
        """
        Map Triton operation name to MLX operation and category

        Args:
            triton_op: Triton operation name (e.g., "tt.binary.add")

        Returns:
            Tuple of (operation callable, operation category)
        """
        # Parse the Triton operation name
        parts = triton_op.split(".")

        # Handle different operation formats
        if len(parts) >= 2 and parts[0] == "tt":
            if parts[1] == "binary":
                if len(parts) >= 3:
                    op_name = parts[2]
                    return self.get_op(op_name, OpCategory.ELEMENTWISE), OpCategory.ELEMENTWISE
            elif parts[1] == "unary":
                if len(parts) >= 3:
                    op_name = parts[2]
                    return self.get_op(op_name, OpCategory.ELEMENTWISE), OpCategory.ELEMENTWISE
            elif parts[1] == "reduce":
                if len(parts) >= 3:
                    op_name = parts[2] + "_reduce"
                    return self.get_op(op_name, OpCategory.REDUCTION), OpCategory.REDUCTION
            elif parts[1] == "cmp":
                if len(parts) >= 3:
                    op_name = parts[2]
                    return self.get_op(op_name, OpCategory.ELEMENTWISE), OpCategory.ELEMENTWISE
            elif parts[1] == "dot":
                return self.get_op("matmul", OpCategory.MATRIX), OpCategory.MATRIX
            elif parts[1] == "conv":
                return self.get_op("conv", OpCategory.TENSOR), OpCategory.TENSOR
            elif parts[1] == "atomic":
                # Atomic operations need special handling
                return None, OpCategory.SYNCHRONIZATION
            elif parts[1] in ["load", "store"]:
                return None, OpCategory.MEMORY

        # Default case
        return None, OpCategory.ELEMENTWISE

    def fuse_operations(self, ops: List[Dict], op_sequence: List[str]) -> Optional[Callable]:
        """
        Attempt to fuse a sequence of operations into an optimized implementation

        Args:
            ops: List of operations
            op_sequence: Sequence of operation names to match

        Returns:
            Fused operation callable or None if fusion not possible
        """
        # Check if we have a matching operation sequence
        if len(ops) < len(op_sequence):
            return None

        # Extract operation types
        op_types = [op.get("type", "").split(".")[-1] for op in ops]

        # Check for specific patterns

        # Pattern 1: Fused multiply-add (a*b + c)
        if op_types[:3] == ["mul", "add", "add"]:
            # This corresponds to a*b + c
            return self.get_op("fma", OpCategory.ELEMENTWISE)

        # Pattern 2: Attention pattern
        if op_types[:4] == ["matmul", "scale", "softmax", "matmul"]:
            return self.get_op("attention", OpCategory.MATRIX)

        # No fusion opportunity found
        return None

class OpConversionRegistry:
    """
    Registry for operation conversion functions with MLX implementations
    """

    def __init__(self, mlx_dispatcher: MLXDispatcher = None):
        """
        Initialize operation conversion registry

        Args:
            mlx_dispatcher: MLX operation dispatcher
        """
        self.dispatcher = mlx_dispatcher or MLXDispatcher()
        self.converters = {}
        self.register_default_converters()

    def register_default_converters(self):
        """Register default converters for common operations"""

        # Register binary operations
        for op in ["add", "sub", "mul", "div", "max", "min"]:
            self.register_converter(f"tt.binary.{op}", self._convert_binary_op)

        # Register unary operations
        for op in ["exp", "log", "sqrt", "sin", "cos", "sigmoid", "abs", "neg"]:
            self.register_converter(f"tt.unary.{op}", self._convert_unary_op)

        # Register comparison operations
        for op in ["eq", "ne", "lt", "le", "gt", "ge"]:
            self.register_converter(f"tt.cmp.{op}", self._convert_comparison_op)

        # Register reduction operations
        for op in ["sum", "max", "min", "mean"]:
            self.register_converter(f"tt.reduce.{op}", self._convert_reduction_op)

        # Register matrix operations
        self.register_converter("tt.dot", self._convert_matmul)

        # Register tensor operations
        self.register_converter("tt.reshape", self._convert_reshape)
        self.register_converter("tt.transpose", self._convert_transpose)

    def register_converter(self, op_type: str, converter_func: Callable):
        """
        Register a converter function for an operation type

        Args:
            op_type: Operation type string (e.g., "tt.binary.add")
            converter_func: Converter function that takes (op, context) and returns result
        """
        self.converters[op_type] = converter_func

    def get_converter(self, op_type: str) -> Optional[Callable]:
        """
        Get converter function for an operation type

        Args:
            op_type: Operation type string

        Returns:
            Converter function or None if not found
        """
        return self.converters.get(op_type)

    def _convert_binary_op(self, op: Dict, context: Dict) -> Any:
        """
        Convert binary operation

        Args:
            op: Operation dictionary
            context: Conversion context with results of previous operations

        Returns:
            Conversion result
        """
        # Extract operation name from the full type
        op_parts = op["type"].split(".")
        if len(op_parts) >= 3:
            op_name = op_parts[2]
        else:
            raise ValueError(f"Invalid binary operation type: {op['type']}")

        # Get MLX operation
        mlx_op, _ = self.dispatcher.map_triton_op(op["type"])
        if mlx_op is None:
            raise ValueError(f"Unsupported binary operation: {op_name}")

        # Get operands
        lhs_id = op.get("lhs_id")
        rhs_id = op.get("rhs_id")

        lhs = context.get(lhs_id)
        rhs = context.get(rhs_id)

        if lhs is None or rhs is None:
            raise ValueError(f"Missing operands for binary operation {op_name}: lhs={lhs_id}, rhs={rhs_id}")

        # Apply operation
        return mlx_op(lhs, rhs)

    def _convert_unary_op(self, op: Dict, context: Dict) -> Any:
        """
        Convert unary operation

        Args:
            op: Operation dictionary
            context: Conversion context with results of previous operations

        Returns:
            Conversion result
        """
        # Extract operation name
        op_parts = op["type"].split(".")
        if len(op_parts) >= 3:
            op_name = op_parts[2]
        else:
            raise ValueError(f"Invalid unary operation type: {op['type']}")

        # Get MLX operation
        mlx_op, _ = self.dispatcher.map_triton_op(op["type"])
        if mlx_op is None:
            raise ValueError(f"Unsupported unary operation: {op_name}")

        # Get operand
        operand_id = op.get("operand_id")
        operand = context.get(operand_id)

        if operand is None:
            raise ValueError(f"Missing operand for unary operation {op_name}: operand={operand_id}")

        # Apply operation
        return mlx_op(operand)

    def _convert_comparison_op(self, op: Dict, context: Dict) -> Any:
        """
        Convert comparison operation

        Args:
            op: Operation dictionary
            context: Conversion context with results of previous operations

        Returns:
            Conversion result
        """
        # Extract operation name
        op_parts = op["type"].split(".")
        if len(op_parts) >= 3:
            op_name = op_parts[2]
        else:
            raise ValueError(f"Invalid comparison operation type: {op['type']}")

        # Get MLX operation
        mlx_op, _ = self.dispatcher.map_triton_op(op["type"])
        if mlx_op is None:
            raise ValueError(f"Unsupported comparison operation: {op_name}")

        # Get operands
        lhs_id = op.get("lhs_id")
        rhs_id = op.get("rhs_id")

        lhs = context.get(lhs_id)
        rhs = context.get(rhs_id)

        if lhs is None or rhs is None:
            raise ValueError(f"Missing operands for comparison operation {op_name}: lhs={lhs_id}, rhs={rhs_id}")

        # Apply operation
        return mlx_op(lhs, rhs)

    def _convert_reduction_op(self, op: Dict, context: Dict) -> Any:
        """
        Convert reduction operation

        Args:
            op: Operation dictionary
            context: Conversion context with results of previous operations

        Returns:
            Conversion result
        """
        # Extract operation name
        op_parts = op["type"].split(".")
        if len(op_parts) >= 3:
            op_name = op_parts[2] + "_reduce"
        else:
            raise ValueError(f"Invalid reduction operation type: {op['type']}")

        # Get MLX operation
        mlx_op, _ = self.dispatcher.map_triton_op(op["type"])
        if mlx_op is None:
            raise ValueError(f"Unsupported reduction operation: {op_name}")

        # Get operand
        operand_id = op.get("operand_id")
        operand = context.get(operand_id)

        if operand is None:
            raise ValueError(f"Missing operand for reduction operation {op_name}: operand={operand_id}")

        # Get dimensions
        dims = op.get("dims", [0])

        # Apply operation
        return mlx_op(operand, dims)

    def _convert_matmul(self, op: Dict, context: Dict) -> Any:
        """
        Convert matrix multiplication

        Args:
            op: Operation dictionary
            context: Conversion context with results of previous operations

        Returns:
            Conversion result
        """
        # Get MLX operation
        mlx_op, _ = self.dispatcher.map_triton_op(op["type"])
        if mlx_op is None:
            raise ValueError(f"Unsupported matrix operation: {op['type']}")

        # Get operands
        a_id = op.get("a_id")
        b_id = op.get("b_id")

        a = context.get(a_id)
        b = context.get(b_id)

        if a is None or b is None:
            raise ValueError(f"Missing operands for matrix operation: a={a_id}, b={b_id}")

        # Get transpose flags
        trans_a = op.get("trans_a", False)
        trans_b = op.get("trans_b", False)

        # Apply operation with optimized GEMM
        return self.dispatcher._optimized_gemm(a, b, trans_a=trans_a, trans_b=trans_b)

    def _convert_reshape(self, op: Dict, context: Dict) -> Any:
        """
        Convert reshape operation

        Args:
            op: Operation dictionary
            context: Conversion context with results of previous operations

        Returns:
            Conversion result
        """
        # Get MLX operation
        mlx_op, _ = self.dispatcher.map_triton_op(op["type"])
        if mlx_op is None:
            raise ValueError(f"Unsupported reshape operation: {op['type']}")

        # Get operand
        input_id = op.get("input_id")
        input_tensor = context.get(input_id)

        if input_tensor is None:
            raise ValueError(f"Missing input for reshape operation: input={input_id}")

        # Get new shape
        new_shape = op.get("new_shape")

        # Apply operation
        return mlx_op(input_tensor, new_shape)

    def _convert_transpose(self, op: Dict, context: Dict) -> Any:
        """
        Convert transpose operation

        Args:
            op: Operation dictionary
            context: Conversion context with results of previous operations

        Returns:
            Conversion result
        """
        # Get MLX operation
        mlx_op, _ = self.dispatcher.map_triton_op(op["type"])
        if mlx_op is None:
            raise ValueError(f"Unsupported transpose operation: {op['type']}")

        # Get operand
        input_id = op.get("input_id")
        input_tensor = context.get(input_id)

        if input_tensor is None:
            raise ValueError(f"Missing input for transpose operation: input={input_id}")

        # Get permutation
        perm = op.get("perm")

        # Apply operation
        if perm:
            return mlx_op(input_tensor, perm)
        else:
            return mlx_op(input_tensor)

# Create global instances for convenience
mlx_dispatcher = MLXDispatcher()
op_conversion_registry = OpConversionRegistry(mlx_dispatcher)