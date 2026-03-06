"""
Metal Fusion Optimizer for Triton on Apple Silicon

This module provides pattern recognition and operation fusion optimizations
specifically designed for Apple Metal GPUs.
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Any, Tuple
from enum import Enum


try:
    # Import safely to handle cases where hardware detection fails
    from MLX.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
except ImportError:
    # Define dummy hardware capabilities
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

from MLX.operation_mapping import MLXDispatcher, OpCategory

class FusionPattern:
    """
    Represents a pattern of operations that can be fused together
    """

    def __init__(self, name: str, op_pattern: List[str], min_hardware_gen=None, pattern_matcher=None):
        """
        Initialize fusion pattern

        Args:
            name: Pattern name
            op_pattern: List of operation types that make up the pattern
            min_hardware_gen: Minimum hardware generation required
            pattern_matcher: Optional function to match more complex patterns
        """
        self.name = name
        self.op_pattern = op_pattern
        self.min_hardware_gen = min_hardware_gen if min_hardware_gen is not None else AppleSiliconGeneration.UNKNOWN
        self.pattern_matcher = pattern_matcher

    def matches(self, ops: List[Dict], start_idx: int = 0) -> bool:
        """
        Check if operations match this pattern

        Args:
            ops: List of operations
            start_idx: Starting index to check from

        Returns:
            True if pattern matches, False otherwise
        """
        if start_idx + len(self.op_pattern) > len(ops):
            return False

        for i, pattern_op in enumerate(self.op_pattern):
            op_idx = start_idx + i
            op_type = ops[op_idx].get("type", "").split(".")[-1]

            if op_type != pattern_op:
                return False

        return True

class FusionOptimizer:
    """
    Optimizer that recognizes and fuses patterns of operations for Metal
    """

    def __init__(self, hardware_capabilities=None, dispatcher=None):
        """
        Initialize fusion optimizer

        Args:
            hardware_capabilities: Optional hardware capabilities
            dispatcher: Optional MLX operation dispatcher
        """
        self.hardware = hardware_capabilities
        self.dispatcher = dispatcher or MLXDispatcher(self.hardware)
        self.patterns = self._create_fusion_patterns()

    def _create_fusion_patterns(self) -> List[FusionPattern]:
        """
        Create list of fusion patterns to recognize

        Returns:
            List of fusion patterns
        """
        patterns = []

        # Pattern 1: Fused multiply-add (a*b + c)
        patterns.append(FusionPattern(
            "fused_multiply_add",
            ["mul", "add"],
            None
        ))

        # Pattern 2: GELU activation (approximate)
        patterns.append(FusionPattern(
            "gelu",
            ["mul", "pow", "mul", "add", "mul", "tanh", "add", "mul"],
            None
        ))

        # Pattern 3: Layer normalization
        patterns.append(FusionPattern(
            "layer_norm",
            ["sub", "pow", "mean", "add", "sqrt", "div", "mul", "add"],
            None
        ))

        # Pattern 4: Attention mechanism
        patterns.append(FusionPattern(
            "attention",
            ["matmul", "div", "add", "softmax", "matmul"],
            None
        ))

        # Pattern 5: Dropout + residual
        patterns.append(FusionPattern(
            "dropout_residual",
            ["mul", "add"],
            None
        ))

        # M3-specific patterns
        if self.hardware and hasattr(self.hardware, "chip_generation"):
            # Pattern 6: SwiGLU activation (M3 specific)
            # SwiGLU: x * sigmoid(gate)
            if self.hardware.chip_generation == AppleSiliconGeneration.M3:
                patterns.append(FusionPattern(
                    "swiglu",
                    ["linear", "sigmoid", "mul"],
                    AppleSiliconGeneration.M3
                ))

                # Alternative SwiGLU pattern (when linear outputs are already computed)
                patterns.append(FusionPattern(
                    "swiglu",
                    ["sigmoid", "mul"],
                    AppleSiliconGeneration.M3
                ))

        return patterns

    def find_fusion_opportunities(self, ops: List[Dict]) -> List[Tuple[int, FusionPattern, int]]:
        """
        Find opportunities for fusion in the operation list

        Args:
            ops: List of operations

        Returns:
            List of (start_index, pattern, pattern_length) tuples
        """
        opportunities = []

        # Iterate through operations looking for patterns
        for i in range(len(ops)):
            for pattern in self.patterns:
                # Check if this pattern is hardware-specific
                if pattern.min_hardware_gen is not None:
                    # Skip if we don't have hardware capabilities
                    if not self.hardware or not hasattr(self.hardware, "chip_generation"):
                        continue
                    # Skip if hardware generation doesn't meet minimum requirement
                    if self.hardware.chip_generation.value < pattern.min_hardware_gen.value:
                        continue

                # Check if pattern matches starting at this position
                if pattern.matches(ops, i):
                    opportunities.append((i, pattern, len(pattern.op_pattern)))

        # Sort opportunities by start index
        opportunities.sort(key=lambda x: x[0])

        return opportunities

    def apply_fusion(self, ops: List[Dict], start_idx: int, pattern: FusionPattern,
                    end_idx: int) -> Tuple[List[Dict], bool]:
        """
        Apply fusion optimization to a pattern of operations

        Args:
            ops: List of operations
            start_idx: Starting index of pattern
            pattern: Fusion pattern
            end_idx: Ending index of pattern

        Returns:
            Tuple of (new operations list, success flag)
        """
        # Make a copy of the operations
        new_ops = ops.copy()

        # Create a new operation representing the fused pattern
        fused_op = {
            "id": f"fused_{pattern.name}_{start_idx}",
            "type": f"tt.fused.{pattern.name}",
            "original_ops": ops[start_idx:end_idx],
            "inputs": self._gather_inputs(ops, start_idx, end_idx),
            "outputs": self._gather_outputs(ops, start_idx, end_idx, ops),
        }

        # Add specialized attributes based on pattern
        if pattern.name == "fused_multiply_add":
            self._setup_fma_op(fused_op, ops, start_idx)
        elif pattern.name == "gelu":
            self._setup_gelu_op(fused_op, ops, start_idx)
        elif pattern.name == "layer_norm":
            self._setup_layer_norm_op(fused_op, ops, start_idx)
        elif pattern.name == "attention" or pattern.name == "flash_attention":
            self._setup_attention_op(fused_op, ops, start_idx)
        elif pattern.name == "dropout_residual":
            self._setup_dropout_residual_op(fused_op, ops, start_idx)
        elif pattern.name == "swiglu":
            self._setup_swiglu_op(fused_op, ops, start_idx)

        # Replace the original operations with the fused operation
        new_ops = new_ops[:start_idx] + [fused_op] + new_ops[end_idx:]

        return new_ops, True

    def _gather_inputs(self, ops: List[Dict], start_idx: int, end_idx: int) -> Dict[str, str]:
        """
        Gather all external inputs to the pattern

        Args:
            ops: List of operations
            start_idx: Start index of pattern
            end_idx: End index of pattern

        Returns:
            Dictionary mapping input parameters to operation IDs
        """
        pattern_op_ids = set()
        for op in ops[start_idx:end_idx]:
            if "id" in op:
                pattern_op_ids.add(op["id"])

        inputs = {}

        # Look through all operations in the pattern
        for i in range(start_idx, end_idx):
            op = ops[i]

            # Check different types of inputs based on operation type
            for input_key in ["lhs_id", "rhs_id", "operand_id", "a_id", "b_id", "c_id",
                             "input_id", "filter_id", "tensor_id"]:
                if input_key in op and op[input_key] not in pattern_op_ids:
                    # This is an external input
                    param_name = f"{input_key}_{i - start_idx}"
                    inputs[param_name] = op[input_key]

        return inputs

    def _gather_outputs(self, ops: List[Dict], start_idx: int, end_idx: int,
                       all_ops: List[Dict]) -> List[str]:
        """
        Gather all outputs from the pattern that are used outside the pattern

        Args:
            ops: Operations in the pattern
            start_idx: Start index of pattern
            end_idx: End index of pattern
            all_ops: All operations

        Returns:
            List of operation IDs that produce outputs used outside the pattern
        """
        pattern_op_ids = set()
        for op in ops[start_idx:end_idx]:
            if "id" in op:
                pattern_op_ids.add(op["id"])

        outputs = []

        # Check all operations outside the pattern for references to pattern outputs
        for i, op in enumerate(all_ops):
            if i < start_idx or i >= end_idx:
                # Check if this operation uses any outputs from the pattern
                for input_key in ["lhs_id", "rhs_id", "operand_id", "a_id", "b_id", "c_id",
                                 "input_id", "filter_id", "tensor_id"]:
                    if input_key in op and op[input_key] in pattern_op_ids:
                        # This pattern output is used outside
                        outputs.append(op[input_key])

        return list(set(outputs))  # Remove duplicates

    def _setup_fma_op(self, fused_op: Dict, ops: List[Dict], start_idx: int):
        """
        Set up a fused multiply-add operation

        Args:
            fused_op: Fused operation being created
            ops: Original operations
            start_idx: Start index of pattern
        """
        # Extract the A, B, and C operands from the mul and add operations
        mul_op = ops[start_idx]
        add_op = ops[start_idx + 1]

        fused_op["a_id"] = mul_op.get("lhs_id")
        fused_op["b_id"] = mul_op.get("rhs_id")
        fused_op["c_id"] = add_op.get("rhs_id") if add_op.get("lhs_id") == mul_op.get("id") else add_op.get("lhs_id")
        fused_op["alpha"] = 1.0
        fused_op["beta"] = 1.0

    def _setup_gelu_op(self, fused_op: Dict, ops: List[Dict], start_idx: int):
        """
        Set up a fused GELU operation

        Args:
            fused_op: Fused operation being created
            ops: Original operations
            start_idx: Start index of pattern
        """
        # Extract the input operand from the first operation (typically the input tensor)
        first_op = ops[start_idx]

        fused_op["input_id"] = first_op.get("lhs_id") or first_op.get("rhs_id") or first_op.get("operand_id")
        fused_op["approximate"] = "tanh"  # Using tanh approximation

    def _setup_layer_norm_op(self, fused_op: Dict, ops: List[Dict], start_idx: int):
        """
        Set up a fused layer normalization operation

        Args:
            fused_op: Fused operation being created
            ops: Original operations
            start_idx: Start index of pattern
        """
        try:
            # Extract key parameters from the operations
            sub_op = ops[start_idx]
            mean_op = ops[start_idx + 2]
            mul_op = ops[start_idx + 6]
            add_op = ops[start_idx + 7]
            div_op = ops[start_idx + 5]  # Division operation

            fused_op["input_id"] = sub_op.get("lhs_id")
            fused_op["gamma_id"] = mul_op.get("lhs_id") if mul_op.get("rhs_id") == div_op.get("id") else mul_op.get("rhs_id")
            fused_op["beta_id"] = add_op.get("rhs_id") if add_op.get("lhs_id") == mul_op.get("id") else add_op.get("lhs_id")
            fused_op["axes"] = mean_op.get("dims", [-1])  # Normalize along last dimension by default
            fused_op["epsilon"] = 1e-5  # Default epsilon
        except (IndexError, KeyError):
            # Set default values if anything goes wrong
            fused_op["input_id"] = ""
            fused_op["gamma_id"] = ""
            fused_op["beta_id"] = ""
            fused_op["axes"] = [-1]
            fused_op["epsilon"] = 1e-5

    def _setup_attention_op(self, fused_op: Dict, ops: List[Dict], start_idx: int):
        """
        Set up a fused attention operation

        Args:
            fused_op: Fused operation being created
            ops: Original operations
            start_idx: Start index of pattern
        """
        # Extract query, key, value matrices
        try:
            matmul1_op = ops[start_idx]
            div_op = ops[start_idx + 1]
            softmax_op = ops[start_idx + 3]
            matmul2_op = ops[start_idx + 4]

            fused_op["query_id"] = matmul1_op.get("a_id")
            fused_op["key_id"] = matmul1_op.get("b_id")
            fused_op["value_id"] = matmul2_op.get("b_id")

            # Extract scale factor from div op if available
            if "rhs_id" in div_op:
                fused_op["scale_id"] = div_op.get("rhs_id")
            else:
                # Assume default scale of 1/sqrt(d_k)
                fused_op["scale"] = None

            # Add mask if available (typically from add operation before softmax)
            if len(ops) > start_idx + 2 and "add" in ops[start_idx + 2].get("type", ""):
                add_op = ops[start_idx + 2]
                fused_op["mask_id"] = add_op.get("rhs_id") if add_op.get("lhs_id") == div_op.get("id") else add_op.get("lhs_id")
        except (IndexError, KeyError):
            # Set default values if anything goes wrong
            fused_op["query_id"] = ""
            fused_op["key_id"] = ""
            fused_op["value_id"] = ""
            fused_op["scale"] = None

    def _setup_dropout_residual_op(self, fused_op: Dict, ops: List[Dict], start_idx: int):
        """
        Set up a fused dropout + residual operation

        Args:
            fused_op: Fused operation being created
            ops: Original operations
            start_idx: Start index of pattern
        """
        mul_op = ops[start_idx]
        add_op = ops[start_idx + 1]

        fused_op["input_id"] = mul_op.get("lhs_id") or mul_op.get("rhs_id")
        fused_op["residual_id"] = add_op.get("rhs_id") if add_op.get("lhs_id") == mul_op.get("id") else add_op.get("lhs_id")

        # Extract dropout probability from multiply constant if possible
        if isinstance(mul_op.get("lhs_id"), (int, float)):
            dropout_factor = mul_op.get("lhs_id")
            fused_op["dropout_p"] = 1.0 - dropout_factor
        elif isinstance(mul_op.get("rhs_id"), (int, float)):
            dropout_factor = mul_op.get("rhs_id")
            fused_op["dropout_p"] = 1.0 - dropout_factor
        else:
            # Default dropout probability
            fused_op["dropout_p"] = 0.1

    def _setup_swiglu_op(self, fused_op: Dict, ops: List[Dict], start_idx: int):
        """
        Set up a fused SwiGLU operation (M3 specific)

        Args:
            fused_op: Fused operation being created
            ops: Original operations
            start_idx: Start index of pattern
        """
        try:
            # Extract the input operands
            mul1_op = ops[start_idx]
            sigmoid_op = ops[start_idx + 1]

            fused_op["x_id"] = mul1_op.get("lhs_id")
            fused_op["gate_id"] = sigmoid_op.get("operand_id")
        except (IndexError, KeyError):
            # Set default values if anything goes wrong
            fused_op["x_id"] = ""
            fused_op["gate_id"] = ""

    def optimize(self, ops: List[Dict]) -> List[Dict]:
        """
        Apply fusion optimizations to operations

        Args:
            ops: List of operations

        Returns:
            Optimized list of operations
        """
        # Find all fusion opportunities
        opportunities = self.find_fusion_opportunities(ops)

        # Apply fusions in reverse order (to avoid index issues)
        current_ops = ops.copy()
        opportunities.sort(key=lambda x: x[0], reverse=True)

        for start_idx, pattern, pattern_len in opportunities:
            end_idx = start_idx + pattern_len
            new_ops, success = self.apply_fusion(current_ops, start_idx, pattern, end_idx)

            if success:
                current_ops = new_ops

        return current_ops

    def execute_fused_op(self, op: Dict, context: Dict) -> Any:
        """
        Execute a fused operation

        Args:
            op: Fused operation
            context: Execution context with tensors

        Returns:
            Result of the fused operation
        """
        op_type = op.get("type", "").split(".")[-1]

        if op_type == "fused_multiply_add":
            return self._execute_fma(op, context)
        elif op_type == "gelu":
            return self._execute_gelu(op, context)
        elif op_type == "layer_norm":
            return self._execute_layer_norm(op, context)
        elif op_type == "attention" or op_type == "flash_attention":
            return self._execute_attention(op, context)
        elif op_type == "dropout_residual":
            return self._execute_dropout_residual(op, context)
        elif op_type == "swiglu":
            return self._execute_swiglu(op, context)
        else:
            raise ValueError(f"Unknown fused operation type: {op_type}")

    def _execute_fma(self, op: Dict, context: Dict) -> Any:
        """
        Execute a fused multiply-add operation

        Args:
            op: Fused operation
            context: Execution context with tensors

        Returns:
            Result tensor
        """
        a = context.get(op.get("a_id"))
        b = context.get(op.get("b_id"))
        c = context.get(op.get("c_id"))

        if a is None or b is None:
            raise ValueError(f"Missing operands for FMA operation: a={op.get('a_id')}, b={op.get('b_id')}")

        # Get alpha and beta factors
        alpha = op.get("alpha", 1.0)
        beta = op.get("beta", 1.0)

        # Use MLX's optimized FMA/GEMM operation if available
        if hasattr(mx, "addmm") and len(a.shape) >= 2 and len(b.shape) >= 2:
            # For matrix inputs, use addmm (C = beta*C + alpha*(A@B))
            return mx.addmm(c if c is not None else 0, a, b, beta=beta, alpha=alpha)
        else:
            # For non-matrix inputs or if addmm not available, compute directly
            result = a * b
            if alpha != 1.0:
                result = result * alpha

            if c is not None:
                if beta != 1.0:
                    result = result + beta * c
                else:
                    result = result + c

            return result

    def _execute_gelu(self, op: Dict, context: Dict) -> Any:
        """
        Execute a fused GELU operation

        Args:
            op: Fused operation
            context: Execution context with tensors

        Returns:
            Result tensor
        """
        x = context.get(op.get("input_id"))
        if x is None:
            raise ValueError(f"Missing input for GELU operation: input={op.get('input_id')}")

        # Check if MLX has a native GELU implementation
        if hasattr(mx, "gelu"):
            return mx.gelu(x)

        # Otherwise compute the approximation manually
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + mx.tanh(sqrt_2_over_pi * (x + 0.044715 * mx.power(x, 3))))

    def _execute_layer_norm(self, op: Dict, context: Dict) -> Any:
        """
        Execute a fused layer normalization operation

        Args:
            op: Fused operation
            context: Execution context with tensors

        Returns:
            Result tensor
        """
        x = context.get(op.get("input_id"))
        gamma = context.get(op.get("gamma_id"))
        beta = context.get(op.get("beta_id"))

        if x is None:
            raise ValueError(f"Missing input for layer norm operation: input={op.get('input_id')}")

        # Get normalization parameters
        axes = op.get("axes", [-1])
        epsilon = op.get("epsilon", 1e-5)

        # Check if MLX has a native layer_norm implementation
        if hasattr(mx, "layer_norm"):
            return mx.layer_norm(x, axes, gamma, beta, epsilon)

        # Otherwise compute manually
        # Compute mean and variance
        mean = mx.mean(x, axis=axes, keepdims=True)
        var = mx.mean(mx.square(x - mean), axis=axes, keepdims=True)

        # Normalize
        x_norm = (x - mean) / mx.sqrt(var + epsilon)

        # Scale and shift
        if gamma is not None:
            x_norm = x_norm * gamma

        if beta is not None:
            x_norm = x_norm + beta

        return x_norm

    def _execute_attention(self, op: Dict, context: Dict) -> Any:
        """
        Execute a fused attention operation

        Args:
            op: Fused operation
            context: Execution context with tensors

        Returns:
            Result tensor
        """
        q = context.get(op.get("query_id"))
        k = context.get(op.get("key_id"))
        v = context.get(op.get("value_id"))

        if q is None or k is None or v is None:
            raise ValueError(f"Missing inputs for attention operation")

        # Get optional parameters
        scale_tensor = context.get(op.get("scale_id"))
        scale = op.get("scale") if scale_tensor is None else scale_tensor
        mask = context.get(op.get("mask_id"))

        # Default scale if not provided
        if scale is None:
            d_k = q.shape[-1]
            scale = 1.0 / (d_k ** 0.5)

        # Use the dispatcher's optimized attention implementation
        return self.dispatcher._optimized_attention(q, k, v, scale, mask)

    def _execute_dropout_residual(self, op: Dict, context: Dict) -> Any:
        """
        Execute a fused dropout + residual operation

        Args:
            op: Fused operation
            context: Execution context with tensors

        Returns:
            Result tensor
        """
        x = context.get(op.get("input_id"))
        residual = context.get(op.get("residual_id"))

        if x is None:
            raise ValueError(f"Missing input for dropout+residual operation: input={op.get('input_id')}")

        # Get dropout probability
        dropout_p = op.get("dropout_p", 0.1)

        # Apply dropout - but only during training
        # For inference, we just scale by (1-p)
        if dropout_p > 0.0:
            if hasattr(mx, "dropout"):
                dropped = mx.dropout(x, dropout_p)
            else:
                # Manual dropout implementation
                dropped = x * (1.0 - dropout_p)
        else:
            dropped = x

        # Add residual connection if provided
        if residual is not None:
            return dropped + residual
        else:
            return dropped

    def _execute_swiglu(self, op: Dict, context: Dict) -> Any:
        """
        Execute a fused SwiGLU operation (M3 specific)

        Args:
            op: Fused operation
            context: Execution context with tensors

        Returns:
            Result tensor
        """
        x = context.get(op.get("x_id"))
        gate = context.get(op.get("gate_id"))

        if x is None or gate is None:
            raise ValueError(f"Missing inputs for SwiGLU operation")

        # SwiGLU: x * sigmoid(gate)
        return x * mx.sigmoid(gate)

# Create global instance with safe initialization
try:
    fusion_optimizer = FusionOptimizer(hardware_capabilities)
except Exception as e:
    print(f"Warning: Could not initialize fusion optimizer with hardware capabilities: {e}")
    fusion_optimizer = FusionOptimizer()