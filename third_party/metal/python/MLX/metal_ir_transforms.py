"""
Metal IR Transformations for Triton

This module provides IR transformations that optimize Triton operations for
Metal Performance Shaders (MPS) on Apple Silicon GPUs.
"""


from typing import Dict, List, Any, Tuple
from enum import Enum

# Import required modules
from MLX.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
from MLX.metal_performance_shaders import Metal32Feature, mps_integration, mlx_metal_kernel_mapping

class TransformPass(Enum):
    """Types of transformation passes"""
    PATTERN_MATCHING = 0
    OPERATION_FUSION = 1
    MEMORY_LAYOUT = 2
    MPS_ACCELERATION = 3
    M3_OPTIMIZATION = 4
    VECTORIZATION = 5
    BARRIER_ELIMINATION = 6
    THREAD_COARSENING = 7

class TransformationContext:
    """Context for IR transformations"""

    def __init__(self, ir_ops: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        """
        Initialize transformation context

        Args:
            ir_ops: List of IR operations
            metadata: Optional metadata for the kernel
        """
        self.original_ops = ir_ops.copy()
        self.current_ops = ir_ops.copy()
        self.metadata = metadata or {}
        self.hardware = hardware_capabilities
        self.transformed_ops = {}  # Maps original op IDs to transformed ops
        self.pattern_matches = []  # List of matched patterns
        self.applied_transformations = []  # List of applied transformations
        self.mps_accelerated_ops = set()  # Set of operation IDs accelerated by MPS

    def add_pattern_match(self, pattern_name: str, start_idx: int, end_idx: int, ops: List[Dict[str, Any]]):
        """
        Add a matched pattern

        Args:
            pattern_name: Name of the matched pattern
            start_idx: Start index in the operations list
            end_idx: End index in the operations list
            ops: List of operations involved in the pattern
        """
        self.pattern_matches.append({
            "pattern_name": pattern_name,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "ops": ops.copy()
        })

    def add_transformation(self, transform_type: TransformPass, op_ids: List[str], description: str):
        """
        Record an applied transformation

        Args:
            transform_type: Type of transformation
            op_ids: List of affected operation IDs
            description: Description of the transformation
        """
        self.applied_transformations.append({
            "type": transform_type,
            "op_ids": op_ids.copy(),
            "description": description,
            "timestamp": len(self.applied_transformations)
        })

    def replace_ops(self, start_idx: int, end_idx: int, new_ops: List[Dict[str, Any]]):
        """
        Replace a range of operations with new operations

        Args:
            start_idx: Start index to replace
            end_idx: End index to replace
            new_ops: New operations to insert

        Returns:
            List of new operation IDs
        """
        # Collect old operation IDs for tracking
        old_op_ids = [op.get("id", f"op_{i}") for i, op in enumerate(self.current_ops[start_idx:end_idx])]

        # Insert new operations
        self.current_ops = self.current_ops[:start_idx] + new_ops + self.current_ops[end_idx:]

        # Collect new operation IDs
        new_op_ids = [op.get("id", f"new_op_{i}") for i, op in enumerate(new_ops)]

        # Record the transformation mapping
        for old_id in old_op_ids:
            self.transformed_ops[old_id] = new_op_ids

        return new_op_ids

    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all transformations

        Returns:
            Dictionary with transformation summary
        """
        return {
            "num_original_ops": len(self.original_ops),
            "num_current_ops": len(self.current_ops),
            "num_pattern_matches": len(self.pattern_matches),
            "num_transformations": len(self.applied_transformations),
            "num_mps_accelerated": len(self.mps_accelerated_ops),
            "transformations_by_type": self._count_transformations_by_type(),
            "hardware_generation": self.hardware.chip_generation.name,
        }

    def _count_transformations_by_type(self) -> Dict[str, int]:
        """
        Count transformations by type

        Returns:
            Dictionary mapping transformation types to counts
        """
        counts = {}
        for transform in self.applied_transformations:
            type_name = transform["type"].name
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

class MPSPatternMatcher:
    """Matcher for MPS-accelerated operation patterns"""

    def __init__(self):
        """Initialize MPS pattern matcher"""
        self.mps_kernel_mapping = mlx_metal_kernel_mapping
        self.mps_integration = mps_integration
        self.hardware = hardware_capabilities

    def find_patterns(self, context: TransformationContext) -> List[Dict[str, Any]]:
        """
        Find MPS-acceleratable patterns in the IR

        Args:
            context: Transformation context

        Returns:
            List of matched patterns
        """
        matched_patterns = []
        ops = context.current_ops

        # Iterate through operations to find patterns
        i = 0
        while i < len(ops):
            # Try to match patterns at this position
            pattern_match = self.mps_kernel_mapping.match_kernel_pattern(ops[i:])

            if pattern_match:
                start_idx = i + pattern_match["start_idx"]
                end_idx = i + pattern_match["end_idx"]

                # Check if this pattern is available on the current hardware
                mps_op = pattern_match["mps_op"]
                if self.mps_integration.is_operation_available(mps_op):
                    matched_patterns.append({
                        "name": pattern_match["name"],
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "mps_op": mps_op,
                        "params": pattern_match["params"],
                        "additional": pattern_match.get("additional", {}),
                        "ops": ops[start_idx:end_idx]
                    })

                    # Record the pattern match in the context
                    context.add_pattern_match(
                        pattern_match["name"],
                        start_idx,
                        end_idx,
                        ops[start_idx:end_idx]
                    )

                    # Skip to end of this pattern
                    i = end_idx
                else:
                    i += 1
            else:
                i += 1

        return matched_patterns

class MPSTransformer:
    """Transformer for MPS-accelerated operations"""

    def __init__(self):
        """Initialize MPS transformer"""
        self.mps_kernel_mapping = mlx_metal_kernel_mapping
        self.mps_integration = mps_integration
        self.hardware = hardware_capabilities
        self.pattern_matcher = MPSPatternMatcher()

    def transform_to_mps_ops(self, context: TransformationContext) -> bool:
        """
        Transform operations to MPS-accelerated operations

        Args:
            context: Transformation context

        Returns:
            True if any transformations were applied, False otherwise
        """
        # Find MPS patterns
        patterns = self.pattern_matcher.find_patterns(context)

        if not patterns:
            return False

        # Apply transformations for each pattern, in reverse order to avoid index issues
        patterns.sort(key=lambda p: p["start_idx"], reverse=True)

        for pattern in patterns:
            self._transform_pattern(context, pattern)

        return True

    def _transform_pattern(self, context: TransformationContext, pattern: Dict[str, Any]):
        """
        Transform a pattern to MPS operations

        Args:
            context: Transformation context
            pattern: Pattern to transform
        """
        start_idx = pattern["start_idx"]
        end_idx = pattern["end_idx"]
        pattern_name = pattern["name"]
        mps_op = pattern["mps_op"]

        # Extract operations involved in the pattern
        old_ops = context.current_ops[start_idx:end_idx]
        old_op_ids = [op.get("id", f"op_{start_idx + i}") for i, op in enumerate(old_ops)]

        # Create a new operation for the MPS operation
        new_op_id = f"mps_{pattern_name}_{old_op_ids[0]}"

        # Extract parameters for the MPS operation
        params = self.mps_kernel_mapping.extract_params(pattern, old_ops)

        # Create the new operation
        new_op = {
            "id": new_op_id,
            "type": f"mps.{pattern_name}",
            "mps_op": mps_op.name,
            "original_ops": old_op_ids,
            "params": params,
            "additional": pattern.get("additional", {})
        }

        # Replace the old operations with the new operation
        new_op_ids = context.replace_ops(start_idx, end_idx, [new_op])

        # Mark the operations as MPS-accelerated
        context.mps_accelerated_ops.add(new_op_id)

        # Record the transformation
        context.add_transformation(
            TransformPass.MPS_ACCELERATION,
            old_op_ids,
            f"Transformed {pattern_name} pattern to MPS operation"
        )

class M3Optimizer:
    """Special optimizer for M3-specific capabilities"""

    def __init__(self):
        """Initialize M3 optimizer"""
        self.hardware = hardware_capabilities
        self.mps_integration = mps_integration

    def is_m3_or_newer(self) -> bool:
        """
        Check if the hardware is M3 or newer

        Returns:
            True if M3 or newer, False otherwise
        """
        return self.hardware.chip_generation.value >= AppleSiliconGeneration.M3.value

    def apply_m3_optimizations(self, context: TransformationContext) -> bool:
        """
        Apply M3-specific optimizations

        Args:
            context: Transformation context

        Returns:
            True if any optimizations were applied, False otherwise
        """
        if not self.is_m3_or_newer():
            return False

        optimizations_applied = False

        # Check for available Metal 3.2 features
        if self.mps_integration.is_feature_available(Metal32Feature.DYNAMIC_CACHING):
            optimizations_applied |= self._apply_dynamic_caching(context)

        if self.mps_integration.is_feature_available(Metal32Feature.MATRIX_PATCHING):
            optimizations_applied |= self._optimize_matrix_operations(context)

        if self.mps_integration.is_feature_available(Metal32Feature.SPARSE_ACCELERATION):
            optimizations_applied |= self._optimize_sparse_operations(context)

        return optimizations_applied

    def _apply_dynamic_caching(self, context: TransformationContext) -> bool:
        """
        Apply dynamic caching optimizations for M3

        Args:
            context: Transformation context

        Returns:
            True if optimizations were applied, False otherwise
        """
        # Dynamic caching is mostly handled by the Metal driver,
        # but we can add hints to operations for better caching

        ops = context.current_ops
        modified = False

        for i, op in enumerate(ops):
            if "matmul" in op.get("type", "") or "conv" in op.get("type", ""):
                # Add dynamic caching hint
                op["dynamic_cache_hint"] = True
                modified = True

                # Record the transformation
                context.add_transformation(
                    TransformPass.M3_OPTIMIZATION,
                    [op.get("id", f"op_{i}")],
                    "Added dynamic caching hint for matrix/convolution operation"
                )

        return modified

    def _optimize_matrix_operations(self, context: TransformationContext) -> bool:
        """
        Optimize matrix operations for M3

        Args:
            context: Transformation context

        Returns:
            True if optimizations were applied, False otherwise
        """
        # Look for matrix operations that can be optimized
        ops = context.current_ops
        modified = False

        for i, op in enumerate(ops):
            if "matmul" in op.get("type", "") or "gemm" in op.get("type", ""):
                # Add M3-specific matrix optimizations
                op["m3_matrix_engine"] = True

                # Get matrix dimensions if available
                a_id = op.get("a_id")
                b_id = op.get("b_id")

                # Record the transformation
                context.add_transformation(
                    TransformPass.M3_OPTIMIZATION,
                    [op.get("id", f"op_{i}")],
                    "Applied M3-specific matrix engine optimizations"
                )

                modified = True

        return modified

    def _optimize_sparse_operations(self, context: TransformationContext) -> bool:
        """
        Optimize operations involving sparse matrices for M3

        Args:
            context: Transformation context

        Returns:
            True if optimizations were applied, False otherwise
        """
        # Search for potential sparse operations
        ops = context.current_ops
        modified = False

        # This is a placeholder for more sophisticated sparse optimization
        # In a real implementation, it would analyze the sparsity patterns
        # and transform operations accordingly

        return modified

class MemoryLayoutOptimizer:
    """Optimizer for memory layouts in Metal"""

    def __init__(self):
        """Initialize memory layout optimizer"""
        self.hardware = hardware_capabilities

    def optimize_memory_layouts(self, context: TransformationContext) -> bool:
        """
        Optimize memory layouts for Metal

        Args:
            context: Transformation context

        Returns:
            True if any optimizations were applied, False otherwise
        """
        # This is a placeholder for memory layout optimizations
        # In a real implementation, it would analyze memory access patterns
        # and optimize layouts for better performance on Metal GPUs

        return False

class BarrierOptimizer:
    """Optimizer to eliminate unnecessary barriers"""

    def __init__(self):
        """Initialize barrier optimizer"""
        self.hardware = hardware_capabilities

    def eliminate_unnecessary_barriers(self, context: TransformationContext) -> bool:
        """
        Eliminate unnecessary barriers

        Args:
            context: Transformation context

        Returns:
            True if any barriers were eliminated, False otherwise
        """
        ops = context.current_ops
        barrier_indices = []

        # Find all barriers
        for i, op in enumerate(ops):
            if op.get("type", "") == "tt.debug_barrier":
                barrier_indices.append(i)

        if not barrier_indices:
            return False

        # Analyze barriers for elimination
        to_remove = []

        for i, barrier_idx in enumerate(barrier_indices):
            if i > 0 and barrier_indices[i] - barrier_indices[i-1] <= 2:
                # Consecutive barriers with no or minimal operations between them
                to_remove.append(barrier_idx)

        if not to_remove:
            return False

        # Remove barriers in reverse order to avoid index issues
        to_remove.sort(reverse=True)
        for idx in to_remove:
            op_id = ops[idx].get("id", f"op_{idx}")
            context.add_transformation(
                TransformPass.BARRIER_ELIMINATION,
                [op_id],
                "Eliminated unnecessary barrier"
            )
            ops.pop(idx)

        # Update operations in context
        context.current_ops = ops

        return True

class MetalIRTransformer:
    """Main class for IR transformations in Metal"""

    def __init__(self):
        """Initialize IR transformer"""
        self.hardware = hardware_capabilities
        self.mps_transformer = MPSTransformer()
        self.m3_optimizer = M3Optimizer()
        self.memory_optimizer = MemoryLayoutOptimizer()
        self.barrier_optimizer = BarrierOptimizer()

    def transform(self, ir_ops: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply transformations to IR operations

        Args:
            ir_ops: List of IR operations
            metadata: Optional metadata for the kernel

        Returns:
            Tuple of (transformed operations, transformation summary)
        """
        # Create transformation context
        context = TransformationContext(ir_ops, metadata)

        # Apply transformations in order
        self._apply_barrier_elimination(context)
        self._apply_mps_acceleration(context)

        # Apply M3-specific optimizations if appropriate
        if self.hardware.chip_generation.value >= AppleSiliconGeneration.M3.value:
            self._apply_m3_optimizations(context)

        # Apply memory layout optimizations
        self._apply_memory_layout_optimizations(context)

        # Return transformed operations and summary
        return context.current_ops, context.get_transformation_summary()

    def _apply_barrier_elimination(self, context: TransformationContext):
        """
        Apply barrier elimination

        Args:
            context: Transformation context
        """
        self.barrier_optimizer.eliminate_unnecessary_barriers(context)

    def _apply_mps_acceleration(self, context: TransformationContext):
        """
        Apply MPS acceleration transformations

        Args:
            context: Transformation context
        """
        self.mps_transformer.transform_to_mps_ops(context)

    def _apply_m3_optimizations(self, context: TransformationContext):
        """
        Apply M3-specific optimizations

        Args:
            context: Transformation context
        """
        self.m3_optimizer.apply_m3_optimizations(context)

    def _apply_memory_layout_optimizations(self, context: TransformationContext):
        """
        Apply memory layout optimizations

        Args:
            context: Transformation context
        """
        self.memory_optimizer.optimize_memory_layouts(context)

# Create a global instance
metal_ir_transformer = MetalIRTransformer()

def transform_ir(ir_ops: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Transform IR operations for Metal

    Args:
        ir_ops: List of IR operations
        metadata: Optional metadata for the kernel

    Returns:
        Tuple of (transformed operations, transformation summary)
    """
    return metal_ir_transformer.transform(ir_ops, metadata)

def main():
    """Main function for testing"""
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python metal_ir_transforms.py <ir_file>")
        return

    with open(sys.argv[1], "r") as f:
        ir_ops = json.load(f)

    transformed_ops, summary = transform_ir(ir_ops)

    print("Transformation summary:")
    print(json.dumps(summary, indent=2))

    output_file = sys.argv[1] + ".transformed.json"
    with open(output_file, "w") as f:
        json.dump(transformed_ops, f, indent=2)

    print(f"Transformed IR written to {output_file}")

if __name__ == "__main__":
    main()