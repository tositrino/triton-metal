"""
Memory layout utilities for Metal backend

This module handles memory layout transformations between different formats
for efficient data movement between Metal and host.
"""

from typing import Tuple
import numpy as np

# Lazy import MLX to avoid unnecessary dependencies
_mx = None

def _get_mlx():
    """Lazy load MLX"""
    global _mx
    if _mx is None:
        import mlx.core as mx
        _mx = mx
    return _mx

class MemoryLayout:
    """Representation of a memory layout for a tensor"""

    def __init__(self, shape: Tuple[int, ...], layout_type: str = "row_major"):
        """Initialize memory layout

        Args:
            shape: Shape of the tensor
            layout_type: Layout type, either "row_major" or "col_major"
        """
        self.shape = shape
        self.layout_type = layout_type
        self.strides = self._compute_strides(shape, layout_type)
        self.size = np.prod(shape) if shape else 0

    def _compute_strides(self, shape: Tuple[int, ...], layout_type: str) -> Tuple[int, ...]:
        """Compute strides for a given shape and layout

        Args:
            shape: Shape of the tensor
            layout_type: Layout type, either "row_major" or "col_major"

        Returns:
            Strides for the tensor
        """
        if not shape:
            return ()

        ndim = len(shape)
        strides = [1] * ndim

        if layout_type == "row_major":
            # Row-major (C-style) strides
            for i in range(ndim - 2, -1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]
        elif layout_type == "col_major":
            # Column-major (Fortran-style) strides
            for i in range(1, ndim):
                strides[i] = strides[i - 1] * shape[i - 1]
        else:
            raise ValueError(f"Unsupported layout type: {layout_type}")

        return tuple(strides)

    def is_contiguous(self) -> bool:
        """Check if layout is contiguous

        Returns:
            True if layout is contiguous, False otherwise
        """
        expected_strides = self._compute_strides(self.shape, self.layout_type)
        return self.strides == expected_strides

    def get_index(self, coords: Tuple[int, ...]) -> int:
        """Get linear index for coordinates

        Args:
            coords: Coordinates in the tensor

        Returns:
            Linear index
        """
        if len(coords) != len(self.shape):
            raise ValueError(f"Coordinates must have same dimension as shape: {len(coords)} vs {len(self.shape)}")

        # Compute linear index
        idx = 0
        for i, (c, s) in enumerate(zip(coords, self.strides)):
            idx += c * s

        return idx

    def get_coords(self, idx: int) -> Tuple[int, ...]:
        """Get coordinates for linear index

        Args:
            idx: Linear index

        Returns:
            Coordinates in the tensor
        """
        if idx < 0 or idx >= self.size:
            raise ValueError(f"Index out of bounds: {idx} not in [0, {self.size})")

        coords = []
        for dim, stride in zip(self.shape, self.strides):
            coord = (idx // stride) % dim
            coords.append(coord)

        return tuple(coords)

    def __eq__(self, other: "MemoryLayout") -> bool:
        """Check if two layouts are equal

        Args:
            other: Other layout to compare

        Returns:
            True if layouts are equal, False otherwise
        """
        return (
            self.shape == other.shape and
            self.layout_type == other.layout_type and
            self.strides == other.strides
        )

    def __str__(self) -> str:
        """Get string representation

        Returns:
            String representation of layout
        """
        return f"MemoryLayout(shape={self.shape}, layout_type={self.layout_type}, strides={self.strides})"

def adapt_tensor(tensor, src_layout: MemoryLayout, dst_layout: MemoryLayout):
    """Adapt tensor from source layout to destination layout

    Args:
        tensor: Input tensor
        src_layout: Source layout
        dst_layout: Destination layout

    Returns:
        Tensor with adapted layout
    """
    mx = _get_mlx()

    # Check if shapes match
    if src_layout.shape != dst_layout.shape:
        raise ValueError(f"Source and destination layouts must have same shape: {src_layout.shape} vs {dst_layout.shape}")

    # If layouts are the same, return the tensor as is
    if src_layout == dst_layout:
        return tensor

    # Handle row-major to column-major conversion (and vice versa)
    if (src_layout.layout_type == "row_major" and dst_layout.layout_type == "col_major") or \
       (src_layout.layout_type == "col_major" and dst_layout.layout_type == "row_major"):
        # For 2D tensors, this is just a transpose
        if len(src_layout.shape) == 2:
            return mx.transpose(tensor)
        # For higher-dimensional tensors, we need to permute the dimensions
        else:
            perm = list(range(len(src_layout.shape)))
            perm.reverse()
            return mx.transpose(tensor, perm)

    # For other conversions, use a general-purpose approach
    result = mx.zeros(dst_layout.shape, dtype=tensor.dtype)

    # Copy elements according to layouts
    # This is inefficient, but should work for any layout
    flat_tensor = mx.reshape(tensor, (-1,))
    flat_result = mx.reshape(result, (-1,))

    for i in range(src_layout.size):
        coords = src_layout.get_coords(i)
        j = dst_layout.get_index(coords)
        flat_result[j] = flat_tensor[i]

    return mx.reshape(flat_result, dst_layout.shape)

def get_optimal_layout(shape: Tuple[int, ...], device_type: str = "metal") -> MemoryLayout:
    """Get optimal memory layout for a given shape and device

    Args:
        shape: Shape of the tensor
        device_type: Device type, either "metal" or "cpu"

    Returns:
        Optimal memory layout for the given shape and device
    """
    if device_type == "metal":
        # For Metal, prefer row-major layout for 2D matrices
        if len(shape) == 2:
            return MemoryLayout(shape, "row_major")
        # For 4D tensors (typical for convolutions), use a specialized layout
        elif len(shape) == 4:
            # NCHW format
            return MemoryLayout(shape, "row_major")
        else:
            return MemoryLayout(shape, "row_major")
    else:
        # For CPU, use row-major layout
        return MemoryLayout(shape, "row_major")

def is_contiguous(tensor) -> bool:
    """Check if tensor has a contiguous memory layout

    Args:
        tensor: Input tensor

    Returns:
        True if tensor has a contiguous memory layout, False otherwise
    """
    mx = _get_mlx()

    # MLX might not provide direct access to strides
    # So we'll use a heuristic approach

    # Clone the tensor and reshape it to 1D
    flat = mx.reshape(tensor, (-1,))

    # Compare the original tensor and reshaping+unreshaping
    reshaped = mx.reshape(flat, tensor.shape)

    # If they match, the tensor is contiguous
    return mx.array_equal(tensor, reshaped)

def get_tensor_layout(tensor) -> MemoryLayout:
    """Get memory layout for a tensor

    Args:
        tensor: Input tensor

    Returns:
        Memory layout for the tensor
    """
    # For MLX tensors, we'll assume row-major layout
    # This might need to be adjusted based on MLX's internal representation
    return MemoryLayout(tensor.shape, "row_major")