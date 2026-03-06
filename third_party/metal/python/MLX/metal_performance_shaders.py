"""
Metal Performance Shaders Integration for Triton

This module provides integration with Apple's Metal Performance Shaders (MPS)
for high-performance operations on Apple Silicon GPUs.
"""

import os
import sys
import mlx.core as mx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum


import metal_hardware_optimizer
from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration

class MPSOperation(Enum):
    """Enum for MPS operations"""
    CONV2D = 0
    DEPTHWISE_CONV = 1
    MATMUL = 2
    BATCH_NORM = 3
    POOLING = 4
    ACTIVATION = 5
    FFT = 6
    SCAN = 7
    GEMM = 8
    # New operations for enhanced Metal 3.2 support
    SPARSE_MATMUL = 9
    CONV3D = 10
    DECONV = 11
    DILATED_CONV = 12
    ROI_POOLING = 13
    LAYERED_TEXTURE_LOAD = 14
    MATRIX_PATCH_OP = 15
    TRANSPOSED_CONV = 16
    ATTENTION = 17

class Metal32Feature(Enum):
    """Enum for Metal 3.2 features available on Apple M3 and newer"""
    DYNAMIC_CACHING = 0
    ENHANCED_MATRIX_ENGINE = 1
    MESH_SHADERS = 2
    SPARSE_ACCELERATION = 3
    LAYERED_TEXTURES = 4
    SCATTER_GATHER_OPS = 5
    MATRIX_PATCHING = 6

class MPSIntegration:
    """Integration with Metal Performance Shaders"""
    
    def __init__(self, hardware_capabilities=None):
        """
        Initialize MPS integration
        
        Args:
            hardware_capabilities: Optional hardware capabilities instance
        """
        self.hardware = hardware_capabilities or metal_hardware_optimizer.hardware_capabilities
        self.mps_ops = self._detect_available_ops()
        self.op_implementations = self._initialize_op_implementations()
        self.metal32_features = self._detect_metal32_features()
    
    def _detect_available_ops(self) -> Dict[MPSOperation, bool]:
        """
        Detect available MPS operations based on hardware
        
        Returns:
            Dictionary mapping operations to availability
        """
        # Check if we're running on Apple Silicon
        if self.hardware.chip_generation == AppleSiliconGeneration.UNKNOWN:
            return {op: False for op in MPSOperation}
        
        # Base availability - all Apple Silicon chips support these
        availability = {
            MPSOperation.CONV2D: True,
            MPSOperation.MATMUL: True,
            MPSOperation.POOLING: True,
            MPSOperation.ACTIVATION: True,
            MPSOperation.GEMM: True,
            MPSOperation.BATCH_NORM: True,
            MPSOperation.DEPTHWISE_CONV: True,
            MPSOperation.FFT: False,  # Specialized, check later
            MPSOperation.SCAN: False,  # Specialized, check later
            # New operations are not available by default
            MPSOperation.SPARSE_MATMUL: False,
            MPSOperation.CONV3D: False,
            MPSOperation.DECONV: False,
            MPSOperation.DILATED_CONV: False,
            MPSOperation.ROI_POOLING: False,
            MPSOperation.LAYERED_TEXTURE_LOAD: False,
            MPSOperation.MATRIX_PATCH_OP: False,
            MPSOperation.TRANSPOSED_CONV: False,
        }
        
        # Update availability based on chip generation
        if self.hardware.chip_generation.value >= AppleSiliconGeneration.M2.value:
            # M2 and newer have improved FFT, SCAN
            availability[MPSOperation.FFT] = True
            availability[MPSOperation.SCAN] = True
            availability[MPSOperation.CONV3D] = True
            availability[MPSOperation.DECONV] = True
            availability[MPSOperation.DILATED_CONV] = True
        
        # M3 has all operations with maximum performance
        if self.hardware.chip_generation.value >= AppleSiliconGeneration.M3.value:
            availability[MPSOperation.SPARSE_MATMUL] = True
            availability[MPSOperation.ROI_POOLING] = True
            availability[MPSOperation.LAYERED_TEXTURE_LOAD] = True
            availability[MPSOperation.MATRIX_PATCH_OP] = True
            availability[MPSOperation.TRANSPOSED_CONV] = True
        
        return availability
    
    def _detect_metal32_features(self) -> Dict[Metal32Feature, bool]:
        """
        Detect Metal 3.2 features based on hardware
        
        Returns:
            Dictionary mapping features to availability
        """
        # Default to all features unavailable
        features = {feature: False for feature in Metal32Feature}
        
        # Only M3 and newer support Metal 3.2 features
        if self.hardware.chip_generation.value >= AppleSiliconGeneration.M3.value:
            features[Metal32Feature.DYNAMIC_CACHING] = True
            features[Metal32Feature.ENHANCED_MATRIX_ENGINE] = True
            features[Metal32Feature.MESH_SHADERS] = True
            features[Metal32Feature.SPARSE_ACCELERATION] = True
            features[Metal32Feature.LAYERED_TEXTURES] = True
            features[Metal32Feature.SCATTER_GATHER_OPS] = True
            features[Metal32Feature.MATRIX_PATCHING] = True
        
        return features
    
    def _initialize_op_implementations(self) -> Dict[MPSOperation, Callable]:
        """
        Initialize operation implementations
        
        Returns:
            Dictionary mapping operations to their implementations
        """
        implementations = {
            MPSOperation.CONV2D: self._impl_conv2d,
            MPSOperation.DEPTHWISE_CONV: self._impl_depthwise_conv,
            MPSOperation.MATMUL: self._impl_matmul,
            MPSOperation.BATCH_NORM: self._impl_batch_norm,
            MPSOperation.POOLING: self._impl_pooling,
            MPSOperation.ACTIVATION: self._impl_activation,
            MPSOperation.FFT: self._impl_fft,
            MPSOperation.SCAN: self._impl_scan,
            MPSOperation.GEMM: self._impl_gemm,
            # New implementations
            MPSOperation.SPARSE_MATMUL: self._impl_sparse_matmul,
            MPSOperation.CONV3D: self._impl_conv3d,
            MPSOperation.DECONV: self._impl_deconv,
            MPSOperation.DILATED_CONV: self._impl_dilated_conv,
            MPSOperation.ROI_POOLING: self._impl_roi_pooling,
            MPSOperation.LAYERED_TEXTURE_LOAD: self._impl_layered_texture_load,
            MPSOperation.MATRIX_PATCH_OP: self._impl_matrix_patch_op,
            MPSOperation.TRANSPOSED_CONV: self._impl_transposed_conv,
        }
        return implementations
    
    def is_operation_available(self, op: MPSOperation) -> bool:
        """
        Check if an MPS operation is available
        
        Args:
            op: MPS operation
            
        Returns:
            True if operation is available, False otherwise
        """
        return self.mps_ops.get(op, False)
    
    def is_feature_available(self, feature: Metal32Feature) -> bool:
        """
        Check if a Metal 3.2 feature is available
        
        Args:
            feature: Metal 3.2 feature
            
        Returns:
            True if feature is available, False otherwise
        """
        return self.metal32_features.get(feature, False)
    
    def run_operation(self, op: MPSOperation, *args, **kwargs) -> Any:
        """
        Run an MPS operation
        
        Args:
            op: MPS operation
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        # Check if operation is available
        if not self.is_operation_available(op):
            raise ValueError(f"MPS operation {op.name} is not available on this hardware")
        
        # Get implementation
        impl = self.op_implementations.get(op)
        if impl is None:
            raise ValueError(f"No implementation for MPS operation {op.name}")
        
        # Run implementation
        return impl(*args, **kwargs)
    
    def _impl_conv2d(self, 
                    input_tensor: Any, 
                    filter_tensor: Any, 
                    stride: Tuple[int, int] = (1, 1),
                    padding: str = "same",
                    dilation: Tuple[int, int] = (1, 1),
                    groups: int = 1) -> Any:
        """
        Implement 2D convolution using MPS
        
        Args:
            input_tensor: Input tensor
            filter_tensor: Filter tensor
            stride: Stride (height, width)
            padding: Padding mode ("same" or "valid")
            dilation: Dilation rate (height, width)
            groups: Number of groups
            
        Returns:
            Convolution result
        """
        # Use MLX's convolution, which is backed by Metal Performance Shaders on Apple Silicon
        if hasattr(mx, "conv"):
            return mx.conv(
                input_tensor, 
                filter_tensor,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups
            )
        else:
            # Fallback to a manual implementation or raise error
            raise NotImplementedError("MLX does not support convolution on this version")
    
    def _impl_depthwise_conv(self,
                           input_tensor: Any,
                           filter_tensor: Any,
                           stride: Tuple[int, int] = (1, 1),
                           padding: str = "same",
                           dilation: Tuple[int, int] = (1, 1)) -> Any:
        """
        Implement depthwise convolution using MPS
        
        Args:
            input_tensor: Input tensor
            filter_tensor: Filter tensor
            stride: Stride (height, width)
            padding: Padding mode ("same" or "valid")
            dilation: Dilation rate (height, width)
            
        Returns:
            Depthwise convolution result
        """
        # Depthwise conv is conv with groups=channels
        channels = input_tensor.shape[-1]
        
        # Use MLX's convolution with groups=channels
        return self._impl_conv2d(
            input_tensor,
            filter_tensor,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels
        )
    
    def _impl_matmul(self, a: Any, b: Any, trans_a: bool = False, trans_b: bool = False) -> Any:
        """
        Implement matrix multiplication using MPS
        
        Args:
            a: First tensor
            b: Second tensor
            trans_a: Transpose first tensor
            trans_b: Transpose second tensor
            
        Returns:
            Matrix multiplication result
        """
        # Handle transposes
        if trans_a:
            a = mx.transpose(a)
        if trans_b:
            b = mx.transpose(b)
        
        # Use MLX's matmul, which is backed by Metal Performance Shaders on Apple Silicon
        return mx.matmul(a, b)
    
    def _impl_batch_norm(self, 
                       input_tensor: Any, 
                       scale: Any, 
                       bias: Any, 
                       mean: Any, 
                       variance: Any,
                       epsilon: float = 1e-5) -> Any:
        """
        Implement batch normalization using MPS
        
        Args:
            input_tensor: Input tensor
            scale: Scale tensor
            bias: Bias tensor
            mean: Mean tensor
            variance: Variance tensor
            epsilon: Epsilon for numerical stability
            
        Returns:
            Batch normalization result
        """
        # Check if MLX has batch norm
        if hasattr(mx, "batch_norm"):
            return mx.batch_norm(input_tensor, scale, bias, mean, variance, epsilon)
        else:
            # Implement manually using MPS-accelerated operations
            normalized = (input_tensor - mean) / mx.sqrt(variance + epsilon)
            return normalized * scale + bias
    
    def _impl_pooling(self, 
                    input_tensor: Any, 
                    pool_size: Tuple[int, int],
                    stride: Tuple[int, int] = None,
                    padding: str = "valid",
                    pool_type: str = "max") -> Any:
        """
        Implement pooling using MPS
        
        Args:
            input_tensor: Input tensor
            pool_size: Pool size (height, width)
            stride: Stride (height, width), default is same as pool_size
            padding: Padding mode ("same" or "valid")
            pool_type: Pooling type ("max" or "average")
            
        Returns:
            Pooling result
        """
        # Use stride=pool_size if not specified
        if stride is None:
            stride = pool_size
        
        # Check pooling type
        if pool_type == "max":
            if hasattr(mx, "max_pool"):
                return mx.max_pool(input_tensor, pool_size, stride, padding)
            else:
                # Implement manually
                raise NotImplementedError("MLX does not support max pooling on this version")
        elif pool_type == "average":
            if hasattr(mx, "avg_pool"):
                return mx.avg_pool(input_tensor, pool_size, stride, padding)
            else:
                # Implement manually
                raise NotImplementedError("MLX does not support average pooling on this version")
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")
    
    def _impl_activation(self, input_tensor: Any, activation_type: str) -> Any:
        """
        Implement activation function using MPS
        
        Args:
            input_tensor: Input tensor
            activation_type: Activation type
            
        Returns:
            Activation result
        """
        # Map activation types to MLX functions
        activation_map = {
            "relu": mx.relu,
            "sigmoid": mx.sigmoid,
            "tanh": mx.tanh,
            "softmax": mx.softmax,
            "gelu": mx.gelu if hasattr(mx, "gelu") else None,
            "selu": mx.selu if hasattr(mx, "selu") else None,
            "swish": lambda x: x * mx.sigmoid(x),
            "mish": lambda x: x * mx.tanh(mx.softplus(x)) if hasattr(mx, "softplus") else None,
        }
        
        # Get activation function
        activation_fn = activation_map.get(activation_type.lower())
        if activation_fn is None:
            raise ValueError(f"Unsupported activation function: {activation_type}")
        
        # Apply activation
        return activation_fn(input_tensor)
    
    def _impl_fft(self, input_tensor: Any, dim: int = -1, norm: str = "backward") -> Any:
        """
        Implement Fast Fourier Transform using MPS
        
        Args:
            input_tensor: Input tensor
            dim: Dimension along which to apply FFT
            norm: Normalization mode
            
        Returns:
            FFT result
        """
        # Check if operation is available for this hardware
        if not self.is_operation_available(MPSOperation.FFT):
            raise NotImplementedError(f"FFT is not available on {self.hardware.chip_generation.name}")
        
        # Check if MLX has FFT implementation
        if hasattr(mx, "fft"):
            return mx.fft(input_tensor, dim, norm)
        else:
            # Could implement using MPS directly, but requires PyObjC and Metal knowledge
            raise NotImplementedError("MLX does not support FFT on this version")
    
    def _impl_scan(self, input_tensor: Any, dim: int = 0, exclusive: bool = False, reverse: bool = False) -> Any:
        """
        Implement scan operations (cumulative sum, etc.) using MPS
        
        Args:
            input_tensor: Input tensor
            dim: Dimension along which to scan
            exclusive: Whether to exclude the current element
            reverse: Whether to scan in reverse order
            
        Returns:
            Scan result
        """
        # Check if operation is available for this hardware
        if not self.is_operation_available(MPSOperation.SCAN):
            raise NotImplementedError(f"Scan operations are not available on {self.hardware.chip_generation.name}")
        
        # Check if MLX has cumsum
        if hasattr(mx, "cumsum"):
            result = mx.cumsum(input_tensor, axis=dim)
            
            # Handle reverse
            if reverse:
                # Reverse the input, compute cumsum, then reverse back
                # Create reverse indices
                indices = list(range(input_tensor.shape[dim]))[::-1]
                
                # Reverse input along specified dimension
                reversed_input = mx.take(input_tensor, mx.array(indices), axis=dim)
                
                # Compute cumsum on reversed input
                result = mx.cumsum(reversed_input, axis=dim)
                
                # Reverse the result back
                result = mx.take(result, mx.array(indices), axis=dim)
            
            # Handle exclusive
            if exclusive:
                # Pad with zeros at the beginning and remove last element
                pad_shape = list(input_tensor.shape)
                pad_shape[dim] = 1
                zeros = mx.zeros(pad_shape)
                
                # Concatenate zeros at the beginning
                result = mx.concatenate([zeros, result], axis=dim)
                
                # Remove last element
                slices = [slice(None)] * len(input_tensor.shape)
                slices[dim] = slice(0, -1)
                result = result[tuple(slices)]
            
            return result
        else:
            raise NotImplementedError("MLX does not support cumulative operations on this version")
    
    def _impl_gemm(self, 
                  a: Any, 
                  b: Any, 
                  c: Any = None, 
                  alpha: float = 1.0, 
                  beta: float = 1.0,
                  trans_a: bool = False, 
                  trans_b: bool = False) -> Any:
        """
        Implement GEMM (General Matrix Multiplication) using MPS
        
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
    
    # Implementations for new operations
    
    def _impl_sparse_matmul(self, 
                          dense_matrix: Any, 
                          sparse_matrix: Any,
                          sparse_indices: Any = None,
                          sparse_values: Any = None,
                          trans_dense: bool = False,
                          trans_sparse: bool = False) -> Any:
        """
        Implement sparse matrix multiplication using MPS
        
        Args:
            dense_matrix: Dense matrix
            sparse_matrix: Sparse matrix or None if using indices/values format
            sparse_indices: Sparse matrix indices (if using COO format)
            sparse_values: Sparse matrix values (if using COO format)
            trans_dense: Transpose dense matrix
            trans_sparse: Transpose sparse matrix
            
        Returns:
            Matrix multiplication result
        """
        # Check feature availability
        if not self.is_feature_available(Metal32Feature.SPARSE_ACCELERATION):
            raise NotImplementedError("Sparse acceleration is not available on this hardware")
        
        # Handle format conversion if needed
        if sparse_matrix is None and sparse_indices is not None and sparse_values is not None:
            # Convert COO format to MLX sparse tensor when supported
            raise NotImplementedError("COO format conversion not implemented")
        
        # For now, we'll use a dense operation instead - future implementation would use Metal's sparse ops
        if trans_dense:
            dense_matrix = mx.transpose(dense_matrix)
        if trans_sparse:
            sparse_matrix = mx.transpose(sparse_matrix)
        
        # MLX doesn't currently expose a sparse matmul operation, so we convert to dense
        # In a full implementation, this would use Metal's sparse operations directly
        return mx.matmul(dense_matrix, sparse_matrix)
    
    def _impl_conv3d(self,
                    input_tensor: Any,
                    filter_tensor: Any,
                    stride: Tuple[int, int, int] = (1, 1, 1),
                    padding: str = "same",
                    dilation: Tuple[int, int, int] = (1, 1, 1)) -> Any:
        """
        Implement 3D convolution using MPS
        
        Args:
            input_tensor: Input tensor
            filter_tensor: Filter tensor
            stride: Stride (depth, height, width)
            padding: Padding mode ("same" or "valid")
            dilation: Dilation rate (depth, height, width)
            
        Returns:
            3D convolution result
        """
        # Check if MLX supports 3D convolution
        if hasattr(mx, "conv") and len(input_tensor.shape) == 5:  # 5D tensor for 3D conv
            return mx.conv(
                input_tensor,
                filter_tensor,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        else:
            # Implement manually using multiple 2D convolutions
            # This is a simplified approach; a full implementation would use Metal directly
            raise NotImplementedError("3D convolution not directly supported by MLX")
    
    def _impl_deconv(self,
                    input_tensor: Any,
                    filter_tensor: Any,
                    stride: Tuple[int, int] = (1, 1),
                    padding: str = "same",
                    output_padding: Tuple[int, int] = (0, 0)) -> Any:
        """
        Implement deconvolution (transposed convolution) using MPS
        
        Args:
            input_tensor: Input tensor
            filter_tensor: Filter tensor
            stride: Stride (height, width)
            padding: Padding mode ("same" or "valid")
            output_padding: Output padding (height, width)
            
        Returns:
            Deconvolution result
        """
        # Check if MLX supports transposed convolution
        if hasattr(mx, "conv_transpose"):
            return mx.conv_transpose(
                input_tensor,
                filter_tensor,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            )
        else:
            # This would require a custom implementation using MPS directly
            raise NotImplementedError("Transposed convolution not directly supported by MLX")
    
    def _impl_dilated_conv(self,
                         input_tensor: Any,
                         filter_tensor: Any,
                         stride: Tuple[int, int] = (1, 1),
                         padding: str = "same",
                         dilation: Tuple[int, int] = (2, 2)) -> Any:
        """
        Implement dilated convolution using MPS
        
        Args:
            input_tensor: Input tensor
            filter_tensor: Filter tensor
            stride: Stride (height, width)
            padding: Padding mode ("same" or "valid")
            dilation: Dilation rate (height, width)
            
        Returns:
            Dilated convolution result
        """
        # MLX's standard conv supports dilation
        return self._impl_conv2d(
            input_tensor,
            filter_tensor,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
    
    def _impl_roi_pooling(self,
                        input_tensor: Any,
                        rois: Any,
                        output_size: Tuple[int, int] = (7, 7),
                        spatial_scale: float = 1.0) -> Any:
        """
        Implement ROI pooling using MPS
        
        Args:
            input_tensor: Input tensor
            rois: Regions of interest
            output_size: Output size (height, width)
            spatial_scale: Spatial scale
            
        Returns:
            ROI pooling result
        """
        # ROI pooling is not directly supported by MLX
        # This would require a custom implementation using MPS directly
        # For now, we provide a simple (inefficient) implementation
        
        # Check feature availability
        if not self.is_feature_available(Metal32Feature.ENHANCED_MATRIX_ENGINE):
            raise NotImplementedError("ROI pooling requires Enhanced Matrix Engine feature")
        
        # Simple implementation that extracts and resizes each ROI
        # This is highly inefficient - a real implementation would use MPS directly
        
        raise NotImplementedError("ROI pooling not implemented")
    
    def _impl_layered_texture_load(self,
                                 texture_array: Any,
                                 indices: Any) -> Any:
        """
        Implement layered texture load using Metal 3.2 features
        
        Args:
            texture_array: Texture array
            indices: Layer indices
            
        Returns:
            Loaded texture data
        """
        # Check feature availability
        if not self.is_feature_available(Metal32Feature.LAYERED_TEXTURES):
            raise NotImplementedError("Layered textures not available on this hardware")
        
        # This would require direct Metal access, not possible through MLX
        raise NotImplementedError("Layered texture load requires direct Metal API access")
    
    def _impl_matrix_patch_op(self,
                            matrix: Any,
                            patches: Any,
                            patch_indices: Any) -> Any:
        """
        Implement matrix patching operation using Metal 3.2
        
        Args:
            matrix: Input matrix
            patches: Patches to apply
            patch_indices: Indices for patches
            
        Returns:
            Patched matrix
        """
        # Check feature availability
        if not self.is_feature_available(Metal32Feature.MATRIX_PATCHING):
            raise NotImplementedError("Matrix patching not available on this hardware")
        
        # For now, implement using standard MLX operations
        # A real implementation would use Metal's optimization for dynamic patching
        
        result = mx.array(matrix)  # Make a copy
        
        # Apply each patch
        for i in range(len(patches)):
            indices = patch_indices[i]
            patch = patches[i]
            
            # Extract indices
            if len(indices) == 2:
                row_start, col_start = indices
                row_end = row_start + patch.shape[0]
                col_end = col_start + patch.shape[1]
                
                # Update the region
                # Note: This is inefficient as it creates intermediate arrays
                # A Metal implementation would do this in-place
                result = mx.scatter(result, indices=(slice(row_start, row_end), slice(col_start, col_end)), values=patch)
        
        return result
    
    def _impl_transposed_conv(self,
                            input_tensor: Any,
                            filter_tensor: Any,
                            stride: Tuple[int, int] = (1, 1),
                            padding: str = "same",
                            output_padding: Tuple[int, int] = (0, 0)) -> Any:
        """
        Implement transposed convolution using MPS
        
        Args:
            input_tensor: Input tensor
            filter_tensor: Filter tensor
            stride: Stride (height, width)
            padding: Padding mode ("same" or "valid")
            output_padding: Output padding (height, width)
            
        Returns:
            Transposed convolution result
        """
        # This is an alias for deconv
        return self._impl_deconv(
            input_tensor,
            filter_tensor,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

class MLXMetalKernelMapping:
    """Maps Triton kernels to Metal Performance Shaders where possible"""
    
    def __init__(self, hardware_capabilities=None):
        """
        Initialize MLX Metal kernel mapping
        
        Args:
            hardware_capabilities: Optional hardware capabilities instance
        """
        self.hardware = hardware_capabilities or metal_hardware_optimizer.hardware_capabilities
        self.mps = MPSIntegration(hardware_capabilities)
        self.kernel_patterns = self._initialize_kernel_patterns()
        
    def _initialize_kernel_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize kernel patterns to recognize
        
        Returns:
            Dictionary of kernel patterns
        """
        patterns = {
            # Pattern for matrix multiplication
            "matmul": {
                "operations": ["matmul"],
                "mps_op": MPSOperation.MATMUL,
                "params": ["a", "b"]
            },
            
            # Pattern for GEMM (matrix multiply with bias)
            "gemm": {
                "operations": ["matmul", "add"],
                "mps_op": MPSOperation.GEMM,
                "params": ["a", "b", "c"]
            },
            
            # Pattern for convolution
            "conv2d": {
                "operations": ["conv"],
                "mps_op": MPSOperation.CONV2D,
                "params": ["input", "filter", "stride", "padding"]
            },
            
            # Pattern for batch normalization
            "batch_norm": {
                "operations": ["sub", "div", "mul", "add"],
                "mps_op": MPSOperation.BATCH_NORM,
                "params": ["input", "scale", "bias", "mean", "var"]
            },
            
            # Pattern for max pooling
            "max_pool": {
                "operations": ["max_reduce"],
                "mps_op": MPSOperation.POOLING,
                "params": ["input", "pool_size", "stride", "padding"]
            },
            
            # Pattern for ReLU activation
            "relu": {
                "operations": ["max"],
                "mps_op": MPSOperation.ACTIVATION,
                "params": ["input"],
                "additional": {"activation_type": "relu"}
            },
            
            # Pattern for softmax
            "softmax": {
                "operations": ["exp", "sum", "div"],
                "mps_op": MPSOperation.ACTIVATION,
                "params": ["input"],
                "additional": {"activation_type": "softmax"}
            },
            
            # New patterns for Metal 3.2 operations
            
            # Sparse matrix multiplication pattern
            "sparse_matmul": {
                "operations": ["sparse_matmul"],
                "mps_op": MPSOperation.SPARSE_MATMUL,
                "params": ["dense", "sparse", "indices", "values"],
                "min_hardware_gen": AppleSiliconGeneration.M3
            },
            
            # 3D convolution pattern
            "conv3d": {
                "operations": ["conv3d"],
                "mps_op": MPSOperation.CONV3D,
                "params": ["input", "filter", "stride", "padding", "dilation"],
                "min_hardware_gen": AppleSiliconGeneration.M2
            },
            
            # Transposed convolution pattern
            "transposed_conv": {
                "operations": ["transpose_conv"],
                "mps_op": MPSOperation.TRANSPOSED_CONV,
                "params": ["input", "filter", "stride", "padding", "output_padding"],
                "min_hardware_gen": AppleSiliconGeneration.M3
            },
            
            # Dilated convolution pattern
            "dilated_conv": {
                "operations": ["dilated_conv"],
                "mps_op": MPSOperation.DILATED_CONV,
                "params": ["input", "filter", "stride", "padding", "dilation"],
                "min_hardware_gen": AppleSiliconGeneration.M2
            },
            
            # ROI pooling pattern
            "roi_pooling": {
                "operations": ["roi_pool"],
                "mps_op": MPSOperation.ROI_POOLING,
                "params": ["input", "rois", "output_size", "spatial_scale"],
                "min_hardware_gen": AppleSiliconGeneration.M3
            },
            
            # Matrix patch operation pattern
            "matrix_patch": {
                "operations": ["patch"],
                "mps_op": MPSOperation.MATRIX_PATCH_OP,
                "params": ["matrix", "patches", "indices"],
                "min_hardware_gen": AppleSiliconGeneration.M3
            }
        }
        
        # Add M3-specific patterns
        if self.hardware.chip_generation.value >= AppleSiliconGeneration.M3.value:
            # FFT pattern (optimized for M3 and newer)
            patterns["fft"] = {
                "operations": ["fft"],
                "mps_op": MPSOperation.FFT,
                "params": ["input", "dim", "norm"]
            }
            
            # Scan operations pattern (optimized for M3 and newer)
            patterns["scan"] = {
                "operations": ["scan"],
                "mps_op": MPSOperation.SCAN,
                "params": ["input", "dim", "exclusive", "reverse"]
            }
            
            # Flash attention pattern (optimized for M3)
            patterns["flash_attention"] = {
                "operations": ["flash_attention"],
                "mps_op": MPSOperation.ATTENTION,
                "params": ["query", "key", "value", "scale", "mask", "dropout_p"]
            }
        
        return patterns
    
    def match_kernel_pattern(self, ops: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Match kernel operations to a pattern
        
        Args:
            ops: List of operations
            
        Returns:
            Matched pattern or None if no match
        """
        # Extract operation types
        op_types = [op.get("type", "").split(".")[-1] for op in ops]
        
        # Try to match each pattern
        for pattern_name, pattern in self.kernel_patterns.items():
            pattern_ops = pattern["operations"]
            
            # Check hardware requirements
            min_hardware_gen = pattern.get("min_hardware_gen")
            if min_hardware_gen is not None and self.hardware.chip_generation.value < min_hardware_gen.value:
                # Skip patterns that require newer hardware
                continue
            
            # Check if operations match the pattern
            if len(op_types) >= len(pattern_ops):
                for i in range(len(op_types) - len(pattern_ops) + 1):
                    if op_types[i:i+len(pattern_ops)] == pattern_ops:
                        # Found a match
                        return {
                            "name": pattern_name,
                            "start_idx": i,
                            "end_idx": i + len(pattern_ops),
                            "mps_op": pattern["mps_op"],
                            "params": pattern["params"],
                            "additional": pattern.get("additional", {})
                        }
        
        return None
    
    def extract_params(self, matched_pattern: Dict[str, Any], ops: List[Dict]) -> Dict[str, Any]:
        """
        Extract parameters for MPS operation
        
        Args:
            matched_pattern: Matched pattern
            ops: List of operations
            
        Returns:
            Dictionary of parameters
        """
        params = {}
        
        # Add additional parameters from pattern
        params.update(matched_pattern.get("additional", {}))
        
        # Extract parameters based on the pattern
        start_idx = matched_pattern["start_idx"]
        end_idx = matched_pattern["end_idx"]
        
        # Get operations for this pattern
        pattern_ops = ops[start_idx:end_idx]
        
        # Pattern-specific parameter extraction
        pattern_name = matched_pattern["name"]
        
        if pattern_name == "matmul":
            # Extract matrices for matrix multiplication
            if len(pattern_ops) > 0:
                matmul_op = pattern_ops[0]
                params["a"] = matmul_op.get("a_id")
                params["b"] = matmul_op.get("b_id")
                params["trans_a"] = matmul_op.get("trans_a", False)
                params["trans_b"] = matmul_op.get("trans_b", False)
                
        elif pattern_name == "gemm":
            # Extract parameters for GEMM
            if len(pattern_ops) > 0:
                matmul_op = pattern_ops[0]
                params["a"] = matmul_op.get("a_id")
                params["b"] = matmul_op.get("b_id")
                params["trans_a"] = matmul_op.get("trans_a", False)
                params["trans_b"] = matmul_op.get("trans_b", False)
            
            if len(pattern_ops) > 1:
                add_op = pattern_ops[1]
                # Determine which operand is the bias
                lhs_id = add_op.get("lhs_id")
                rhs_id = add_op.get("rhs_id")
                if lhs_id == matmul_op.get("id"):
                    params["c"] = rhs_id
                else:
                    params["c"] = lhs_id
        
        elif pattern_name == "conv2d":
            # Extract parameters for 2D convolution
            if len(pattern_ops) > 0:
                conv_op = pattern_ops[0]
                params["input"] = conv_op.get("input_id")
                params["filter"] = conv_op.get("filter_id")
                params["stride"] = conv_op.get("stride", (1, 1))
                params["padding"] = conv_op.get("padding", "same")
                params["dilation"] = conv_op.get("dilation", (1, 1))
                params["groups"] = conv_op.get("groups", 1)
        
        # New patterns for Metal 3.2 operations
        
        elif pattern_name == "sparse_matmul":
            # Extract parameters for sparse matrix multiplication
            if len(pattern_ops) > 0:
                sparse_op = pattern_ops[0]
                params["dense"] = sparse_op.get("dense_id")
                params["sparse"] = sparse_op.get("sparse_id")
                params["indices"] = sparse_op.get("indices_id")
                params["values"] = sparse_op.get("values_id")
                params["trans_dense"] = sparse_op.get("trans_dense", False)
                params["trans_sparse"] = sparse_op.get("trans_sparse", False)
        
        elif pattern_name == "conv3d":
            # Extract parameters for 3D convolution
            if len(pattern_ops) > 0:
                conv3d_op = pattern_ops[0]
                params["input"] = conv3d_op.get("input_id")
                params["filter"] = conv3d_op.get("filter_id")
                params["stride"] = conv3d_op.get("stride", (1, 1, 1))
                params["padding"] = conv3d_op.get("padding", "same")
                params["dilation"] = conv3d_op.get("dilation", (1, 1, 1))
        
        elif pattern_name == "transposed_conv":
            # Extract parameters for transposed convolution
            if len(pattern_ops) > 0:
                tconv_op = pattern_ops[0]
                params["input"] = tconv_op.get("input_id")
                params["filter"] = tconv_op.get("filter_id")
                params["stride"] = tconv_op.get("stride", (1, 1))
                params["padding"] = tconv_op.get("padding", "same")
                params["output_padding"] = tconv_op.get("output_padding", (0, 0))
        
        elif pattern_name == "dilated_conv":
            # Extract parameters for dilated convolution
            if len(pattern_ops) > 0:
                dconv_op = pattern_ops[0]
                params["input"] = dconv_op.get("input_id")
                params["filter"] = dconv_op.get("filter_id")
                params["stride"] = dconv_op.get("stride", (1, 1))
                params["padding"] = dconv_op.get("padding", "same")
                params["dilation"] = dconv_op.get("dilation", (2, 2))
        
        elif pattern_name == "roi_pooling":
            # Extract parameters for ROI pooling
            if len(pattern_ops) > 0:
                roi_op = pattern_ops[0]
                params["input"] = roi_op.get("input_id")
                params["rois"] = roi_op.get("rois_id")
                params["output_size"] = roi_op.get("output_size", (7, 7))
                params["spatial_scale"] = roi_op.get("spatial_scale", 1.0)
        
        elif pattern_name == "matrix_patch":
            # Extract parameters for matrix patching
            if len(pattern_ops) > 0:
                patch_op = pattern_ops[0]
                params["matrix"] = patch_op.get("matrix_id")
                params["patches"] = patch_op.get("patches_id")
                params["indices"] = patch_op.get("indices_id")
        
        elif pattern_name == "flash_attention":
            # Extract parameters for flash attention
            if len(pattern_ops) > 0:
                attn_op = pattern_ops[0]
                params["query"] = attn_op.get("query_id")
                params["key"] = attn_op.get("key_id")
                params["value"] = attn_op.get("value_id")
                params["scale"] = attn_op.get("scale")
                params["mask"] = attn_op.get("mask_id")
                params["dropout_p"] = attn_op.get("dropout_p", 0.0)
                
        return params
    
    def execute_with_mps(self, 
                        context: Dict[str, Any], 
                        matched_pattern: Dict[str, Any], 
                        params: Dict[str, Any]) -> Any:
        """
        Execute operation using Metal Performance Shaders
        
        Args:
            context: Execution context with tensors
            matched_pattern: Matched pattern
            params: Extracted parameters
            
        Returns:
            Operation result
        """
        # Get MPS operation from pattern
        mps_op = matched_pattern["mps_op"]
        
        # Check if operation is available
        if not self.mps.is_operation_available(mps_op):
            raise ValueError(f"MPS operation {mps_op.name} is not available on this hardware")
        
        # Prepare arguments for MPS operation
        mps_args = []
        mps_kwargs = {}
        
        # Convert parameters to actual tensors from context
        for param_name, param_id in params.items():
            if param_name in matched_pattern["params"]:
                # This is a positional parameter
                if isinstance(param_id, str) and param_id in context:
                    mps_args.append(context[param_id])
                else:
                    # Use parameter directly if it's not a reference to context
                    mps_args.append(param_id)
            else:
                # This is a keyword parameter
                if isinstance(param_id, str) and param_id in context:
                    mps_kwargs[param_name] = context[param_id]
                else:
                    # Use parameter directly if it's not a reference to context
                    mps_kwargs[param_name] = param_id
        
        # Execute operation with MPS
        return self.mps.run_operation(mps_op, *mps_args, **mps_kwargs)


# Create global instance
mps_integration = MPSIntegration(hardware_capabilities)
mlx_metal_kernel_mapping = MLXMetalKernelMapping(hardware_capabilities) 