#!/usr/bin/env python
"""
Sample Triton kernel with reduction operations.

This kernel demonstrates different types of reduction operations
that would benefit from the COALESCED memory layout in the Metal backend.
"""

import triton
import triton.language as tl

@triton.jit
def reduction_sample_kernel(
    # Pointers to input and output arrays
    x_ptr, y_ptr, z_ptr, output_ptr,
    # Input tensor dimensions
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # Reduction axis
    axis: tl.constexpr,
    # Block dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    # Stride variables
    stride_xm, stride_xn, stride_ym, stride_yn, stride_zm, stride_zn, stride_out
):
    """
    Sample kernel with different reduction operations
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block start indices
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Offset pointers
    x_offset = m_start * stride_xm + n_start * stride_xn
    y_offset = m_start * stride_ym + n_start * stride_yn
    z_offset = m_start * stride_zm + n_start * stride_zn
    
    # Create block indices
    m_range = tl.arange(0, BLOCK_M)
    n_range = tl.arange(0, BLOCK_N)
    
    # Compute block offsets 
    m_offsets = m_range[:, None] * stride_xm
    n_offsets = n_range[None, :] * stride_xn
    
    # Block coordinates
    m_coor = m_start + m_range
    n_coor = n_start + n_range
    
    # Load data
    x_ptrs = x_ptr + x_offset + m_offsets + n_offsets
    mask = (m_coor < M)[:, None] & (n_coor < N)[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Load y for additional operations
    y_ptrs = y_ptr + y_offset + m_offsets + n_offsets
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    
    # Load z for additional operations
    z_ptrs = z_ptr + z_offset + m_offsets + n_offsets
    z = tl.load(z_ptrs, mask=mask, other=0.0)
    
    # ===== Example 1: Sum reduction along axis 0 =====
    # This will use COALESCED layout
    if axis == 0:
        sum_result = tl.sum(x, axis=0)
        # Store the result
        if pid_m == 0:
            tl.store(output_ptr + n_start, sum_result)
            
    # ===== Example 2: Sum reduction along axis 1 =====
    # This will use COALESCED layout
    elif axis == 1:
        sum_result = tl.sum(x, axis=1)
        # Store the result
        if pid_n == 0:
            tl.store(output_ptr + m_start, sum_result)
            
    # ===== Example 3: Mean reduction =====
    # This will use COALESCED layout
    elif axis == 2:
        mean_result = tl.mean(x, axis=0)
        # Store the result
        if pid_m == 0:
            tl.store(output_ptr + n_start, mean_result)
            
    # ===== Example 4: Max reduction =====
    # This will use COALESCED layout
    elif axis == 3:
        max_result = tl.max(x, axis=0)
        # Store the result
        if pid_m == 0:
            tl.store(output_ptr + n_start, max_result)
            
    # ===== Example 5: Multi-axis reduction =====
    # This combines multiple reductions and will use COALESCED layout
    elif axis == 4:
        # First reduce along axis 0
        temp = tl.sum(x, axis=0)
        # Then reduce along axis 1 (original axis 1)
        final = tl.sum(temp)
        # Store the result
        if pid_m == 0 and pid_n == 0:
            tl.store(output_ptr, final)
            
    # ===== Example 6: ArgMax reduction =====
    # This will use COALESCED layout
    elif axis == 5:
        argmax_result = tl.argmax(x, axis=1)
        # Store the result
        if pid_n == 0:
            tl.store(output_ptr + m_start, argmax_result)
    
    # ===== Example 7: Partial reduction with elementwise ops =====
    # This demonstrates a more complex pattern with reduction
    elif axis == 6:
        # Elementwise operation first
        temp = x * y + z
        # Then reduce
        sum_result = tl.sum(temp, axis=0)
        # Store the result
        if pid_m == 0:
            tl.store(output_ptr + n_start, sum_result)
            
    # ===== Example 8: Softmax-like pattern =====
    # This demonstrates a pattern similar to softmax with multiple reductions
    elif axis == 7:
        # Max for numerical stability
        row_max = tl.max(x, axis=1)
        # Broadcast and subtract
        x_stable = x - row_max[:, None]
        # Exp
        x_exp = tl.exp(x_stable)
        # Sum for normalization
        row_sum = tl.sum(x_exp, axis=1)
        # Normalize
        softmax_result = x_exp / row_sum[:, None]
        # Store the result (just the first value for simplicity)
        tl.store(output_ptr + m_start * stride_out + n_start * 1, softmax_result[0, 0])


def invoke_reduction_kernel(x, y, z, axis=0):
    """
    Helper function to invoke the reduction kernel
    
    Args:
        x: Input tensor
        y: Second input tensor
        z: Third input tensor
        axis: Reduction axis
        
    Returns:
        Output tensor after reduction
    """
    import numpy as np
    
    # Get tensor shapes
    M, N = x.shape
    
    # Compute output shape based on axis
    if axis in [0, 2, 3]:
        output_shape = (N,)
    elif axis in [1, 5]:
        output_shape = (M,)
    elif axis == 4:
        output_shape = (1,)
    elif axis in [6]:
        output_shape = (N,)
    elif axis == 7:
        output_shape = (M, N)  # Full output for softmax
    
    # Create output array
    output = np.zeros(output_shape, dtype=np.float32)
    
    # Get strides
    stride_xm, stride_xn = x.strides[0] // 4, x.strides[1] // 4
    stride_ym, stride_yn = y.strides[0] // 4, y.strides[1] // 4
    stride_zm, stride_zn = z.strides[0] // 4, z.strides[1] // 4
    
    # Get output stride
    if output.ndim == 1:
        stride_out = output.strides[0] // 4
    else:
        stride_out = output.strides[0] // 4
    
    # Determine grid and block sizes
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch the kernel
    reduction_sample_kernel[grid](
        x, y, z, output,
        M, N, 1,  # K is not used in this example
        axis,
        BLOCK_M, BLOCK_N,
        stride_xm, stride_xn, stride_ym, stride_yn, stride_zm, stride_zn, stride_out
    )
    
    return output


if __name__ == "__main__":
    print("This is a sample Triton kernel with various reduction operations")
    print("that benefit from the COALESCED memory layout in the Metal backend.")
    print("\nUse the analyze_memory_layouts.py tool to analyze this kernel:")
    print("python tools/analyze_memory_layouts.py --file tools/sample_kernel.py --verbose") 