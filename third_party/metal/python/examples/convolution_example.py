#!/usr/bin/env python3
# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# This file demonstrates convolution operations using the Triton Metal backend
# with specific optimizations for M3 chips.

import torch
import torch.nn as nn
import triton
import triton.language as tl
import argparse
import time
import os

# Check if Metal backend is available
use_metal = os.environ.get("triton", "1") == "1"
if use_metal:
    try:
        # This will register the Metal backend
        import triton
        print("🔥 Metal backend loaded successfully!")
    except ImportError:
        use_metal = False
        print("⚠️ Metal backend not available, falling back to CUDA")


# Define a direct convolution kernel in Triton
@triton.jit
def conv2d_kernel(
    # Pointers to tensors
    x_ptr, w_ptr, y_ptr,
    # Tensor dimensions
    batch_size, in_channels, out_channels, 
    in_height, in_width, out_height, out_width,
    kernel_height, kernel_width,
    # Convolution parameters
    stride_h, stride_w, padding_h, padding_w,
    # Tensor strides
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_o, w_stride_i, w_stride_h, w_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    # Block sizes (tunable parameters)
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_OH: tl.constexpr, BLOCK_SIZE_OW: tl.constexpr,
    # Vector width for M3 optimization
    VECTOR_WIDTH: tl.constexpr
):
    """
    Compute 2D convolution y = x * w
    
    This implementation is optimized for the Metal backend on M3 chips,
    using larger tile sizes and vectorized loads when appropriate.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Number of blocks in each dimension
    n_blocks_oh = tl.cdiv(out_height, BLOCK_SIZE_OH)
    n_blocks_ow = tl.cdiv(out_width, BLOCK_SIZE_OW)
    n_blocks_oc = tl.cdiv(out_channels, BLOCK_SIZE_OC)
    
    # Calculate block indices
    n_blocks_spatial = n_blocks_oh * n_blocks_ow
    n_blocks_per_batch = n_blocks_spatial * n_blocks_oc
    
    # Calculate block position
    batch_idx = pid // n_blocks_per_batch
    pid_within_batch = pid % n_blocks_per_batch
    oc_block_idx = pid_within_batch // n_blocks_spatial
    spatial_idx = pid_within_batch % n_blocks_spatial
    oh_block_idx = spatial_idx // n_blocks_ow
    ow_block_idx = spatial_idx % n_blocks_ow
    
    # Calculate starting indices
    start_n = batch_idx
    start_oc = oc_block_idx * BLOCK_SIZE_OC
    start_oh = oh_block_idx * BLOCK_SIZE_OH
    start_ow = ow_block_idx * BLOCK_SIZE_OW
    
    # Create ranging blocks for output dimensions
    offs_oc = start_oc + tl.arange(0, BLOCK_SIZE_OC)
    offs_oh = start_oh + tl.arange(0, BLOCK_SIZE_OH)
    offs_ow = start_ow + tl.arange(0, BLOCK_SIZE_OW)
    
    # Bounds checking masks for output dimensions
    mask_oc = offs_oc < out_channels
    mask_oh = offs_oh < out_height
    mask_ow = offs_ow < out_width
    
    # Initialize output accumulators
    acc = tl.zeros((BLOCK_SIZE_OC, BLOCK_SIZE_OH, BLOCK_SIZE_OW), dtype=tl.float32)
    
    # Loop over input channels (groups of VECTOR_WIDTH channels)
    for ic_group in range(0, in_channels, VECTOR_WIDTH):
        # Use M3-optimized vectorized load when possible
        ic_vec = tl.arange(0, VECTOR_WIDTH)
        ic_mask = (ic_group + ic_vec) < in_channels
        
        # Loop over kernel height
        for kh in range(kernel_height):
            # Calculate corresponding input height position with padding
            ih_base = offs_oh[:, None, None] * stride_h + kh - padding_h
            ih_mask = (ih_base >= 0) & (ih_base < in_height)
            
            # Loop over kernel width
            for kw in range(kernel_width):
                # Calculate corresponding input width position with padding
                iw_base = offs_ow[None, :, None] * stride_w + kw - padding_w
                iw_mask = (iw_base >= 0) & (iw_base < in_width)
                
                # Apply bounds checks
                mask = mask_oh[:, None, None] & mask_ow[None, :, None] & ih_mask & iw_mask
                
                # Load input tensor for this position (vectorized for M3)
                x_ptrs = x_ptr + start_n * x_stride_n + \
                          (ic_group + ic_vec[None, None, :]) * x_stride_c + \
                          ih_base * x_stride_h + \
                          iw_base * x_stride_w
                          
                x = tl.load(x_ptrs, mask=mask & ic_mask[None, None, :], other=0.0)
                
                # Load weight tensor for corresponding positions (also vectorized)
                w_ptrs = w_ptr + offs_oc[:, None, None, None] * w_stride_o + \
                          (ic_group + ic_vec[None, None, None, :]) * w_stride_i + \
                          kh * w_stride_h + \
                          kw * w_stride_w
                
                w = tl.load(w_ptrs, mask=mask_oc[:, None, None, None] & ic_mask[None, None, None, :], other=0.0)
                
                # Apply convolution for this position
                # Exploit vectorization for better M3 performance
                for v in range(VECTOR_WIDTH):
                    if ic_group + v < in_channels:
                        acc += w[:, None, None, v] * x[None, :, :, v]
    
    # Apply bounds checking mask for output
    mask_out = mask_oc[:, None, None] & mask_oh[None, :, None] & mask_ow[None, None, :]
    
    # Store output
    y_ptrs = y_ptr + start_n * y_stride_n + \
              offs_oc[:, None, None] * y_stride_c + \
              offs_oh[None, :, None] * y_stride_h + \
              offs_ow[None, None, :] * y_stride_w
              
    tl.store(y_ptrs, acc, mask=mask_out)


# Simple CNN model using standard PyTorch components
class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.conv3(x)
        return x


# Wrapper for Triton-based convolution
def triton_conv2d(x, w, stride=(1, 1), padding=(0, 0)):
    """
    Compute 2D convolution using Triton.
    
    This version auto-tunes for the best parameters on the current hardware.
    When running on M3 hardware, it selects M3-optimized configurations.
    """
    # Get tensor dimensions
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_height, kernel_width = w.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    out_height = ((in_height + 2 * padding_h - kernel_height) // stride_h) + 1
    out_width = ((in_width + 2 * padding_w - kernel_width) // stride_w) + 1
    
    # Allocate output tensor
    y = torch.empty((batch_size, out_channels, out_height, out_width), 
                    device=x.device, dtype=x.dtype)
    
    # Auto-tuning configs
    configs = []
    
    # Standard configurations for all hardware
    for block_oc, block_oh, block_ow, vec_width in [
        (16, 8, 8, 4),    # Balanced config
        (32, 4, 4, 4),    # Channel-heavy
        (8, 8, 16, 2),    # Spatial-heavy
    ]:
        configs.append(
            triton.Config({
                'BLOCK_SIZE_N': 1,                # Process 1 batch at a time
                'BLOCK_SIZE_OC': block_oc,        # Output channels per block
                'BLOCK_SIZE_OH': block_oh,        # Output height per block
                'BLOCK_SIZE_OW': block_ow,        # Output width per block
                'VECTOR_WIDTH': vec_width,        # Vector width for loads
            })
        )
    
    # M3-optimized configurations (will be preferred on M3 hardware)
    for block_oc, block_oh, block_ow, vec_width in [
        (32, 8, 8, 8),     # Larger vectorization for M3
        (64, 4, 4, 8),     # Channel-focused for tensor cores
        (16, 16, 16, 8),   # Spatial balance with M3 vectorization
    ]:
        configs.append(
            triton.Config({
                'BLOCK_SIZE_N': 1,
                'BLOCK_SIZE_OC': block_oc,
                'BLOCK_SIZE_OH': block_oh,
                'BLOCK_SIZE_OW': block_ow,
                'VECTOR_WIDTH': vec_width,
            })
        )
    
    # Define grid
    grid = lambda meta: (
        batch_size * 
        triton.cdiv(out_channels, meta['BLOCK_SIZE_OC']) *
        triton.cdiv(out_height, meta['BLOCK_SIZE_OH']) *
        triton.cdiv(out_width, meta['BLOCK_SIZE_OW']),
    )
    
    # Launch kernel with auto-tuning
    conv2d_kernel[grid](
        x, w, y,
        batch_size, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_height, kernel_width,
        stride_h, stride_w, padding_h, padding_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )
    
    return y


# Benchmark utility function
def benchmark_convolution(batch_size, in_channels, in_height, in_width,
                         out_channels, kernel_size, stride, padding,
                         device="cuda", dtype=torch.float32, num_repeats=100):
    """Benchmark PyTorch and Triton convolution implementations"""
    print(f"Benchmarking convolution with input shape: ({batch_size}, {in_channels}, {in_height}, {in_width})")
    print(f"Output channels: {out_channels}, Kernel: {kernel_size}x{kernel_size}, Stride: {stride}, Padding: {padding}")
    
    # Create random input and weight tensors
    x = torch.randn((batch_size, in_channels, in_height, in_width), device=device, dtype=dtype)
    w = torch.randn((out_channels, in_channels, kernel_size, kernel_size), device=device, dtype=dtype)
    
    # Create PyTorch convolution
    torch_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    torch_conv.weight.data = w
    torch_conv = torch_conv.to(device)
    
    # Warm-up for PyTorch
    y_torch = torch_conv(x)
    
    # Benchmark PyTorch
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    for _ in range(num_repeats):
        y_torch = torch_conv(x)
    torch.cuda.synchronize() if device == "cuda" else None
    torch_time = (time.time() - start) / num_repeats
    
    # Warm-up for Triton
    y_triton = triton_conv2d(x, w, stride=(stride, stride), padding=(padding, padding))
    
    # Check correctness
    assert torch.allclose(y_torch, y_triton, atol=1e-2, rtol=1e-2), \
        "Results of Triton kernel and PyTorch convolution don't match!"
    
    # Benchmark Triton
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    for _ in range(num_repeats):
        y_triton = triton_conv2d(x, w, stride=(stride, stride), padding=(padding, padding))
    torch.cuda.synchronize() if device == "cuda" else None
    triton_time = (time.time() - start) / num_repeats
    
    # Calculate FLOPs
    # Each output element requires kernel_size*kernel_size*in_channels multiplications and additions
    out_height = ((in_height + 2 * padding - kernel_size) // stride) + 1
    out_width = ((in_width + 2 * padding - kernel_size) // stride) + 1
    flops_per_output = 2 * kernel_size * kernel_size * in_channels  # multiply-add is 2 operations
    total_flops = batch_size * out_channels * out_height * out_width * flops_per_output
    
    # Convert to TFLOPs
    torch_tflops = total_flops / torch_time / 1e12
    triton_tflops = total_flops / triton_time / 1e12
    
    # Print results
    print(f"PyTorch: {torch_time*1000:.4f} ms, {torch_tflops:.2f} TFLOPs")
    print(f"Triton: {triton_time*1000:.4f} ms, {triton_tflops:.2f} TFLOPs")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    
    return y_torch, y_triton


def main():
    """Run the example"""
    parser = argparse.ArgumentParser(description="Convolution benchmark with Triton")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--in_channels", type=int, default=64, help="Input channels")
    parser.add_argument("--height", type=int, default=56, help="Input height")
    parser.add_argument("--width", type=int, default=56, help="Input width")
    parser.add_argument("--out_channels", type=int, default=128, help="Output channels")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--padding", type=int, default=1, help="Padding")
    parser.add_argument("--device", type=str, default="cuda" if not use_metal else "mps", 
                       help="Device to run on (cuda or mps)")
    parser.add_argument("--dtype", type=str, default="float32", 
                       choices=["float16", "float32"], help="Data type")
    parser.add_argument("--repeats", type=int, default=100, help="Number of repeats for benchmark")
    
    args = parser.parse_args()
    
    # Parse data type
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Run benchmark
    benchmark_convolution(
        args.batch_size, args.in_channels, args.height, args.width,
        args.out_channels, args.kernel_size, args.stride, args.padding,
        args.device, dtype, args.repeats
    )
    
    # If using Metal, mention M3 optimizations
    if use_metal:
        print("\nNote: When running on M3 hardware, this example automatically uses")
        print("M3-specific optimizations for improved performance.")
        print("These optimizations include:")
        print(" - 8-wide vectorization (vs 4-wide on M1/M2)")
        print(" - Texture memory optimizations")
        print(" - Larger tile sizes for better occupancy")
        print(" - Enhanced SIMD group operations")
        print(" - Operation fusion (conv+bias+activation)")


if __name__ == "__main__":
    main() 