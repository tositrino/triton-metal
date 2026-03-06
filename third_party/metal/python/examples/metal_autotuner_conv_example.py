#!/usr/bin/env python
"""
Example demonstrating the use of the Metal auto-tuner with a convolution kernel
"""

import os
import sys
import time
import numpy as np
import argparse

# Add parent directory to path to allow importing from metal_auto_tuner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MLX.metal_auto_tuner import (
    MetalAutoTuner,
    TunableParam,
    ParamType,
    ConfigurationResult,
    get_conv_metal_params
)

# Import Triton if available
try:
    import triton
    import triton.language as tl
    from triton.runtime import driver
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton not available. Will run a simulated example.")

# Convolution kernel
@triton.jit
def conv2d_kernel(
    # Pointers to tensors
    input_ptr, filter_ptr, output_ptr,
    # Dimensions
    batch_size, in_h, in_w, in_c,   # Input dimensions
    out_h, out_w, out_c,            # Output dimensions
    filter_h, filter_w,             # Filter dimensions
    stride_h, stride_w,             # Stride
    padding_h, padding_w,           # Padding
    # Strides for each tensor
    input_batch_stride, input_h_stride, input_w_stride, input_c_stride,
    filter_out_stride, filter_h_stride, filter_w_stride, filter_in_stride,
    output_batch_stride, output_h_stride, output_w_stride, output_c_stride,
    # Block dimensions (tunable)
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr, BLOCK_Z: tl.constexpr, BLOCK_C: tl.constexpr,
    # Other tunable parameters
    FILTER_TILE_SIZE: tl.constexpr,
):
    """2D convolution kernel"""
    # Compute output pixel position
    pid_batch = tl.program_id(axis=0)
    pid_out_y = tl.program_id(axis=1)
    pid_out_x = tl.program_id(axis=2)
    
    # Compute output channel and spatial location
    out_c_offset = pid_out_y * BLOCK_Y
    out_h_offset = pid_out_x * BLOCK_X
    
    # Compute input location
    in_h_start = out_h_offset * stride_h - padding_h
    in_w_start = out_c_offset * stride_w - padding_w
    
    # Compute filter position
    filter_offset = tl.arange(0, BLOCK_C)
    
    # Initialize output accumulator
    output = tl.zeros((BLOCK_X, BLOCK_Y, BLOCK_C), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for ic_block in range(0, tl.cdiv(in_c, BLOCK_Z)):
        in_c_base = ic_block * BLOCK_Z
        
        # Loop over filter height
        for fh in range(0, filter_h, FILTER_TILE_SIZE):
            # Loop over filter width
            for fw in range(0, filter_w, FILTER_TILE_SIZE):
                # Compute actual filter height and width for this tile
                f_h_size = min(FILTER_TILE_SIZE, filter_h - fh)
                f_w_size = min(FILTER_TILE_SIZE, filter_w - fw)
                
                # Iterate over the filter tile and input patches
                for ofh in range(f_h_size):
                    for ofw in range(f_w_size):
                        for oz in range(min(BLOCK_Z, in_c - in_c_base)):
                            # Load input values
                            in_h_pos = in_h_start + fh + ofh
                            in_w_pos = in_w_start + fw + ofw
                            in_c_pos = in_c_base + oz
                            
                            # Generate spatial offsets for the blocks
                            in_h_offsets = tl.arange(0, BLOCK_X)
                            in_w_offsets = tl.arange(0, BLOCK_Y)
                            
                            # Create masks for valid input positions
                            mask_h = (in_h_pos + in_h_offsets >= 0) & (in_h_pos + in_h_offsets < in_h)
                            mask_w = (in_w_pos + in_w_offsets >= 0) & (in_w_pos + in_w_offsets < in_w)
                            
                            # Load input patches
                            input_vals = tl.load(
                                input_ptr + 
                                pid_batch * input_batch_stride +
                                (in_h_pos + in_h_offsets[:, None]) * input_h_stride +
                                (in_w_pos + in_w_offsets[None, :]) * input_w_stride +
                                in_c_pos * input_c_stride,
                                mask=mask_h[:, None] & mask_w[None, :],
                                other=0.0
                            )
                            
                            # Load filter values
                            filter_vals = tl.load(
                                filter_ptr +
                                filter_offset[:, None, None] * filter_out_stride +
                                (fh + ofh) * filter_h_stride +
                                (fw + ofw) * filter_w_stride +
                                in_c_pos * filter_in_stride,
                                mask=(filter_offset[:, None, None] < out_c),
                                other=0.0
                            )
                            
                            # Compute convolution for this position
                            output += input_vals[None, :, :] * filter_vals[:, None, None]
    
    # Store output
    output_h_offsets = tl.arange(0, BLOCK_X)
    output_w_offsets = tl.arange(0, BLOCK_Y)
    output_c_offsets = tl.arange(0, BLOCK_C)
    
    # Check output bounds
    mask_h = (out_h_offset + output_h_offsets) < out_h
    mask_w = (out_c_offset + output_w_offsets) < out_w
    mask_c = (output_c_offsets) < out_c
    
    # Store the output
    tl.store(
        output_ptr +
        pid_batch * output_batch_stride +
        (out_h_offset + output_h_offsets[:, None, None]) * output_h_stride +
        (out_c_offset + output_w_offsets[None, :, None]) * output_w_stride +
        output_c_offsets[None, None, :] * output_c_stride,
        output,
        mask=mask_h[:, None, None] & mask_w[None, :, None] & mask_c[None, None, :]
    )


def test_conv2d(batch_size, in_h, in_w, in_c, out_c, filter_h, filter_w,
               block_x, block_y, block_z, block_c, filter_tile_size, 
               num_warps, num_stages, use_shared_memory):
    """Test convolution with given parameters"""
    # Check if Triton with Metal backend is available
    if not HAS_TRITON:
        # Simulate performance based on parameters
        # In real implementation, this would run the actual kernel
        flops = 2 * batch_size * out_c * in_c * filter_h * filter_w * ((in_h - filter_h + 1) * (in_w - filter_w + 1))
        theoretical_gflops = 0
        
        # Simulate performance with a model:
        # - Better utilization with appropriate block sizes
        # - Performance increases with warps up to a point
        # - Multiple stages can hide memory latency
        
        # Base performance value
        base_perf = 800.0  # GFLOPS
        
        # Block size efficiency (penalize very small or very large blocks)
        block_x_eff = 1.0 - abs(np.log2(block_x / 16)) * 0.3
        block_y_eff = 1.0 - abs(np.log2(block_y / 16)) * 0.3
        block_z_eff = 1.0 - abs(np.log2(block_z / 4)) * 0.3
        block_c_eff = 1.0 - abs(np.log2(block_c / 32)) * 0.3
        block_eff = block_x_eff * block_y_eff * block_z_eff * block_c_eff
        
        # Filter tile size efficiency (optimal value depends on filter size)
        # For 3x3 filters, 3 is optimal. For 5x5, 5 is better, etc.
        filter_tile_eff = 1.0 - abs(min(filter_tile_size, min(filter_h, filter_w)) / 
                                  min(filter_h, filter_w) - 1.0) * 0.5
        
        # Warps efficiency (more warps is better up to a point)
        warps_eff = min(num_warps / 8.0, 1.2)
        
        # Stages efficiency (more stages is better up to a point)
        stages_eff = min(num_stages / 3.0, 1.5)
        
        # Shared memory usage benefits computations that reuse data
        shared_mem_eff = 1.1 if use_shared_memory else 1.0
        
        # Combine factors
        theoretical_gflops = base_perf * block_eff * filter_tile_eff * warps_eff * stages_eff * shared_mem_eff
        
        # Add some noise to simulate real hardware variability
        theoretical_gflops *= (0.9 + 0.2 * np.random.random())
        
        # Calculate runtime in milliseconds
        runtime_ms = flops / (theoretical_gflops * 1e9) * 1000
        
        return ConfigurationResult(
            config={
                "block_x": block_x,
                "block_y": block_y,
                "block_z": block_z,
                "block_c": block_c,
                "filter_tile_size": filter_tile_size,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "use_shared_memory": use_shared_memory
            },
            runtime_ms=runtime_ms,
            success=True,
            metrics={
                "gflops": theoretical_gflops,
                "flops": flops
            }
        )
    
    # Parameters for convolution
    stride_h = 1
    stride_w = 1
    padding_h = filter_h // 2
    padding_w = filter_w // 2
    
    # Output dimensions
    out_h = in_h  # With padding
    out_w = in_w  # With padding
    
    # Create tensors
    input_tensor = np.random.normal(0, 1, (batch_size, in_h, in_w, in_c)).astype(np.float32)
    filter_tensor = np.random.normal(0, 1, (out_c, filter_h, filter_w, in_c)).astype(np.float32)
    output_tensor = np.zeros((batch_size, out_h, out_w, out_c), dtype=np.float32)
    
    # Create device tensors
    # In a real implementation, these would be moved to the device
    input_dev = input_tensor
    filter_dev = filter_tensor
    output_dev = output_tensor
    
    # Compute strides
    input_batch_stride = in_h * in_w * in_c
    input_h_stride = in_w * in_c
    input_w_stride = in_c
    input_c_stride = 1
    
    filter_out_stride = filter_h * filter_w * in_c
    filter_h_stride = filter_w * in_c
    filter_w_stride = in_c
    filter_in_stride = 1
    
    output_batch_stride = out_h * out_w * out_c
    output_h_stride = out_w * out_c
    output_w_stride = out_c
    output_c_stride = 1
    
    # Define grid
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_h, meta['BLOCK_X']),
        triton.cdiv(out_w, meta['BLOCK_Y']),
    )
    
    try:
        # Time the kernel execution
        start_time = time.time()
        
        conv2d_kernel[grid](
            input_dev, filter_dev, output_dev,
            batch_size, in_h, in_w, in_c,
            out_h, out_w, out_c,
            filter_h, filter_w,
            stride_h, stride_w,
            padding_h, padding_w,
            input_batch_stride, input_h_stride, input_w_stride, input_c_stride,
            filter_out_stride, filter_h_stride, filter_w_stride, filter_in_stride,
            output_batch_stride, output_h_stride, output_w_stride, output_c_stride,
            BLOCK_X=block_x, BLOCK_Y=block_y, BLOCK_Z=block_z, BLOCK_C=block_c,
            FILTER_TILE_SIZE=filter_tile_size,
            num_warps=num_warps,
            num_stages=num_stages
        )
        
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000
        
        # Reference computation on CPU (simplified)
        # In a real implementation, we'd do a proper convolution for validation
        success = True
        
        # Calculate performance in GFLOPS
        flops = 2 * batch_size * out_c * in_c * filter_h * filter_w * out_h * out_w
        gflops = flops / (runtime_ms / 1000) / 1e9
        
        return ConfigurationResult(
            config={
                "block_x": block_x,
                "block_y": block_y,
                "block_z": block_z,
                "block_c": block_c,
                "filter_tile_size": filter_tile_size,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "use_shared_memory": use_shared_memory
            },
            runtime_ms=runtime_ms,
            success=success,
            metrics={
                "gflops": gflops,
                "flops": flops
            }
        )
    
    except Exception as e:
        # If kernel execution fails, return failure
        print(f"Kernel execution failed: {e}")
        return ConfigurationResult(
            config={
                "block_x": block_x,
                "block_y": block_y,
                "block_z": block_z,
                "block_c": block_c,
                "filter_tile_size": filter_tile_size,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "use_shared_memory": use_shared_memory
            },
            runtime_ms=float("inf"),
            success=False,
            metrics={"error": str(e)}
        )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Metal Conv2D Auto-Tuner Example")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--height", type=int, default=64, help="Input height")
    parser.add_argument("--width", type=int, default=64, help="Input width")
    parser.add_argument("--in_channels", type=int, default=64, help="Input channels")
    parser.add_argument("--out_channels", type=int, default=128, help="Output channels")
    parser.add_argument("--filter_size", type=int, default=3, help="Filter size (square)")
    parser.add_argument("--trials", type=int, default=20, help="Number of tuning trials")
    parser.add_argument("--strategy", type=str, default="random", 
                        choices=["random", "grid", "bayesian"], help="Search strategy")
    parser.add_argument("--parallel", action="store_true", help="Use parallel tuning")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    args = parser.parse_args()
    
    print(f"Auto-tuning Conv2D: {args.batch}x{args.height}x{args.width}x{args.in_channels} → "
          f"{args.out_channels} (filter: {args.filter_size}x{args.filter_size})")
    
    # Get tunable parameters for convolution
    params = get_conv_metal_params()
    
    # Create auto-tuner
    tuner = MetalAutoTuner(
        f"conv2d_{args.batch}_{args.height}_{args.width}_{args.in_channels}_{args.out_channels}_{args.filter_size}",
        params,
        n_trials=args.trials,
        search_strategy=args.strategy,
        cache_dir=args.cache_dir
    )
    
    # Define evaluation function
    def evaluate_config(config):
        return test_conv2d(
            args.batch, args.height, args.width, args.in_channels, args.out_channels, 
            args.filter_size, args.filter_size,
            config["block_x"], config["block_y"], config["block_z"], config["block_c"], 
            config["filter_tile_size"], config["num_warps"], config["num_stages"], 
            config["use_shared_memory"]
        )
    
    # Run tuning
    print(f"Running {args.trials} tuning trials with {args.strategy} strategy"
          f"{' in parallel' if args.parallel else ''}...")
    best_config = tuner.tune(
        evaluate_config, 
        parallel=args.parallel, 
        num_workers=args.workers
    )
    
    # Print best configuration
    print(f"\nBest configuration found:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    
    # Get performance metrics for best configuration
    best_result = test_conv2d(
        args.batch, args.height, args.width, args.in_channels, args.out_channels, 
        args.filter_size, args.filter_size,
        best_config["block_x"], best_config["block_y"], best_config["block_z"], best_config["block_c"], 
        best_config["filter_tile_size"], best_config["num_warps"], best_config["num_stages"], 
        best_config["use_shared_memory"]
    )
    
    print(f"\nPerformance of best configuration:")
    print(f"  Runtime: {best_result.runtime_ms:.3f} ms")
    print(f"  GFLOPS: {best_result.metrics.get('gflops', 0):.2f}")


if __name__ == "__main__":
    main() 