"""
Transformer Attention Example Using Triton Metal Backend

This example demonstrates how to implement transformer attention using Triton
with the Metal backend on Apple Silicon GPUs.
"""

import os
import sys
import time
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    print("MLX not found. Please install it with 'pip install mlx'")
    MLX_AVAILABLE = False
    sys.exit(1)

# Try to import Triton
try:
    import triton
    import triton.language as tl
except ImportError:
    print("Triton not found. Please install it with 'pip install triton'")
    sys.exit(1)

# Import our modules
from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
from m3_optimizations import m3_optimizer

# Set backend to metal
os.environ["TRITON_BACKEND"] = "metal"

# Define attention kernel
@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    batch, seqlen, n_heads, d_head,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Compute attention with scaled dot product.
    
    Args:
        q_ptr: Pointer to query tensor (batch, n_heads, seqlen, d_head)
        k_ptr: Pointer to key tensor (batch, n_heads, seqlen, d_head)
        v_ptr: Pointer to value tensor (batch, n_heads, seqlen, d_head)
        output_ptr: Pointer to output tensor (batch, n_heads, seqlen, d_head)
        batch: Batch size
        seqlen: Sequence length
        n_heads: Number of attention heads
        d_head: Dimension of each attention head
        stride_*: Strides for each tensor dimension
        scale: Scaling factor (1/sqrt(d_head))
        BLOCK_*: Block sizes for tiling
    """
    # Get program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    # Compute pointers
    q_batch_offset = pid_batch * stride_qb + pid_head * stride_qh
    k_batch_offset = pid_batch * stride_kb + pid_head * stride_kh
    v_batch_offset = pid_batch * stride_vb + pid_head * stride_vh
    o_batch_offset = pid_batch * stride_ob + pid_head * stride_oh
    
    # Loop over sequence length blocks
    for start_m in range(0, seqlen, BLOCK_M):
        # Compute attention for this block
        
        # Load query block
        q_ptrs = q_batch_offset + start_m * stride_qm + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, d_head)[None, :] * stride_qk
        q_mask = (start_m + tl.arange(0, BLOCK_M)[:, None]) < seqlen
        q = tl.load(q_ptr + q_ptrs, mask=q_mask, other=0.0)
        
        # Initialize accumulator for softmax normalization
        softmax_normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
        
        # Initialize output
        o = tl.zeros((BLOCK_M, d_head), dtype=tl.float32)
        
        # Loop over key/value blocks
        for start_n in range(0, seqlen, BLOCK_N):
            # Load key block
            k_ptrs = k_batch_offset + start_n * stride_kn + tl.arange(0, BLOCK_N)[None, :] * stride_kn + tl.arange(0, d_head)[:, None] * stride_kk
            k_mask = (start_n + tl.arange(0, BLOCK_N)[None, :]) < seqlen
            k = tl.load(k_ptr + k_ptrs, mask=k_mask, other=0.0)
            
            # Compute attention scores
            scores = tl.dot(q, k) * scale
            
            # Apply causal mask (optional)
            causal_mask = (start_m + tl.arange(0, BLOCK_M)[:, None]) >= (start_n + tl.arange(0, BLOCK_N)[None, :])
            scores = tl.where(causal_mask, scores, float("-inf"))
            
            # Compute softmax
            scores_max = tl.max(scores, axis=1, keepdims=True)
            scores_exp = tl.exp(scores - scores_max)
            
            # Load value block
            v_ptrs = v_batch_offset + start_n * stride_vn + tl.arange(0, BLOCK_N)[None, :] * stride_vn + tl.arange(0, d_head)[:, None] * stride_vk
            v = tl.load(v_ptr + v_ptrs, mask=k_mask, other=0.0)
            
            # Update output with attention
            o += tl.dot(scores_exp, v)
            softmax_normalizer += tl.sum(scores_exp, axis=1)
        
        # Normalize output
        o = o / softmax_normalizer[:, None]
        
        # Store output
        output_ptrs = o_batch_offset + start_m * stride_om + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, d_head)[None, :] * stride_ok
        output_mask = (start_m + tl.arange(0, BLOCK_M)[:, None]) < seqlen
        tl.store(output_ptr + output_ptrs, o, mask=output_mask)

def triton_attention(q, k, v, causal=True):
    """
    Compute attention using Triton kernel
    
    Args:
        q: Query tensor (batch, n_heads, seqlen, d_head)
        k: Key tensor (batch, n_heads, seqlen, d_head)
        v: Value tensor (batch, n_heads, seqlen, d_head)
        causal: Whether to use causal attention
        
    Returns:
        Attention output
    """
    # Extract dimensions
    batch, n_heads, seqlen, d_head = q.shape
    
    # Compute scale
    scale = 1.0 / (d_head ** 0.5)
    
    # Allocate output
    output = mx.empty_like(q)
    
    # Define block sizes - these can be tuned
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = d_head
    
    # Launch kernel
    grid = (batch, n_heads, 1)
    
    # If M3, optimize kernel launch parameters
    if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
        grid, (BLOCK_M, BLOCK_N, BLOCK_K) = m3_optimizer.optimize_kernel_launch(
            grid, (BLOCK_M, BLOCK_N, BLOCK_K)
        )
    
    # Compute strides
    stride_qb, stride_qh, stride_qm, stride_qk = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
    stride_kb, stride_kh, stride_kn, stride_kk = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
    stride_vb, stride_vh, stride_vn, stride_vk = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
    stride_ob, stride_oh, stride_om, stride_ok = output.stride(0), output.stride(1), output.stride(2), output.stride(3)
    
    # Launch kernel
    attention_kernel[grid](
        q, k, v, output,
        batch, seqlen, n_heads, d_head,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return output

def mlx_attention(q, k, v, causal=True):
    """
    Compute attention using MLX directly
    
    Args:
        q: Query tensor (batch, n_heads, seqlen, d_head)
        k: Key tensor (batch, n_heads, seqlen, d_head)
        v: Value tensor (batch, n_heads, seqlen, d_head)
        causal: Whether to use causal attention
        
    Returns:
        Attention output
    """
    # Extract dimensions
    batch, n_heads, seqlen, d_head = q.shape
    
    # Compute scale
    scale = 1.0 / (d_head ** 0.5)
    
    # For each batch and head
    outputs = []
    for b in range(batch):
        head_outputs = []
        for h in range(n_heads):
            # Extract query, key, value for this batch and head
            q_bh = q[b, h]
            k_bh = k[b, h]
            v_bh = v[b, h]
            
            # If on M3, use M3-optimized attention
            if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
                # Compute scores
                scores = m3_optimizer.optimize_matmul(q_bh, mx.transpose(k_bh)) * scale
                
                # Apply causal mask if needed
                if causal:
                    # Create causal mask (lower triangular)
                    mask = np.tril(np.ones((seqlen, seqlen)))
                    mask = mx.array(mask)
                    scores = mx.where(mask == 0, float("-inf"), scores)
                
                # Apply softmax
                attn = mx.softmax(scores, axis=-1)
                
                # Compute attention output
                output_h = m3_optimizer.optimize_matmul(attn, v_bh)
            else:
                # Compute scores
                scores = mx.matmul(q_bh, mx.transpose(k_bh)) * scale
                
                # Apply causal mask if needed
                if causal:
                    # Create causal mask (lower triangular)
                    mask = np.tril(np.ones((seqlen, seqlen)))
                    mask = mx.array(mask)
                    scores = mx.where(mask == 0, float("-inf"), scores)
                
                # Apply softmax
                attn = mx.softmax(scores, axis=-1)
                
                # Compute attention output
                output_h = mx.matmul(attn, v_bh)
            
            head_outputs.append(output_h)
        outputs.append(mx.stack(head_outputs))
    
    return mx.stack(outputs)

def test_attention():
    """Test transformer attention implementation"""
    # Define dimensions
    batch = 2
    n_heads = 4
    seqlen = 256
    d_head = 64
    
    # Create random input tensors
    q = mx.random.normal((batch, n_heads, seqlen, d_head))
    k = mx.random.normal((batch, n_heads, seqlen, d_head))
    v = mx.random.normal((batch, n_heads, seqlen, d_head))
    
    # Run MLX implementation
    start_time = time.time()
    mlx_output = mlx_attention(q, k, v, causal=True)
    mlx_time = time.time() - start_time
    print(f"MLX Attention time: {mlx_time:.6f}s")
    
    # Run Triton implementation
    start_time = time.time()
    triton_output = triton_attention(q, k, v, causal=True)
    triton_time = time.time() - start_time
    print(f"Triton Attention time: {triton_time:.6f}s")
    
    # Compare results
    max_diff = mx.max(mx.abs(mlx_output - triton_output))
    print(f"Maximum difference: {max_diff}")
    if max_diff < 1e-5:
        print("✅ Implementations match!")
    else:
        print("❌ Implementations do not match!")
    
    # Print speedup
    if mlx_time > 0:
        speedup = mlx_time / triton_time
        print(f"Triton/Metal speedup: {speedup:.2f}x")

def benchmark_attention():
    """Benchmark transformer attention implementation"""
    # Define dimensions to benchmark
    batch_sizes = [1, 2, 4]
    n_heads_list = [4, 8, 16]
    seqlen_list = [128, 256, 512, 1024]
    d_head_list = [32, 64, 128]
    
    print("\nBenchmarking Attention:")
    print(f"{'Batch':>6} {'Heads':>6} {'SeqLen':>8} {'Dim':>6} {'MLX (ms)':>10} {'Triton (ms)':>12} {'Speedup':>8}")
    print("-" * 65)
    
    for batch in batch_sizes:
        for n_heads in n_heads_list:
            for seqlen in seqlen_list:
                for d_head in d_head_list:
                    # Skip very large configurations
                    if batch * n_heads * seqlen * d_head > 32 * 1024 * 1024:
                        continue
                    
                    # Create random input tensors
                    q = mx.random.normal((batch, n_heads, seqlen, d_head))
                    k = mx.random.normal((batch, n_heads, seqlen, d_head))
                    v = mx.random.normal((batch, n_heads, seqlen, d_head))
                    
                    # Warm up
                    _ = mlx_attention(q, k, v, causal=True)
                    _ = triton_attention(q, k, v, causal=True)
                    
                    # Run MLX implementation
                    start_time = time.time()
                    mlx_output = mlx_attention(q, k, v, causal=True)
                    mlx_time = time.time() - start_time
                    
                    # Run Triton implementation
                    start_time = time.time()
                    triton_output = triton_attention(q, k, v, causal=True)
                    triton_time = time.time() - start_time
                    
                    # Convert to milliseconds
                    mlx_ms = mlx_time * 1000
                    triton_ms = triton_time * 1000
                    
                    # Calculate speedup
                    speedup = mlx_time / triton_time if triton_time > 0 else 0
                    
                    # Print results
                    print(f"{batch:6d} {n_heads:6d} {seqlen:8d} {d_head:6d} {mlx_ms:10.2f} {triton_ms:12.2f} {speedup:8.2f}x")

if __name__ == "__main__":
    print(f"Apple {hardware_capabilities.chip_generation.name} detected")
    print(f"MLX version: {mx.__version__}")
    print(f"Triton version: {triton.__version__}")
    
    # Test attention implementation
    test_attention()
    
    # Benchmark attention
    benchmark_attention() 