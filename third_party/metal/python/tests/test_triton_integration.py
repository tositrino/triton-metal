"""
End-to-end test for Triton Python API integration with Metal backend.

This test verifies that the Triton Python API correctly uses the Metal backend
for compilation and execution, with a particular focus on reduction operations
that should use the COALESCED memory layout.
"""

import os
import sys
import unittest
import numpy as np
from typing import Dict, List, Any, Tuple



# Try to import triton
try:
    import triton
    import triton.language as tl
    from triton.runtime.autotuner import Autotuner, autotune
    from triton.runtime.jit import JITFunction
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton not available. Tests will be skipped.")

# Try to import MLX for result verification
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available. Some verification steps will be skipped.")

# Import metal components for internal checks
from MLX.metal_memory_manager import get_metal_memory_manager, MemoryLayout
from mlx.memory_layout_optimizer import ReductionLayoutPattern, optimize_memory_layout

@unittest.skipIf(not HAS_TRITON, "Triton not available")
class TestTritonMetalIntegration(unittest.TestCase):
    """Test end-to-end Triton to Metal integration"""
    
    def setUp(self):
        """Set up test case"""
        # Check if Metal backend is available
        self.has_metal_backend = 'metal' in triton.runtime.backends
        if not self.has_metal_backend:
            self.skipTest("Metal backend not available")
        
        # Create memory manager for internal checks
        self.memory_manager = get_metal_memory_manager()
        
        # Create reduction pattern for verification
        self.reduction_pattern = ReductionLayoutPattern()
        
        # Set up size parameters for tests
        self.M = 1024
        self.N = 1024
    
    def test_backend_initialization(self):
        """Test Metal backend initialization"""
        # Check backend registration
        self.assertIn('metal', triton.runtime.backends)
        print(f"Available backends: {list(triton.runtime.backends.keys())}")
    
    def test_simple_reduction_kernel(self):
        """Test a simple reduction kernel on Metal backend"""
        
        @triton.jit
        def reduction_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            # Create block ID and offset
            pid = tl.program_id(0)
            block_offset = pid * BLOCK_SIZE
            
            # Create offsets for this block
            offsets = block_offset + tl.arange(0, BLOCK_SIZE)
            
            # Bounds check
            mask = offsets < n_elements
            
            # Load data
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            
            # Perform reduction
            reduced = tl.sum(x, axis=0)
            
            # Store result
            tl.store(out_ptr + pid, reduced)
        
        # Create input data
        x = np.random.randn(self.M).astype(np.float32)
        output = np.zeros((1,), dtype=np.float32)
        
        # Get device pointers
        x_device = triton.testing.to_device(x)
        output_device = triton.testing.to_device(output)
        
        # Determine grid and block size
        BLOCK_SIZE = 128
        grid = (self.M + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel with explicit backend
        reduction_kernel[grid](
            x_device, output_device, self.M, BLOCK_SIZE=BLOCK_SIZE,
            backend='metal'
        )
        
        # Get the result back
        result = triton.testing.to_numpy(output_device)
        
        # Verify the result
        expected = np.sum(x)
        np.testing.assert_allclose(result, expected, rtol=1e-3)
        print(f"Reduction result: {result[0]}, expected: {expected}")
    
    def test_2d_reduction_kernel(self):
        """Test a 2D reduction kernel on Metal backend"""
        
        @triton.jit
        def reduction_kernel_2d(x_ptr, out_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
            # Create program IDs
            pid_m = tl.program_id(0)
            
            # Create offsets
            m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            n_offsets = tl.arange(0, BLOCK_SIZE_N)
            
            # Create a mask for bounds checking
            mask_m = m_offsets < M
            
            # Initialize the accumulator
            acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            
            # Loop over columns
            for n in range(0, N, BLOCK_SIZE_N):
                # Create masks for bounds checking
                mask_n = (n + n_offsets) < N
                
                # Combined mask
                mask = mask_m[:, None] & mask_n[None, :]
                
                # Compute the pointer to the start of the block
                block_ptr = x_ptr + m_offsets[:, None] * N + (n + n_offsets)[None, :]
                
                # Load the block
                x = tl.load(block_ptr, mask=mask, other=0.0)
                
                # Reduce along the N dimension
                acc += tl.sum(x, axis=1)
            
            # Store the result
            tl.store(out_ptr + m_offsets, acc, mask=mask_m)
        
        # Create input data
        x = np.random.randn(self.M, self.N).astype(np.float32)
        output = np.zeros((self.M,), dtype=np.float32)
        
        # Get device pointers
        x_device = triton.testing.to_device(x)
        output_device = triton.testing.to_device(output)
        
        # Determine grid and block size
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        grid = (self.M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        
        # Launch kernel with explicit backend
        reduction_kernel_2d[grid](
            x_device, output_device, self.M, self.N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
            backend='metal'
        )
        
        # Get the result back
        result = triton.testing.to_numpy(output_device)
        
        # Verify the result
        expected = np.sum(x, axis=1)
        np.testing.assert_allclose(result, expected, rtol=1e-3)
        print(f"2D Reduction result shape: {result.shape}, matches expected: {np.allclose(result, expected, rtol=1e-3)}")
    
    @unittest.skipIf(not HAS_MLX, "MLX not available for verification")
    def test_reduction_with_autotuning(self):
        """Test reduction with autotuning using Metal backend"""
        
        @triton.jit
        def reduction_kernel_autotuned(
            x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr
        ):
            # Create block ID and offset
            pid = tl.program_id(0)
            block_offset = pid * BLOCK_SIZE
            
            # Create offsets for this block
            offsets = block_offset + tl.arange(0, BLOCK_SIZE)
            
            # Bounds check
            mask = offsets < N
            
            # Load data
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            
            # Perform reduction
            reduced = tl.sum(x, axis=0)
            
            # Store result
            tl.store(out_ptr + pid, reduced)
        
        # Create autotuner
        @autotune(
            configs=[
                triton.Config({'BLOCK_SIZE': 64}),
                triton.Config({'BLOCK_SIZE': 128}),
                triton.Config({'BLOCK_SIZE': 256}),
                triton.Config({'BLOCK_SIZE': 512}),
                triton.Config({'BLOCK_SIZE': 1024}),
            ],
            key=['N'],
        )
        def reduction_launcher(x, N):
            # Allocate output
            grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
            output = mx.zeros((grid(reduction_kernel_autotuned.configs[0])[0],), dtype=x.dtype)
            
            # Launch kernel
            reduction_kernel_autotuned[grid](
                x, output, N,
                backend='metal'
            )
            
            # Return output
            return output
        
        # Create input data
        if HAS_MLX:
            x = mx.random.normal((self.N,), dtype=mx.float32)
            
            # Run the kernel
            result = reduction_launcher(x, self.N)
            
            # Verify with MLX
            expected = mx.sum(x)
            sum_of_partial_sums = mx.sum(result)
            
            # Check if close
            self.assertTrue(
                mx.abs(expected - sum_of_partial_sums) < 1e-3,
                f"Expected {expected}, got {sum_of_partial_sums}"
            )
            print(f"Autotuned reduction result: {sum_of_partial_sums}, expected: {expected}")
    
    def test_memory_layout_internal(self):
        """Test internal verification of memory layout for reduction kernels"""
        # Define a mock reduction operation similar to what would be generated
        reduction_op = {
            "type": "tt.reduce",
            "id": "test_reduction",
            "input_shapes": [[self.M, self.N]],
            "args": {"axis": 1},
            "output_shape": [self.M, 1]
        }
        
        # Optimize using memory manager
        optimized_op = self.memory_manager._optimize_reduction_memory(reduction_op.copy())
        
        # Check that COALESCED layout is applied
        self.assertIn("execution_parameters", optimized_op)
        self.assertIn("memory_layout", optimized_op["execution_parameters"])
        self.assertEqual(
            optimized_op["execution_parameters"]["memory_layout"],
            MemoryLayout.COALESCED.value
        )
        
        # Check that reduction pattern identifies this operation
        self.assertTrue(
            self.reduction_pattern.is_applicable(reduction_op, None),
            "ReductionLayoutPattern should identify this operation"
        )
        
        # Check that COALESCED is the optimal layout
        optimal_layout = self.reduction_pattern.get_optimal_layout(
            reduction_op["input_shapes"][0], None
        )
        self.assertEqual(
            optimal_layout,
            MemoryLayout.COALESCED,
            f"COALESCED should be optimal layout, got {optimal_layout}"
        )
        
        print("Internal verification confirms COALESCED layout for reduction operations")

if __name__ == "__main__":
    unittest.main() 