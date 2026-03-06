#!/usr/bin/env python
"""
Test script for Metal backend integration with Triton Python API.

This script verifies that the Metal backend is correctly registered with
the Triton Python API and can be used to run simple kernels.
"""

import os
import sys
import argparse
import numpy as np

# Try to import triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Error: Triton not available. Please install Triton first.")
    sys.exit(1)

def test_backend_registration():
    """Test that the Metal backend is registered with Triton"""
    print("Testing Metal backend registration...")
    
    # Check if Metal backend is in the list of available backends
    if 'metal' in triton.runtime.backends:
        print("✅ Metal backend is registered!")
        for name, backend in triton.runtime.backends.items():
            print(f"  - {name}: {backend}")
    else:
        print("❌ Metal backend is NOT registered!")
        print("Available backends:")
        for name in triton.runtime.backends:
            print(f"  - {name}")

def test_simple_kernel():
    """Test a simple kernel using the Metal backend"""
    print("\nTesting a simple kernel with Metal backend...")
    
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # Create block ID
        pid = tl.program_id(0)
        block_offset = pid * BLOCK_SIZE
        
        # Create offsets for this block
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        
        # Create a mask to handle boundary conditions
        mask = offsets < n_elements
        
        # Load data
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        
        # Perform operation
        output = x + y
        
        # Store result
        tl.store(output_ptr + offsets, output, mask=mask)
    
    # Test parameters
    n_elements = 1024
    BLOCK_SIZE = 128
    
    # Generate input data
    x = np.random.rand(n_elements).astype(np.float32)
    y = np.random.rand(n_elements).astype(np.float32)
    output = np.zeros_like(x)
    
    # Move data to device
    x_device = triton.testing.to_device(x)
    y_device = triton.testing.to_device(y)
    output_device = triton.testing.to_device(output)
    
    # Calculate grid dimensions
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    print(f"Running kernel with grid={grid}, block_size={BLOCK_SIZE}...")
    
    try:
        # Run kernel with Metal backend
        add_kernel[grid](
            x_device, y_device, output_device, n_elements, BLOCK_SIZE=BLOCK_SIZE,
            backend='metal'
        )
        
        # Get result back to host
        result = triton.testing.to_numpy(output_device)
        
        # Verify result
        expected = x + y
        correct = np.allclose(result, expected)
        
        if correct:
            print("✅ Kernel execution successful!")
            print(f"  - Sample result[0:5]: {result[0:5]}")
            print(f"  - Expected[0:5]: {expected[0:5]}")
        else:
            print("❌ Kernel execution produced incorrect results!")
            print(f"  - Result[0:5]: {result[0:5]}")
            print(f"  - Expected[0:5]: {expected[0:5]}")
            
    except Exception as e:
        print(f"❌ Kernel execution failed: {e}")
        import traceback
        traceback.print_exc()

def test_reduction_kernel():
    """Test a reduction kernel using the Metal backend"""
    print("\nTesting a reduction kernel with Metal backend...")
    
    @triton.jit
    def reduce_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # Create block ID
        pid = tl.program_id(0)
        block_offset = pid * BLOCK_SIZE
        
        # Create offsets for this block
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        
        # Create a mask to handle boundary conditions
        mask = offsets < n_elements
        
        # Load data
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Perform reduction
        sum_x = tl.sum(x, axis=0)
        
        # Store result (only for the first thread block)
        tl.atomic_add(output_ptr, sum_x)
    
    # Test parameters
    n_elements = 1024
    BLOCK_SIZE = 128
    
    # Generate input data
    x = np.random.rand(n_elements).astype(np.float32)
    output = np.zeros(1, dtype=np.float32)
    
    # Move data to device
    x_device = triton.testing.to_device(x)
    output_device = triton.testing.to_device(output)
    
    # Calculate grid dimensions
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    print(f"Running reduction kernel with grid={grid}, block_size={BLOCK_SIZE}...")
    
    try:
        # Run kernel with Metal backend
        reduce_kernel[grid](
            x_device, output_device, n_elements, BLOCK_SIZE=BLOCK_SIZE,
            backend='metal'
        )
        
        # Get result back to host
        result = triton.testing.to_numpy(output_device)
        
        # Verify result
        expected = np.sum(x)
        correct = np.allclose(result, expected, rtol=1e-3)
        
        if correct:
            print("✅ Reduction kernel execution successful!")
            print(f"  - Result: {result[0]}")
            print(f"  - Expected: {expected}")
        else:
            print("❌ Reduction kernel execution produced incorrect results!")
            print(f"  - Result: {result[0]}")
            print(f"  - Expected: {expected}")
            
    except Exception as e:
        print(f"❌ Reduction kernel execution failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test Metal backend integration with Triton")
    parser.add_argument("--test", choices=["registration", "simple", "reduction", "all"], 
                       default="all", help="Test to run")
    
    args = parser.parse_args()
    
    # Run the selected test
    if args.test == "registration" or args.test == "all":
        test_backend_registration()
    
    if args.test == "simple" or args.test == "all":
        test_simple_kernel()
    
    if args.test == "reduction" or args.test == "all":
        test_reduction_kernel()

if __name__ == "__main__":
    main() 