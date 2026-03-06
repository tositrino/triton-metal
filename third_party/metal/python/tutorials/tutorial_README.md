# Triton Metal Backend Compatibility Tutorial

This tutorial provides a comprehensive guide to check system compatibility with Triton's Metal backend and optimize your code for Apple Silicon GPUs, especially M3 chips.

## Overview

The Triton Metal backend enables running Triton kernels on Apple Silicon GPUs using Metal API. This tutorial helps you:

1. Check if your system meets all requirements
2. Understand memory layout optimizations for Apple Silicon
3. Leverage M3-specific optimizations if available
4. Troubleshoot common issues
5. Run a simple example to verify functionality

## Files

- `tutorial_metal_compatibility.py` - Main tutorial script
- `test_tutorial_metal_compatibility.py` - Unit tests for the tutorial
- `tutorial_README.md` - This file

## Prerequisites

- macOS 13.5 or newer
- Apple Silicon Mac (M1, M2, or M3 series)
- Python 3.8 or newer
- MLX package installed

## Running the Tutorial

Simply execute the tutorial script to check your system:

```bash
python tutorial_metal_compatibility.py
```

The script will:
1. Check your macOS version
2. Verify you're running on Apple Silicon
3. Detect your M-series chip generation
4. Check for M3-specific optimizations if applicable
5. Verify MLX installation
6. Check memory layout optimizations
7. Provide a summary and troubleshooting tips

## Testing

To run the unit tests for the tutorial:

```bash
python test_tutorial_metal_compatibility.py
```

The test suite mocks hardware detection and ensures all components function properly.

## Key Features Explained

### System Compatibility Check

The tutorial checks for:
- macOS 13.5+ (required for proper Metal API support)
- Apple Silicon hardware
- M-series generation (M1, M2, M3, etc.)
- MLX package installation

### Memory Layout Optimizations

The Metal backend includes the COALESCED memory layout optimization for reduction operations, which is especially effective on M3 chips. The tutorial checks for:
- Presence of COALESCED layout definition
- Consistency between modules

### M3-Specific Optimizations

When running on M3 chips, the tutorial detects:
- Dynamic Caching
- Enhanced Matrix Coprocessor
- Shared Memory Atomics
- Enhanced SIMD capabilities
- Advanced Warp Scheduling
- Memory Compression

## Example Code

The tutorial includes a simple vector addition example to demonstrate using the Metal backend:

```python
import os
os.environ["TRITON_BACKEND"] = "metal"

import triton
import triton.language as tl
import mlx.core as mx

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Grid-stride loop
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform operation
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

# Launch kernel with appropriate grid
grid = (triton.cdiv(n_elements, 1024),)
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```

## Troubleshooting

If you encounter issues:
- Ensure macOS 13.5+ and Apple Silicon
- Install MLX package: `pip install mlx`
- Set the environment variable: `os.environ["TRITON_BACKEND"] = "metal"`
- For debugging, enable logs: `os.environ["triton_DEBUG"] = "1"`
- Check memory allocation errors by reducing batch sizes
- Verify grid and block dimensions for kernel launches

## For More Information

For detailed information about the Metal backend and M3 optimizations, see:
- The main [Triton Metal Backend README](./README.md)
- [M3 Optimizations Documentation](../../metal/python/docs/M3_OPTIMIZATIONS.md)
- [Performance Optimization Guide](../../metal/python/docs/PERFORMANCE_OPTIMIZATION.md)

## Contributing

If you'd like to contribute to the tutorial or add more examples, please follow the main Triton contribution guidelines. 