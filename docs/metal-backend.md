# Metal Backend for Triton

This document provides an overview of the Metal backend for Triton, which enables running Triton kernels on Apple Silicon GPUs.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [M3-Specific Optimizations](#m3-specific-optimizations)
- [Examples](#examples)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## Requirements

- macOS 13.5 or higher
- Apple Silicon Mac (M1, M2, or M3)
- Python 3.9 or higher
- MLX 0.3.0 or higher

## Installation

The Metal backend is automatically included when installing this version of Triton. Make sure you have the necessary dependencies:

```bash
# Install the base package with Metal support
pip install -e ".[metal]"

# Or if you want to explicitly enable Metal
TRITON_BUILD_WITH_METAL=ON pip install -e .
```

## Usage

To use the Metal backend in your code:

```python
import os
# Set this before importing Triton
os.environ["TRITON_BACKEND"] = "metal"

import triton
import triton.language as tl

@triton.jit
def my_kernel(
    # Kernel definition here
    ...
):
    # Kernel code here
    ...

# Use the kernel as usual
my_kernel[grid](...)
```

## M3-Specific Optimizations

The Metal backend includes automatic optimizations for M3 chips:

- **64KB Shared Memory**: Utilizes M3's larger shared memory (vs 32KB on M1/M2)
- **8-wide Vectorization**: Takes advantage of M3's wider SIMD paths
- **Tensor Cores**: Optimized matrix operations using tensor cores
- **Enhanced SIMD Operations**: Uses 32-wide SIMD groups vs 16-wide on M1/M2
- **Dynamic Register Caching**: Leverages improved caching capabilities

These optimizations are automatically applied when running on M3 hardware.

## Examples

See the `third_party/metal/python/examples/` directory for complete examples:

- `matmul_example.py` - Matrix multiplication with M3 optimizations
- `convolution_example.py` - Optimized convolution operations
- `backend_comparison_example.py` - Performance comparison with other backends

## Performance Tips

1. **Tile Sizes**: For matrix multiplication, use larger tile sizes on M3 (128x128 or 256x256)
2. **Vectorization**: Explicitly vectorize memory access patterns for better performance
3. **Shared Memory**: Utilize shared memory aggressively on M3 chips
4. **Thread Count**: Optimal thread count is typically 8 for M1/M2 and 16 for M3
5. **Memory Layout**: Prefer blocked layouts for M3's memory hierarchy

## Troubleshooting

### Common Issues

- **MLX Not Found**: Make sure MLX is installed (`pip install mlx>=0.3.0`)
- **Metal Not Available**: Verify you're running on Apple Silicon with macOS 13.5+
- **Performance Issues**: Check that M3 optimizations are being applied properly

### Reporting Problems

If you encounter issues, please report them at:
https://github.com/chenxingqiang/triton-metal/issues

Include your system information, macOS version, and a minimal reproducing example. 