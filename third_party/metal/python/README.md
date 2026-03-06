# Triton Metal Backend for Apple Silicon

This package provides a Metal backend for Triton to run on Apple Silicon GPUs (M1, M2, and M3 chips) using Apple's MLX framework.

## Installation

Install the required dependencies:

```bash
pip install mlx triton
```

## Usage

To use the Metal backend, set the `TRITON_BACKEND` environment variable before importing Triton:

```python
import os
os.environ["TRITON_BACKEND"] = "metal"

import triton
import triton.language as tl
```

## Features

The Triton Metal backend supports:

1. **Standard Triton Operations**: Most standard Triton operations are supported and mapped to MLX equivalents
2. **Hardware-Specific Optimizations**: Specialized optimizations for M1, M2, and M3 chips
3. **Advanced Memory Patterns**: Support for complex memory access patterns
4. **Metal Performance Shaders**: Integration with Apple's high-performance GPU kernels
5. **Operation Fusion**: Automatic fusion of common operation patterns
6. **M3-Specific Features**: Dynamic caching and enhanced matrix operations on M3

## Examples

### Basic Example: Vector Addition

```python
import os
os.environ["TRITON_BACKEND"] = "metal"

import triton
import triton.language as tl
import numpy as np
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

# Example usage
def main():
    n_elements = 1024 * 1024
    
    # Create input arrays
    x = mx.random.normal((n_elements,))
    y = mx.random.normal((n_elements,))
    output = mx.zeros((n_elements,))
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, 1024),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    # Verify result
    mx_output = x + y
    diff = mx.abs(output - mx_output).max()
    print(f"Max difference: {diff}")
    
if __name__ == "__main__":
    main()
```

### Advanced Example: Matrix Multiplication

```python
import os
os.environ["TRITON_BACKEND"] = "metal"

import triton
import triton.language as tl
import mlx.core as mx
import numpy as np

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block start indices
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Create offset arrays
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K) * BLOCK_K, BLOCK_K):
        # Create mask for bounds checking
        k_offsets = k + tl.arange(0, BLOCK_K)
        
        # Check bounds
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
        
        # Compute pointers for A and B
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        # Load data
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute matrix multiply
        acc += tl.dot(a, b)
    
    # Store output
    offs_cm = offs_m[:, None] * stride_cm
    offs_cn = offs_n[None, :] * stride_cn
    c_ptrs = c_ptr + offs_cm + offs_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    tl.store(c_ptrs, acc, mask=c_mask)

# Example usage
def main():
    # Matrix dimensions
    M, N, K = 1024, 1024, 1024
    
    # Create matrices
    a = mx.random.normal((M, K))
    b = mx.random.normal((K, N))
    c = mx.zeros((M, N))
    
    # Block sizes
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    # Launch grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Get strides
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    # Verify result
    mx_c = mx.matmul(a, b)
    diff = mx.abs(c - mx_c).max()
    print(f"Max difference: {diff}")
    
if __name__ == "__main__":
    main()
```

## Advanced Features

### M3-Specific Optimizations

The Metal backend includes optimizations specifically for Apple M3 chips:

- **Dynamic Caching**: Takes advantage of M3's dynamic caching features for improved memory access patterns
- **Enhanced Matrix Operations**: Utilizes M3's improved matrix coprocessor
- **Shared Memory Atomics**: Uses hardware-accelerated atomic operations
- **Enhanced SIMD**: Leverages improved SIMD capabilities

Example using M3 optimizations:

```python
import os
os.environ["TRITON_BACKEND"] = "metal"

import triton
import mlx.core as mx
from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
from m3_optimizations import m3_optimizer

# Check if running on M3
if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
    # Use M3-optimized matrix multiply
    def optimized_matmul(a, b):
        return m3_optimizer.optimize_matmul(a, b)
else:
    # Fallback for other chips
    def optimized_matmul(a, b):
        return mx.matmul(a, b)

# Use the optimized function in your code
a = mx.random.normal((1024, 1024))
b = mx.random.normal((1024, 1024))
c = optimized_matmul(a, b)
```

### Complex Memory Access Patterns

The Metal backend supports various memory access patterns:

- **Contiguous**: Standard sequential access
- **Strided**: Regular strides with optimized implementation
- **Block**: 2D blocked access patterns for matrices
- **Scatter/Gather**: Complex indexed access patterns
- **Broadcast**: Efficient broadcasting operations

Example using advanced memory patterns:

```python
import os
os.environ["TRITON_BACKEND"] = "metal"

import triton
import triton.language as tl
import mlx.core as mx
from advanced_memory_patterns import memory_pattern_optimizer

# Example with strided access
@triton.jit
def strided_kernel(x_ptr, output_ptr, stride, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create strided indices
    indices = block_start + tl.arange(0, BLOCK_SIZE) * stride
    mask = indices < n_elements
    
    # Load with strided pattern
    x = tl.load(x_ptr + indices, mask=mask)
    
    # Store result
    tl.store(output_ptr + indices, x * 2.0, mask=mask)
```

### Integration with Metal Performance Shaders

For high-performance operations, the backend integrates with Metal Performance Shaders:

```python
import os
os.environ["TRITON_BACKEND"] = "metal"

import triton
import mlx.core as mx
from metal_performance_shaders import mps_integration, MPSOperation

# Example using MPS for convolution
def mps_optimized_conv2d(input_tensor, filters, stride=(1, 1), padding="same"):
    # Check if operation is available
    if mps_integration.is_operation_available(MPSOperation.CONV2D):
        # Use MPS accelerated convolution
        return mps_integration.run_operation(
            MPSOperation.CONV2D,
            input_tensor, 
            filters,
            stride=stride,
            padding=padding
        )
    else:
        # Fallback implementation
        return mx.conv(input_tensor, filters, stride=stride, padding=padding)
```

### Operation Fusion

The Metal backend automatically detects and fuses common operation patterns for better performance:

1. **FMA Fusion**: Fuses multiply-add sequences
2. **GELU Activation**: Fuses the sequence of operations in GELU 
3. **Layer Normalization**: Fuses operations in layer norm
4. **Attention Mechanism**: Fuses operations in attention computation
5. **Reduction Operations**: Fuses scan and reduction patterns

## Performance Tips

1. **Use Appropriate Block Sizes**: Tune block sizes for your specific workload and hardware generation
2. **Leverage Hardware Features**: Use M3-specific optimizations when available
3. **Minimize Memory Transfers**: Keep data on the GPU as much as possible
4. **Use Fused Operations**: Prefer fused operations where available
5. **Profile Your Kernels**: Benchmark and optimize your critical kernels

## Troubleshooting

### Common Issues

1. **Memory Allocation Errors**: Reduce batch sizes or tensor dimensions
2. **Kernel Launch Failures**: Check your grid and block dimensions
3. **Precision Issues**: Ensure appropriate data types and handling of numerical edge cases
4. **MLX Import Errors**: Make sure you have installed MLX correctly for your platform

### Debugging

Set the `triton_DEBUG` environment variable for more detailed logging:

```python
import os
os.environ["triton_DEBUG"] = "1"
os.environ["TRITON_BACKEND"] = "metal"
import triton
```

## Contributing

Contributions to the Metal backend are welcome! Areas where help is particularly appreciated:

1. Adding support for more Triton operations
2. Improving performance of existing operations
3. Enhancing M3-specific optimizations
4. Adding more examples and benchmarks
5. Improving documentation

## License

The Triton Metal backend follows the same license as Triton itself.

## Metal IR Transformations

The Metal backend includes a sophisticated IR transformation system that optimizes Triton operations for Apple Silicon GPUs. These transformations take advantage of Metal Performance Shaders (MPS) and hardware-specific capabilities, especially for M3 chips.

### Key Components

1. **Pattern Matching**: Identifies operation patterns in the IR that can be mapped to optimized MPS operations.

2. **MPS Transformation**: Transforms identified patterns into optimized MPS operations.

3. **M3-Specific Optimizations**: Special optimizations for Apple Silicon M3 chips, including:
   - Dynamic caching for matrix operations
   - Optimized matrix engine usage
   - Sparse matrix acceleration

4. **Barrier Optimization**: Eliminates unnecessary synchronization barriers.

5. **Memory Layout Optimization**: Optimizes memory access patterns for Metal GPUs.

### Usage

The IR transformations are automatically applied during the compilation process. The compiler will:

1. Parse the Triton IR
2. Apply Metal-specific transformations
3. Map operations to MPS implementations where beneficial
4. Generate optimized Metal code

### Advanced Usage

For debugging or performance analysis, you can directly apply transformations to an IR file:

```python
import metal_ir_transforms

# Load IR operations from file or create manually
ir_ops = [...]  # List of IR operation dictionaries

# Apply transformations
transformed_ops, summary = metal_ir_transforms.transform_ir(ir_ops)

# Use the transformed operations or examine the summary
print(summary)
```

### Testing

The transformation system includes unit tests that verify the correct functioning of each component. Run the tests with:

```bash
python test_metal_ir_transforms.py
```

To generate a test IR file for manual inspection:

```bash
python test_metal_ir_transforms.py --create-test-ir
```

This will create a file `test_ir.json` with a sample IR that can be used for testing.

## Performance Considerations

The Metal IR transformations aim to maximize performance by:

1. Reducing operation count through fusion
2. Leveraging specialized Metal hardware features
3. Minimizing synchronization overhead
4. Optimizing memory access patterns

For the best performance on M3 chips, ensure that the Metal Performance Shaders are available and the Metal 3.2 features are enabled.

# Triton Metal Backend with M3 Optimizations

This directory contains the implementation of the Triton Metal backend, specifically optimized for Apple Silicon GPUs, with special focus on M3-specific optimizations.

## Overview

The Triton Metal backend enables running Triton kernels on Apple Silicon GPUs using the Metal API. It leverages the MLX framework as the underlying compute engine and provides optimizations tailored to each Apple Silicon generation.

## M3-Specific Optimizations

The backend includes specialized optimizations for M3 chips that leverage the unique hardware features of the M3 architecture:

- Enhanced SIMD operations (32-wide vs 16-wide on previous generations)
- Improved vectorization (8-wide vs 4-wide on previous generations)
- Larger shared memory (64KB vs 32KB on previous generations)
- Tensor cores for matrix operations
- Dynamic register caching

Key components:

1. **M3 Graph Optimizer** (`m3_graph_optimizer.py`): Applies M3-specific optimizations to computation graphs
2. **M3 Memory Manager** (`m3_memory_manager.py`): Optimizes memory layouts for M3's memory hierarchy
3. **M3 Fusion Optimizer** (`m3_fusion_optimizer.py`): Implements operation fusion patterns for M3 hardware

## Integration

The backend automatically detects when it's running on M3 hardware and applies the appropriate optimizations without requiring any changes to existing Triton code.

## Documentation

For detailed information about the M3 optimizations, see:

- [M3 Optimizations Guide](../../metal/python/docs/M3_OPTIMIZATIONS.md) - Comprehensive overview of M3-specific optimizations
- [Performance Optimization Guide](../../metal/python/docs/PERFORMANCE_OPTIMIZATION.md) - General performance optimization techniques
- [Architecture Document](../../metal/python/docs/ARCHITECTURE.md) - Overall Metal backend architecture

## Testing

The M3-specific optimizations can be tested using the provided test scripts:

```bash
# Test M3 graph optimizer
python tests/test_m3_graph_optimizer.py

# Test M3 memory manager
python tests/test_m3_memory_manager.py

# Test M3 fusion optimizer
python tests/test_m3_optimizations.py

# Integration test
python tests/test_m3_integration.py
```

## Performance

On M3 hardware, these optimizations can provide significant performance improvements:

- Matrix multiplication: Up to 1.5x faster
- Reduction operations: Up to 1.4x faster
- Element-wise operations: Up to 1.45x faster

## Requirements

- Apple Silicon Mac with M3 chip (for M3-specific optimizations)
- macOS 13.5 or higher
- MLX 0.3.0 or higher

# Triton Metal Backend Compatibility Tools

This directory contains tools for checking and verifying compatibility of your system with the Triton Metal backend, with a focus on Apple Silicon (particularly M3+) compatibility and support for optimized memory layouts.

## Interactive Tutorial

The `tutorial_metal_compatibility.py` script provides a comprehensive interactive tutorial that:
- Checks your system compatibility with the Metal backend
- Detects and explains M3-specific optimizations
- Demonstrates memory layout optimizations for Apple Silicon
- Includes a simple example to verify functionality

To run the tutorial:
```bash
python tutorial_metal_compatibility.py
```

See [tutorial_README.md](tutorial_README.md) for more information.

## System Compatibility Check

The `check_system.py` script verifies if your system meets all the requirements to run the Triton Metal backend with optimized memory layouts (such as COALESCED layout for reduction operations).

### Prerequisites

To run the compatibility check, you need:
- macOS 13.5 or newer
- Apple Silicon Mac (M1, M2, or M3 series)
- Python 3.8 or newer
- The MLX package installed
- The Metal backend modules installed

**Note:** Unlike regular Triton usage, the Metal backend implementation does not require the Triton package itself to be installed. The Metal backend provides its own implementation.

### Running the Check

To run the system compatibility check:

```bash
python check_system.py
```

### What Does It Check?

The tool checks the following:

1. **System Requirements**
   - macOS version (13.5+ required)
   - Apple Silicon hardware presence
   - M-series generation detection (M3 recommended)
   - M3-specific optimizations if running on M3 hardware

2. **Required Packages**
   - MLX installation

3. **Metal Backend Components**
   - Metal hardware detection capabilities
   - COALESCED memory layout definition

4. **Implementation Verification**
   - Consistent definitions across modules
   - ReductionLayoutPattern implementation
   - Memory manager implementation

### Apple M3 Considerations

When running on Apple M3 hardware, the tool will check for M3-specific optimizations in the Metal backend. M3 chips offer additional benefits for the COALESCED memory layout, particularly for reduction operations.

If you're running on an M1 or M2 chip, you'll receive a note that M3 or newer chips are recommended for optimal performance, although the backend will still function correctly.

## Testing

To run the unit tests for the system compatibility check:

```bash
python test_check_system.py
```

## Troubleshooting

If any checks fail, the script will provide details about what's missing or misconfigured. Common issues include:

1. **Outdated macOS**: Update to macOS 13.5 or newer
2. **Missing packages**: Install required packages using pip (mainly MLX)
3. **Backend components not found**: Ensure the Metal backend is properly installed
4. **Inconsistent definitions**: This may indicate an installation issue or version mismatch

## Getting Support

If you encounter issues with the compatibility checker or the Metal backend in general, please:

1. Check the Triton documentation
2. Verify all required components are installed
3. Check for updates to the Metal backend
4. File an issue in the Triton repository if the problem persists

## Contributing

If you'd like to contribute to the Metal backend or these compatibility tools, please see the main Triton contribution guidelines. 