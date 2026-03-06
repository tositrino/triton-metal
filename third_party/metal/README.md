# Triton Metal Backend

This directory contains the Metal backend implementation for Triton, enabling Triton kernels to run on Apple Silicon GPUs (M1, M2, and M3 series).

## Directory Structure

The Metal backend is organized as follows:

```
third_party/metal/
├── backend/                # Metal backend implementation
│   ├── include/            # Metal backend headers
│   └── lib/                # Metal backend libraries
├── include/                # Public Metal dialect headers
│   └── triton/
│       └── Dialect/
│           └── TritonMetal/
│               ├── IR/     # Metal IR definitions
│               └── Transforms/ # Metal-specific transformations
├── language/               # Metal language extension
│   └── metal/              # Metal-specific functions and utilities
├── lib/                    # Implementation of Metal dialect and transforms
│   └── Dialect/
│       └── TritonMetal/
│           ├── IR/          # Metal IR implementation
│           └── Transforms/  # Metal transform implementation
└── python/                 # Python bindings for Metal backend
    └── triton/       # Metal-specific Python implementation
        ├── __init__.py     # Package initialization
        ├── compiler.py     # Metal compiler implementation
        ├── runtime.py      # Metal runtime implementation
        ├── check_system.py # System compatibility checker
        └── tests/          # Metal backend tests
```

## Prerequisites

- macOS 11.0 (Big Sur) or later
- Apple Silicon Mac (M1, M2, or M3 series)
- MLX (Apple's machine learning framework)

## Installation

The Metal backend is included as part of the Triton installation on Apple Silicon Macs. To install:

```bash
pip install triton
```

Or from source:

```bash
git clone https://github.com/openai/triton.git
cd triton
python setup.py install
```

## Usage

### Basic Usage

To use the Metal backend, you need to specify the target when creating a Triton function:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

# Use the Metal backend
grid = (n_elements + 1024 - 1) // 1024
add_kernel[grid](x_ptr, y_ptr, output_ptr, n_elements, 1024, target='metal')
```

### Metal-Specific Features

The Metal backend includes several specific features for Apple Silicon GPUs:

```python
import triton
import triton.language as tl
from triton.backends.metal import get_chip_generation, get_max_threads_per_threadgroup

# Get Metal-specific information
chip_gen = get_chip_generation()
max_threads = get_max_threads_per_threadgroup()

print(f"Running on {chip_gen} with {max_threads} max threads per threadgroup")
```

## Testing

To run the Metal backend tests:

```bash
cd third_party/metal/python
python -m unittest discover -s triton/tests
```

## Contributing

When contributing to the Metal backend, please ensure your changes follow these guidelines:

1. Follow the existing code style and structure
2. Add appropriate tests for new features
3. Update documentation as needed
4. Ensure compatibility with the latest macOS and Apple Silicon chips

## License

The Metal backend is part of the Triton project and is licensed under the same terms as Triton. 