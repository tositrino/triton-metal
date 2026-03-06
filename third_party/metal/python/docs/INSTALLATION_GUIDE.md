# Triton Metal Backend Installation Guide

This guide provides detailed instructions for installing and using the Triton Metal backend on Apple Silicon Macs.

## System Requirements

### Hardware Requirements
- Apple Silicon Mac (M1, M2, or M3 series)
- Minimum 8GB RAM (16GB+ recommended for larger models)

### Software Requirements
- macOS 13.5 or higher
- Python 3.8 or higher
- Xcode 15.0 or higher with Command Line Tools
- MLX 0.3.0 or higher

## Installation

### 1. Environment Setup

First, ensure you have the required development tools installed:

```bash
# Install Command Line Tools if not already installed
xcode-select --install

# Verify installation
xcode-select -p
```

### 2. Python Environment

We recommend using a virtual environment:

```bash
# Create a new virtual environment
python -m venv triton-metal-env

# Activate the environment
source triton-metal-env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install MLX

Install the MLX framework, which is required for the Metal backend:

```bash
pip install mlx>=0.3.0
```

### 4. Install Triton with Metal Backend

#### Option 1: Install from Source

```bash
# Clone the repository
git clone https://github.com/chenxingqiang/triton-metal.git
cd triton

# Switch to the branch with Metal support
git checkout metal-backend

# Install
pip install -e .
```

#### Option 2: Install via Pip (when available)

```bash
pip install triton-metal
```

### 5. Verify Installation

Run the environment checker to verify that everything is set up correctly:

```bash
python -c "import triton; print(triton.backends.metal.check_environment())"
```

You should see confirmation that the Metal backend is available.

## Configuration

### Environment Variables

You can configure the Metal backend using the following environment variables:

- `triton_DEBUG=1`: Enable debug mode with verbose logging
- `triton_CACHE_DIR=/path/to/cache`: Set custom cache directory
- `triton_FORCE_SYNC=1`: Force synchronous execution
- `triton_DISABLE_AUTOTUNING=1`: Disable autotuning system
- `MLX_METAL_USE_SYSTEM_GPU=1`: Force using the system GPU

### Configuration File

Advanced settings can be configured in `~/.triton/metal_config.json`:

```json
{
    "cache_dir": "/path/to/cache",
    "debug_level": 0,
    "auto_tuning": true,
    "preferred_threadgroup_size": 128,
    "max_shared_memory": "auto",
    "optimization_level": 3
}
```

## Usage

### Basic Usage

Using the Metal backend is transparent - the Triton compiler automatically selects the Metal backend when running on Apple Silicon:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

# The kernel will automatically run on Metal when on Apple Silicon
```

### Explicit Backend Selection

You can explicitly specify the Metal backend:

```python
@triton.jit(backend="metal")
def my_kernel(...):
    # Kernel code
    ...
```

### Metal-Specific Optimizations

For Metal-specific optimizations, you can use the `metal` namespace:

```python
import triton
import triton.language as tl
import triton.backends.metal as metal

@triton.jit
def optimized_kernel(...):
    # Use Metal-specific optimizations when available
    if metal.is_available():
        # Metal-specific code path
        ...
    else:
        # Generic fallback
        ...
```

### Memory Layout Optimizations

Metal-specific memory layout optimizations can significantly improve performance:

```python
# Example of using COALESCED layout for reduction operations
@triton.jit
def reduction_kernel(
    input_ptr, output_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Using COALESCED layout improves performance on Metal
    x = tl.load(input_ptr + offsets, mask=mask, layout="COALESCED")
    
    # Perform reduction
    result = tl.sum(x, axis=0)
    
    # Store result
    if pid == 0:
        tl.store(output_ptr, result)
```

## Performance Tuning

### Autotuning

The Metal backend includes an autotuning system that optimizes kernel parameters. Enable it with:

```python
@triton.jit(autotune=True)
def my_kernel(...):
    # Kernel code
    ...
```

### Manual Tuning

For manual performance tuning, consider:

1. **Threadgroup Size**: Optimal size depends on your operation and chip generation
   ```python
   @triton.jit(config={'THREADS_PER_WARP': 32, 'WARPS_PER_BLOCK': 4})
   ```

2. **Memory Access Patterns**: Adjust based on your operation type
   ```python
   # For reduction operations on Metal, COALESCED layout is often optimal
   data = tl.load(ptr + offsets, mask=mask, layout="COALESCED")
   ```

3. **Shared Memory Usage**: Optimize shared memory access patterns
   ```python
   # Declare shared memory with optimal layout
   shared_mem = tl.shared_memory(shape, type)
   ```

### Chip-Specific Optimizations

Different Apple Silicon generations have different optimal settings:

- **M1**: Basic optimizations, focus on memory access patterns
- **M2**: Enhanced atomic operations, better half-precision support
- **M3**: Advanced memory layouts, operation fusion

## Troubleshooting

### Common Issues

1. **Compilation Errors**:
   - Check macOS and Xcode versions
   - Verify MLX installation
   - Examine error messages for unsupported operations

2. **Performance Issues**:
   - Try different threadgroup sizes
   - Experiment with memory layouts
   - Enable autotuning

3. **Memory Errors**:
   - Check for buffer overflows
   - Verify memory alignment
   - Ensure proper synchronization

### Debugging

Enable debug mode for detailed logging:

```bash
triton_DEBUG=1 python your_script.py
```

### Support Resources

- [GitHub Issues](https://github.com/chenxingqiang/triton-metal/issues)
- [Triton Discussion Forum](https://github.com/chenxingqiang/triton-metal/discussions)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)

## Advanced Topics

### Custom Metal Operations

For operations not supported by Triton, you can create custom Metal implementations:

```python
import triton
import mlx.core as mx

# Register a custom operation implementation
@triton.backends.metal.register_custom_op("my_custom_op")
def custom_op_implementation(x, y):
    # Implement using MLX operations
    return mx.custom_metal_function(x, y)

# Use in Triton kernel
@triton.jit
def kernel_with_custom_op(...):
    # ...
    result = tl.custom("my_custom_op", x, y)
    # ...
```

### Interfacing with MLX

The Metal backend can interoperate with MLX:

```python
import mlx.core as mx
import triton
import triton.language as tl

# Create MLX arrays
x_mlx = mx.random.normal((1024, 1024))
y_mlx = mx.random.normal((1024, 1024))

# Define Triton kernel
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    # Kernel implementation
    ...

# Launch using MLX arrays directly
grid = (1024 // 128 + 1,)
add_kernel[grid](x_mlx, y_mlx, output, 1024, BLOCK_SIZE=128)
```

## Performance Benchmarks

For reference, here are typical performance numbers for common operations on different Apple Silicon chips:

| Operation | Size | M1 | M2 | M3 |
|-----------|------|-----|-----|-----|
| MatMul | 1024×1024 (FP32) | 2.5 TFLOPs | 3.2 TFLOPs | 4.8 TFLOPs |
| MatMul | 1024×1024 (FP16) | 4.8 TFLOPs | 7.5 TFLOPs | 9.6 TFLOPs |
| Reduction | 16M elements | 250 GB/s | 320 GB/s | 450 GB/s |
| ElementWise | 16M elements | 320 GB/s | 400 GB/s | 580 GB/s |

*Note: These are approximate values and may vary based on specific hardware configurations and software versions.* 