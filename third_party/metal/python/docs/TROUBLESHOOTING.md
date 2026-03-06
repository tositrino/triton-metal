# Triton Metal Backend Troubleshooting Guide

This guide helps diagnose and resolve common issues with the Triton Metal backend on Apple Silicon Macs.

## Diagnostic Tools

### Environment Checker

The Metal backend includes a built-in environment checker to verify your system configuration:

```python
# Run from Python
import triton
triton.backends.metal.check_environment()

# Or from command line
python -m triton.backends.metal.check_environment
```

This will check:
- Python version
- macOS version
- Metal support
- MLX installation
- GPU detection
- Compiler availability

### Debug Mode

Enable debug mode to get detailed logs:

```bash
triton_DEBUG=1 python your_script.py
```

Debug levels:
- Level 1: Basic information about kernel compilation and execution
- Level 2: Detailed operation mappings and intermediate representations
- Level 3: Complete shader code and hardware-specific details

### Performance Profiler

Use the built-in profiler to identify performance bottlenecks:

```python
import triton
import triton.backends.metal as metal

# Start profiling
metal.start_profiling()

# Run your code
...

# Stop profiling and get report
report = metal.stop_profiling()
print(report)
```

## Common Issues and Solutions

### Installation Issues

#### MLX Not Found

**Symptoms:**
- Error: `ImportError: No module named 'mlx'`
- Error: `Cannot import MLX backend for Metal support`

**Solutions:**
1. Install MLX: `pip install mlx>=0.3.0`
2. Verify installation: `python -c "import mlx; print(mlx.__version__)"`
3. Check that you're using the correct Python environment

#### Metal Support Not Detected

**Symptoms:**
- Error: `Metal is not supported on this system`
- Error: `Could not initialize Metal device`

**Solutions:**
1. Verify you're using an Apple Silicon Mac
2. Ensure macOS version is 13.5+
3. Check Xcode installation: `xcode-select -p`
4. Run: `xcrun metal --version`

### Compilation Issues

#### Unsupported Operations

**Symptoms:**
- Error: `Operation X is not supported by the Metal backend`
- Error: `Cannot convert Triton operation to MLX operation`

**Solutions:**
1. Use supported alternative operations
2. Check the operation compatibility table in the documentation
3. Implement a custom operation using MLX directly

```python
# Example of implementing a custom equivalent
@triton.jit
def my_kernel(...):
    # Instead of unsupported_op
    # Use a combination of supported operations
    result = tl.operation1(tl.operation2(x))
```

#### Memory Layout Issues

**Symptoms:**
- Error: `Cannot convert memory layout`
- Performance is much lower than expected

**Solutions:**
1. Use `COALESCED` layout for reduction operations
2. Use `BLOCK` layout for matrix operations
3. Avoid complex strided memory access patterns

```python
# Example fix for memory layout issues
# Instead of this
x = tl.load(ptr + complex_offset_calculation)

# Try explicit layout
x = tl.load(ptr + simple_offset, layout="COALESCED")
```

#### Compiler Errors

**Symptoms:**
- Error: `Metal shader compilation failed`
- Error with long shader code and Metal error message

**Solutions:**
1. Check for unsupported operations or patterns
2. Reduce kernel complexity
3. Break large kernels into smaller ones
4. Check resource limits (threadgroup memory, etc.)

### Runtime Issues

#### Out of Memory

**Symptoms:**
- Error: `Metal runtime error: out of memory`
- Process crashes without error

**Solutions:**
1. Reduce batch sizes or model dimensions
2. Use smaller data types (e.g., float16 instead of float32)
3. Release unused memory explicitly: `triton.backends.metal.clear_cache()`
4. Check system memory pressure with Activity Monitor

#### Incorrect Results

**Symptoms:**
- Results differ from expected values
- Results differ from CUDA backend

**Solutions:**
1. Check for numerical precision issues (especially with float16)
2. Verify kernel logic and memory access patterns
3. Add explicit synchronization: `triton.backends.metal.synchronize()`
4. Compare results with CPU implementation

#### Slow Performance

**Symptoms:**
- Metal backend is much slower than expected
- Operations take longer than CUDA equivalent

**Solutions:**
1. Enable autotuning: `@triton.jit(autotune=True)`
2. Optimize threadgroup sizes for your chip
3. Check memory access patterns
4. Use chip-specific optimizations

```python
# Example of autotuning configuration
@triton.jit(
    autotune=True,
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ]
)
def my_kernel(...):
    # Kernel code
```

### Chip-Specific Issues

#### M1 Issues

**Symptoms:**
- Atomic operations are slow
- Performance degrades for certain workloads

**Solutions:**
1. Avoid atomic operations when possible
2. Use reduction patterns instead of atomics
3. Optimize for bandwidth rather than compute

#### M2 Issues

**Symptoms:**
- Half-precision performance is not optimal
- Unexpected performance differences from M1

**Solutions:**
1. Explicitly use float16 for compute-intensive operations
2. Adjust threadgroup sizes for M2 (typically larger than M1)
3. Use M2-specific optimizations via hardware detection

#### M3 Issues

**Symptoms:**
- Advanced optimization features not being utilized
- Operation fusion not occurring automatically

**Solutions:**
1. Ensure you're using latest MLX version with M3 optimizations
2. Manually adjust fusion patterns
3. Use explicit memory layout optimizations

## Advanced Troubleshooting

### Shader Debugging

For shader debugging, you can extract and analyze the Metal shaders:

```python
import triton
import triton.backends.metal as metal

# Enable shader dump
metal.set_debug_option("dump_shaders", True)

# Run your kernel
@triton.jit
def my_kernel(...):
    # Kernel code
    ...

# Shaders will be dumped to the specified directory
# Default: ~/.triton/metal/shaders/
```

You can then examine the shaders using Metal Developer Tools.

### Memory Layout Analyzer

Use the memory layout analyzer to understand memory access patterns:

```python
import triton.backends.metal as metal

# Analyze memory accesses
report = metal.analyze_memory_access(your_function, sample_inputs)
print(report)
```

### Comparing with Reference Implementations

Create a reference CPU implementation to validate results:

```python
def cpu_reference(x, y):
    # CPU implementation of the same logic
    return x + y

# Compare with Metal implementation
metal_result = metal_kernel(x_metal, y_metal)
cpu_result = cpu_reference(x_cpu, y_cpu)

# Check if results match within tolerance
import numpy as np
assert np.allclose(metal_result, cpu_result, rtol=1e-5, atol=1e-5)
```

## Hardware-Specific Diagnostics

### Check Hardware Capabilities

```python
import triton.backends.metal as metal

# Get hardware capabilities
capabilities = metal.get_hardware_capabilities()
print(capabilities)

# Check specific features
if capabilities["supports_fast_atomics"]:
    # Use fast atomic operations
    ...
else:
    # Use alternative approach
    ...
```

### Memory Bandwidth Test

Test raw memory bandwidth to diagnose performance issues:

```python
import triton.backends.metal as metal

# Run memory bandwidth test
bandwidth = metal.test_memory_bandwidth(size_mb=1024)
print(f"Memory bandwidth: {bandwidth:.2f} GB/s")

# Compare with expected values for your chip
# M1: ~200-250 GB/s
# M2: ~250-320 GB/s
# M3: ~350-450 GB/s
```

### Compute Throughput Test

Test raw compute throughput:

```python
import triton.backends.metal as metal

# Run compute throughput test
tflops = metal.test_compute_throughput(dtype="float32")
print(f"Compute throughput: {tflops:.2f} TFLOPs")

# Compare with expected values for your chip
```

## Compatibility Checks

### Check MLX Compatibility

```python
import mlx.core as mx

# Check MLX version
print(f"MLX version: {mx.__version__}")

# Verify MLX device
print(f"MLX device: {mx.get_default_device()}")

# Run basic MLX test
a = mx.array([1, 2, 3])
b = a + 1
mx.eval(b)
print("MLX basic test passed")
```

### Check Metal Framework Compatibility

```bash
# Check Metal framework version
system_profiler SPDisplaysDataType | grep Metal

# Check Metal compiler
xcrun metal -v
```

## Submitting Bug Reports

When submitting bug reports, include:

1. **System Information**:
   - Apple Silicon model (M1/M2/M3 and variant)
   - macOS version
   - Python version
   - MLX version
   - Triton version

2. **Reproduction Steps**:
   - Minimal code example that reproduces the issue
   - Expected vs. actual behavior
   - Performance metrics if applicable

3. **Logs and Diagnostics**:
   - Debug logs (run with `triton_DEBUG=1`)
   - Environment check results
   - Error messages
   - Performance profiling results if relevant

Submit issues to: [Triton GitHub Issues](https://github.com/chenxingqiang/triton-metal/issues) 