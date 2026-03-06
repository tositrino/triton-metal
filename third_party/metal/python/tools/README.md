# Metal Backend Tools

This directory contains tools for analyzing and working with the Metal backend for Apple Silicon GPUs.

## Overview

The Metal backend provides specialized optimizations for Apple Silicon GPUs, including the COALESCED memory layout for reduction operations. These tools help developers understand how these optimizations are applied and benchmark their performance.

## Available Tools

### 1. Simple Analyzer (`simple_analyzer.py`)

A lightweight tool for identifying which reduction operations would use the COALESCED memory layout.

**Usage:**
```bash
# Analyze operations from a JSON file
./simple_analyzer.py --json sample_ops.json

# Analyze a single operation
./simple_analyzer.py --operation "tt.reduce:[1024,512]:1"

# Output results to a file
./simple_analyzer.py --json sample_ops.json --output results.json

# Enable verbose output
./simple_analyzer.py --json sample_ops.json --verbose

# Show version information
./simple_analyzer.py --version
```

### 2. Memory Layout Analyzer (`analyze_memory_layouts.py`)

A more comprehensive tool that analyzes which memory layouts are applied to different operations, with detailed information about optimization parameters.

**Usage:**
```bash
# Analyze operations from a JSON file
./analyze_memory_layouts.py --json sample_ops.json

# Output results to a file
./analyze_memory_layouts.py --json sample_ops.json --output results.json

# Enable verbose output
./analyze_memory_layouts.py --json sample_ops.json --verbose
```

### 3. Sample Operation Generator (`create_sample_ops.py`)

Generates sample operation JSON files for testing the analyzers.

**Usage:**
```bash
# Generate a basic sample
./create_sample_ops.py --output sample_ops.json

# Generate a larger sample with more operations
./create_sample_ops.py --output large_sample_ops.json --large
```

### 4. Reduction Benchmark (`benchmark_reduction_layouts.py`)

Benchmarks the performance of reduction operations with different memory layouts.

**Usage:**
```bash
# Run benchmark with default settings
./benchmark_reduction_layouts.py

# Specify problem sizes
./benchmark_reduction_layouts.py --sizes "128x1024,256x1024,512x1024"

# Specify layouts to benchmark
./benchmark_reduction_layouts.py --layouts "DEFAULT,ROW_MAJOR,COALESCED"

# Specify data type
./benchmark_reduction_layouts.py --dtype float16

# Save plot to file
./benchmark_reduction_layouts.py --output benchmark_results.png
```

### 5. Sample Reduction Kernels (`sample_reduction_kernel.py`)

Contains example Triton kernels that implement various reduction operations that would use the COALESCED memory layout.

**Usage:**
```bash
# Run the sample kernels
python sample_reduction_kernel.py
```

## JSON Format

The analyzers expect operations to be defined in a JSON file with the following format:

```json
{
  "ops": [
    {
      "type": "tt.reduce",
      "id": "sum_reduce_1d",
      "input_shapes": [[1024]],
      "args": {"axis": 0}
    },
    {
      "type": "tt.sum",
      "id": "sum_reduce_2d",
      "input_shapes": [[512, 512]],
      "args": {"axis": 1}
    }
  ]
}
```

Each operation should include:
- `type`: The operation type (e.g., "tt.reduce", "tt.sum", "tt.matmul")
- `id`: A unique identifier (optional)
- `input_shapes`: Array of input tensor shapes
- `args`: Additional operation arguments (e.g., reduction axis)

## COALESCED Memory Layout

The COALESCED memory layout (value 8) is a specialized memory layout optimized for reduction operations on Apple Silicon GPUs. It provides significant performance benefits by:

- Ensuring memory accesses are coalesced
- Enhancing SIMD utilization
- Minimizing synchronization overhead
- Leveraging hardware-specific features

For detailed information about the COALESCED memory layout, see the [COALESCED_LAYOUT.md](../docs/COALESCED_LAYOUT.md) documentation.

## How to Check If COALESCED Layout Is Being Applied

To check if your reduction operations are using the COALESCED memory layout:

1. **Using simple_analyzer.py**:
   ```bash
   # For a specific operation
   ./simple_analyzer.py --operation "tt.sum:[1024,1024]:1"
   
   # For operations in your code
   ./simple_analyzer.py --json your_operations.json
   ```

2. **Using the Memory Layout Analyzer**:
   ```bash
   ./analyze_memory_layouts.py --json your_operations.json
   ```

3. **Programmatically**:
   ```python
   from metal_memory_manager import MemoryLayout, get_metal_memory_manager
   
   # Get the memory layout for a specific operation
   layout = get_metal_memory_manager().get_memory_layout_for_op(op)
   
   # Check if it's using COALESCED layout
   is_coalesced = (layout & MemoryLayout.COALESCED.value) == MemoryLayout.COALESCED.value
   ```

## Performance Impact

The COALESCED memory layout typically provides 1.5x-3x speedup for reduction operations compared to other memory layouts. Performance gains are most significant for:

- Large reductions (>1024 elements)
- Row-wise reductions in matrices
- Multi-axis reductions

Run the benchmark tool to see the performance impact on your specific workloads:

```bash
./benchmark_reduction_layouts.py --sizes "128x1024,256x1024,512x1024,1024x1024"
```

## Requirements

- Triton with Metal backend support
- Python 3.7+
- NumPy
- Matplotlib (for benchmarking and visualization)

## Examples

### Example 1: Identify all reduction operations in a project

```bash
find /path/to/project -name "*.py" -exec grep -l "tt.reduce\|tt.sum\|tt.mean" {} \; > reduction_files.txt
cat reduction_files.txt | xargs ./simple_analyzer.py
```

### Example 2: Benchmark different memory layouts

```bash
./benchmark_reduction_layouts.py --sizes "128x1024,256x1024,512x1024,1024x1024" --output benchmark_results.png
```

### Example 3: Generate and analyze a large sample

```bash
./create_sample_ops.py --output large_sample.json --large
./analyze_memory_layouts.py --json large_sample.json --output analysis_results.json
```

### Example 4: Analyze a specific operation from your code

If you have a reduction operation in your code:

```python
@triton.jit
def my_reduction(input_ptr, output_ptr, n, m, ...):
    # Your reduction code here
    ...
```

You can analyze it using:

```bash
./simple_analyzer.py --operation "tt.sum:[n,m]:1"
```

Where `n` and `m` are the dimensions of your input tensor, and `1` is the axis along which you're reducing. 