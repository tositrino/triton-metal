# Metal Backend Test Suite

This directory contains test tools for validating and benchmarking the Metal backend for Triton.

## Contents

- `test_end_to_end.py`: Comprehensive end-to-end tests for the Metal backend
- `test_performance_benchmark.py`: Performance benchmarks for various operations
- `compare_cuda_metal.py`: Tool to compare Metal and CUDA benchmark results
- `check_environment.py`: Script to verify the test environment is correctly set up
- `test_chip_compatibility.py`: Tests for validating Metal backend across Apple Silicon generations (M1/M2/M3)

## Requirements

- Python 3.8+
- MLX (`pip install mlx`)
- NumPy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)
- Metal backend

## Check Environment

Before running the tests, you can check that your environment is correctly set up:

```bash
# Check environment
python check_environment.py
```

This will verify:
- Python version is sufficient
- Required packages are installed
- Running on Apple Silicon Mac
- Metal hardware is available
- Metal backend modules are accessible

## End-to-End Tests

The end-to-end tests verify that the Metal backend works correctly for various operations:

```bash
# Run all tests
python test_end_to_end.py

# Run with specific data types
python test_end_to_end.py --dtypes float32 float16

# Run with specific matrix sizes
python test_end_to_end.py --sizes 64x64 128x128 1024x1024
```

## Performance Benchmarks

The performance benchmarks measure the execution time and GFLOPS for various operations:

```bash
# Run all benchmarks
python test_performance_benchmark.py

# Run specific operations
python test_performance_benchmark.py --ops matmul reduction

# Run with specific data types
python test_performance_benchmark.py --dtypes float32

# Run with specific matrix sizes
python test_performance_benchmark.py --sizes 128x128 1024x1024

# Set number of runs and warmup iterations
python test_performance_benchmark.py --num-runs 20 --warmup 10

# Specify output directory
python test_performance_benchmark.py --output-dir my_benchmark_results
```

## CUDA Comparison Tool

The CUDA comparison tool helps compare the performance of Metal and CUDA backends:

```bash
# Compare benchmark results
python compare_cuda_metal.py --metal-results metal_benchmark_results.json --cuda-results cuda_benchmark_results.json

# Specify output directory
python compare_cuda_metal.py --metal-results metal_benchmark_results.json --cuda-results cuda_benchmark_results.json --output-dir comparison_results
```

This will generate comparison plots and a summary table in the specified output directory.

## Chip Compatibility Tests

The chip compatibility tests validate the Metal backend across different Apple Silicon generations (M1/M2/M3):

```bash
# Run compatibility tests
python test_chip_compatibility.py

# Run with verbose output
python test_chip_compatibility.py --verbose
```

This will run tests tailored to each chip generation and verify that chip-specific optimizations are correctly applied.

## Example Workflow

1. Check environment setup:
   ```bash
   python check_environment.py
   ```

2. Run Metal performance benchmarks:
   ```bash
   python test_performance_benchmark.py --output-dir metal_results
   ```

3. Run CUDA performance benchmarks on an NVIDIA system:
   ```bash
   python test_performance_benchmark.py --output-dir cuda_results
   ```

4. Copy the CUDA benchmark results to the Metal system.

5. Compare the results:
   ```bash
   python compare_cuda_metal.py --metal-results metal_results/benchmark_results.json --cuda-results cuda_results/benchmark_results.json --output-dir comparison
   ```

6. Analyze the comparison results and summary table.

7. Verify chip-specific optimizations:
   ```bash
   python test_chip_compatibility.py --verbose
   ```

## Adding New Tests

To add new tests to the end-to-end test suite:

1. Add new test methods to the `EndToEndTests` class in `test_end_to_end.py`.
2. Update the `run_all_tests` method to include your new tests.

Similarly, to add new benchmarks:

1. Add new benchmark methods to the `MetalBenchmark` class in `test_performance_benchmark.py`.
2. Update the `run_all_benchmarks` method to include your new benchmarks.

To add new chip compatibility tests:

1. Add new test methods to the `ChipCompatibilityTester` class in `test_chip_compatibility.py`.
2. Update the `_register_tests` method to include your new tests with appropriate minimum chip generation requirements. 