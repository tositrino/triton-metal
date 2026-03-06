# Metal Backend Testing Guide

This document provides an overview of the testing structure for the Triton Metal backend, which enables Triton to run efficiently on Apple Silicon GPUs.

## Testing Architecture

The Metal backend testing is organized into two main categories:

1. **C++ Unit Tests** - Located in the `unittest/` directory, these test the Metal dialect, transformations, and MLIR integration
2. **Python Integration Tests** - Located in the `python/triton/metal/tests/` directory, these test the Python API and end-to-end functionality

## C++ Unit Tests

### Dialect Tests
Tests for the TritonMetal dialect IR and operations:
- `unittest/Dialect/TritonMetal/IR/DialectTest.cpp`

### Transform Tests
Tests for Metal-specific optimization passes:
- `unittest/Dialect/TritonMetal/Transforms/TransformsTest.cpp`
- `unittest/Dialect/TritonMetal/Transforms/MemoryOptimizerTest.cpp`
- `unittest/Dialect/TritonMetal/Transforms/M3OptimizationsTest.cpp`

### Integration Tests
Tests for integration with other components:
- `unittest/Dialect/TritonMetal/MLXIntegrationTest.cpp`
- `unittest/Dialect/TritonMetal/HardwareDetectionTest.cpp`

### Platform-Specific Tests
Tests that directly interact with Metal APIs:
- `unittest/Metal/MetalBackendTest.cpp`
- `unittest/Metal/M3OptimizationsTest.cpp`
- `unittest/Metal/MetalMemoryManagerTest.cpp`

## Python Integration Tests

The Python integration tests verify that the Metal backend works correctly when used through the Python API:

- `python/triton/metal/tests/test_integration.py` - Basic integration tests
- `python/triton/metal/tests/test_metal_memory_manager.py` - Memory manager tests
- `python/triton/metal/tests/test_reduction_memory.py` - Reduction operation tests
- `python/triton/metal/tests/test_m3_optimizations.py` - M3-specific optimizations tests

## Running the Tests

### C++ Unit Tests
```bash
cd build
ninja check-triton-unit-tests
```

### Python Integration Tests
```bash
cd python
python -m pytest triton/metal/tests
```

## Adding New Tests

### Adding C++ Unit Tests
1. Add the test file to the appropriate directory
2. Update the corresponding CMakeLists.txt file
3. Make sure to conditionally run tests only on Apple Silicon hardware
4. Document the test in the appropriate README

### Adding Python Integration Tests
1. Add the test file to the `python/triton/metal/tests` directory
2. Make sure to skip tests when not running on Apple Silicon
3. Update the `run_tests.py` script

## Continuous Integration

The Metal backend tests are run automatically on GitHub Actions using macOS runners with Apple Silicon hardware. The workflow is defined in `.github/workflows/metal_tests.yml`.

## Test Requirements

- Apple Silicon hardware (M1, M2, or M3)
- macOS 13.5 or higher
- Metal framework
- MLX framework (version 0.3.0 or higher) 