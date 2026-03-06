# Metal Backend Testing Architecture

This document describes the testing architecture for the Metal backend in Triton.

## Overview

The testing framework for the Metal backend is divided into several categories:

1. **Dialect-level Tests**: Tests for the Metal dialect operations, transformations, and optimizations
2. **Backend-level Tests**: Tests for the Metal backend implementation, including hardware-specific features
3. **Integration Tests**: Tests for the integration with other frameworks like MLX
4. **Hardware-specific Tests**: Tests targeting specific Apple Silicon generations (M1, M2, M3)

## Test Structure

The tests are organized in two main directories:

- `unittest/Dialect/TritonMetal/`: Tests for the TritonMetal dialect
- `unittest/Metal/`: Tests for the Metal backend implementation

### Dialect-level Tests

These tests focus on the MLIR dialect definition, including operations, transformations, and passes.

- `DialectTest.cpp`: Tests for the Metal dialect operations
- `TransformsTest.cpp`: Tests for Metal-specific passes
- `MemoryOptimizerTest.cpp`: Tests for memory optimization passes
- `M3OptimizationsTest.cpp`: Tests for M3-specific optimizations

### Backend-level Tests

These tests focus on the Metal backend implementation, hardware detection, and optimizations.

- `MetalBackendTest.cpp`: Tests for Metal backend functionality
- `MetalMemoryManagerTest.cpp`: Tests for memory management
- `OperationFusionTest.cpp`: Tests for operation fusion optimizations
- `HardwareDetectionTest.cpp`: Tests for hardware detection
- `MLXIntegrationTest.cpp`: Tests for MLX framework integration
- `TensorCoreTest.cpp`: Tests for tensor core utilization on M3 hardware

## Running Tests

The tests can be run using the `run_metal_tests` executable. On non-Apple hardware, the tests that require Metal hardware will be skipped automatically.

### Configuration

Tests can be configured using environment variables:

- `triton_IS_M3=1`: Simulate running on M3 hardware
- `triton_GENERATION=M1|M2|M3`: Specify the Apple Silicon generation

## M3-specific Optimizations

The Metal backend includes several optimizations specific to the M3 chip:

1. **Larger shared memory**: 64KB vs 32KB on M1/M2
2. **Wider vector operations**: 8-wide vs 4-wide on M1/M2
3. **Tensor core support**: Enhanced tensor cores for matrix operations
4. **Dynamic caching**: Improved cache management

The tests verify that these optimizations are correctly applied when running on M3 hardware.

## Tensor Core Testing

The `TensorCoreTest.cpp` file contains tests specifically for M3 tensor core functionality:

1. **Detection**: Tests that tensor cores are correctly detected on M3 hardware
2. **Dimensions**: Tests optimal matrix dimensions for tensor core operations
3. **Computation**: Tests matrix multiplication with tensor cores
4. **Mixed precision**: Tests FP16 computation with FP32 accumulation
5. **Performance**: Compares tensor core vs standard implementation performance

## MLX Integration

The Metal backend includes integration with Apple's MLX framework for accelerated array computing. The tests verify:

1. **Compilation**: Triton kernels can be compiled to MLX
2. **Execution**: Triton operations can be executed using MLX
3. **Performance**: M3-specific optimizations are applied when using MLX

## Memory Management

The tests also verify the Metal memory management features:

1. **Optimal tile sizes**: Based on hardware capabilities
2. **Threadgroup sizes**: Optimized for different chip generations
3. **Vector width**: Optimal vector width based on hardware
4. **Memory layout**: Optimal memory layout for different operations

## Adding New Tests

To add new tests, follow these steps:

1. Create a new test file in the appropriate directory
2. Add the test to the corresponding CMakeLists.txt
3. Add the test to run_metal_tests.cpp

For hardware-specific tests, make sure to include the following:

```cpp
#ifdef __APPLE__
  // Test code here
#else
  GTEST_SKIP() << "Test only runs on Apple hardware";
#endif
```

For M3-specific tests, use the `isAppleM3Hardware()` utility function to conditionally run tests. 