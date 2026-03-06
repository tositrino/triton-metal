# Triton Metal Backend Tests

This directory contains tests for the Triton Metal backend, which targets Apple Silicon GPUs (M1, M2, and M3). The tests validate various aspects of the Metal backend implementation, with a particular focus on M3-specific optimizations.

## Test Categories

The tests are organized into several categories:

1. **MetalBackendTest**: Basic functionality tests for the Metal backend
2. **M3OptimizationsTest**: Tests for M3-specific optimizations
3. **MetalMemoryManagerTest**: Tests for the Metal memory management system

## Test Organization

The Metal tests are organized hierarchically:

1. **Dialect Tests** (`unittest/Dialect/TritonMetal/`): Tests for the TritonMetal dialect operations and transformations
2. **Backend Tests** (`unittest/Metal/`): Tests for the Metal backend implementation

## Running the Tests

To run the Metal tests, use the following command:

```bash
# Run all tests
ninja check-triton-metal

# Run specific test
./build/unittest/Metal/TestMetalBackend
./build/unittest/Metal/TestM3Optimizations
./build/unittest/Metal/TestMetalMemoryManager
```

## Hardware-Specific Testing

Many tests are specifically designed to test hardware-specific features and will only run on Apple Silicon hardware. Tests that require Metal support use the `__APPLE__` preprocessor macro to conditionally skip on non-Apple platforms.

For M3-specific tests, the tests can detect the hardware or use environment variables to simulate M3 hardware:

```bash
# Force tests to behave as if running on M3 hardware
export triton_IS_M3=1
```

## Test Structure

Each test follows a similar structure:

1. **Mock implementations**: Simplified versions of the actual implementation for testing
2. **Test fixtures**: Google Test fixtures that set up the test environment
3. **Test cases**: Individual test cases that validate specific functionality
4. **Hardware-specific tests**: Tests that only run on specific hardware platforms

## Testing M3-Specific Features

The M3-specific tests focus on the following features:

- **64KB shared memory** (vs 32KB on M1/M2)
- **8-wide vectorization** (vs 4-wide on M1/M2)
- **32-wide SIMD groups**
- **Enhanced tensor cores**
- **Dynamic caching**

## MLX Integration Tests

The MLX integration tests validate the integration between Triton and the MLX framework for Metal on Apple Silicon. These tests ensure that Triton operations can be correctly mapped to MLX operations.

## Adding New Tests

When adding new tests:

1. Create a new test file in the appropriate directory
2. Add the test to the corresponding CMakeLists.txt file
3. Use the `#ifdef __APPLE__` directive to skip tests on non-Apple platforms
4. Use mock classes to simulate the actual implementation when needed
5. Ensure tests can run with or without actual M3 hardware using environment variables

## Test Reports

Test results are reported using the standard Google Test format. For Metal-specific tests, additional information may be printed to help diagnose issues on specific hardware platforms. 