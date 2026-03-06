# Triton Metal Dialect Tests

This directory contains tests for the Triton Metal dialect, which extends the Triton dialect with Metal-specific operations and transformations for Apple Silicon GPUs.

## Test Categories

The tests are organized into several directories:

1. **IR Tests** (`IR/`): Tests for the TritonMetal dialect IR operations
2. **Transformation Tests** (`Transforms/`): Tests for the TritonMetal dialect transformation passes
3. **Integration Tests** (`./*.cpp`): Tests for integration with other components

## IR Tests

The IR tests validate the basic functionality of the TritonMetal dialect:

- **DialectTest.cpp**: Tests loading the TritonMetal dialect and creating modules with it
- Additional tests for specific operations as they are added to the dialect

## Transformation Tests

The transformation tests validate the TritonMetal-specific passes:

- **TransformsTest.cpp**: Tests for core transformation passes
- **MemoryOptimizerTest.cpp**: Tests for memory optimization passes
- **M3OptimizationsTest.cpp**: Tests for M3-specific optimization passes

## Integration Tests

The integration tests validate how the TritonMetal dialect works with other components:

- **MLXIntegrationTest.cpp**: Tests integration with the MLX framework
- **HardwareDetectionTest.cpp**: Tests hardware detection for Apple Silicon

## Running the Tests

To run the TritonMetal dialect tests, use the following command:

```bash
# Run all tests
ninja check-triton-metal

# Run specific test
./build/unittest/Dialect/TritonMetal/IR/TestTritonMetalDialect
./build/unittest/Dialect/TritonMetal/Transforms/TestTritonMetalTransforms
```

## Hardware-Specific Testing

Many tests are specifically designed for Apple Silicon and use the `__APPLE__` preprocessor macro to conditionally skip on non-Apple platforms.

## Test Structure

Each test follows the standard Google Test structure:

1. **Test fixtures**: Set up the test environment (e.g., MLIRContext with the TritonMetal dialect loaded)
2. **Test cases**: Individual test cases that validate specific functionality
3. **Main function**: Initializes Google Test and runs the tests

## Adding New Dialect Features

When adding new features to the TritonMetal dialect:

1. Add new IR tests for the operations
2. Add new transformation tests for the passes
3. Update the existing tests to cover the new functionality

## Dependencies

The tests depend on:
- MLIR testing utilities
- Google Test
- The TritonMetal dialect implementation

## Future Expansion

As the TritonMetal dialect evolves to support more Metal-specific features, additional tests will be added to validate:

- M3-specific operations and optimizations
- Integration with MLX framework
- Integration with the Metal compiler pipeline
- Performance optimization passes

## Note on Integration Tests

The integration tests may use mock implementations when the actual components are not available, allowing the tests to run even without Apple Silicon hardware. 