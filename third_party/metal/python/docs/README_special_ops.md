# Special Mathematical Functions for Triton Metal Backend

This module provides MLX-based implementations of special mathematical functions needed for the Triton Metal backend on Apple Silicon.

## Overview

The `special_ops.py` module implements various special mathematical functions that may not be directly available in MLX but are required by Triton kernels. The implementations are optimized for Apple Silicon hardware and provide accurate numerical approximations.

## Function Categories

The module is organized into two main classes:

1. **SpecialMathFunctions**: Core special mathematical functions
   - Error functions (erf, erfc)
   - Gamma family functions (gamma, lgamma, digamma)
   - Bessel functions (j0, j1, i0e, i1e)
   - Hyperbolic functions (sinh, cosh, asinh, acosh, atanh)
   - Numerically stable variants (expm1, log1p)

2. **NumericalFunctions**: Fast approximations and utility functions
   - Fast approximations (fast_exp, fast_sigmoid, fast_tanh)
   - Numerical utilities (rsqrt)

## Usage

The module is designed to be used by the Triton-to-MLX conversion layer. Functions can be accessed through the operation mapping:

```python
from special_ops import get_special_ops_map

# Get the mapping of Triton ops to MLX implementations
ops_map = get_special_ops_map()

# Access a specific function
erf_func = ops_map['tt.erf']
result = erf_func(input_tensor)
```

## Implementation Details

Most functions follow a tiered implementation strategy:

1. Use MLX built-in functions when available
2. Fall back to accurate numerical approximations when needed
3. For complex functions, use piecewise approximations for different input ranges

## Testing

The implementation includes comprehensive test coverage:

1. **Correctness Tests**: All functions are validated against scipy/numpy reference implementations
2. **Range Tests**: Functions are tested across various input ranges (small, medium, large values)
3. **Performance Tests**: Comparative benchmarks against reference implementations

To run the tests:

```bash
cd third_party/metal/python
python run_tests.py
```

Test results and performance comparisons will be saved in the `test_results/` directory.

## Performance Considerations

- Most functions are optimized for both accuracy and performance
- For very large arrays, MLX's hardware acceleration provides significant speedups
- Some functions offer fast approximation variants with controlled error bounds

## Adding New Functions

To add a new special function:

1. Implement the function in the appropriate class in `special_ops.py`
2. Add it to the operations map in `get_special_ops_map()`
3. Add corresponding tests in `test_special_ops.py` 