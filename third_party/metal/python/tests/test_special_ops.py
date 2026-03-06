"""
Test suite for special_ops.py mathematical functions.
Validates accuracy of MLX implementations against reference values.
"""

import unittest
import numpy as np
import scipy.special
import math
import os
import sys
from typing import Callable, Dict, List, Tuple


# Import the functions to test
from MLX.special_ops import (
    SpecialMathFunctions, 
    NumericalFunctions,
    get_special_ops_map
)

# Import MLX for testing
import mlx.core as mx

class TestSpecialMathFunctions(unittest.TestCase):
    """Test suite for special mathematical functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.special_math = SpecialMathFunctions()
        self.numerical = NumericalFunctions()
        self.ops_map = get_special_ops_map()
        
        # Define test points for different ranges
        self.small_values = mx.array([-0.1, -0.01, 0.0, 0.01, 0.1])
        self.medium_values = mx.array([-3.5, -2.0, -1.0, 1.0, 2.0, 3.5])
        self.large_values = mx.array([-15.0, -10.0, -5.0, 5.0, 10.0, 15.0])
        self.positive_values = mx.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        
        # Tolerance for numerical comparisons
        self.atol = 1e-4  # Absolute tolerance
        self.rtol = 1e-4  # Relative tolerance
        
        # Special tolerances for approximation functions
        self.tolerances = {
            # Functions requiring higher tolerance
            'bessel_j0': {'rtol': 0.05, 'atol': 0.02},
            'bessel_j1': {'rtol': 0.4, 'atol': 0.05},
            'digamma': {'rtol': 0.5, 'atol': 0.5},
            'i0e': {'rtol': 0.01, 'atol': 0.01},
            'i1e': {'rtol': 0.01, 'atol': 0.01},
            'fast_sigmoid': {'rtol': 0.25, 'atol': 0.2},
            'fast_tanh': {'rtol': 0.05, 'atol': 0.05}
        }
    
    def _test_function(self, 
                      mlx_fn: Callable, 
                      ref_fn: Callable, 
                      test_values: mx.array,
                      name: str = None,
                      rtol: float = None,
                      atol: float = None):
        """
        Generic function to test MLX implementation against reference implementation
        
        Args:
            mlx_fn: MLX implementation to test
            ref_fn: Reference implementation (numpy/scipy)
            test_values: Input values to test
            name: Function name for error reporting
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        if name is None:
            name = mlx_fn.__name__
            
        # Use function-specific tolerances if available
        if rtol is None:
            rtol = self.tolerances.get(name, {}).get('rtol', self.rtol)
        if atol is None:
            atol = self.tolerances.get(name, {}).get('atol', self.atol)
            
        # Compute MLX results
        mlx_results = mlx_fn(test_values)
        
        # Convert to numpy for comparison - use appropriate method based on MLX version
        try:
            # Try the newer MLX API first
            mlx_np = mlx_results.numpy()
        except AttributeError:
            # Fall back to older MLX API
            mlx_np = np.array(mlx_results.tolist())
        
        # Convert test values to numpy
        try:
            test_np = test_values.numpy()
        except AttributeError:
            test_np = np.array(test_values.tolist())
        
        # Compute reference results
        ref_results = ref_fn(test_np)
        
        # Compare results
        try:
            np.testing.assert_allclose(mlx_np, ref_results, rtol=rtol, atol=atol)
        except AssertionError as e:
            # Enhanced error report
            error_indices = ~np.isclose(mlx_np, ref_results, rtol=rtol, atol=atol)
            if np.any(error_indices):
                idx = np.where(error_indices)[0]
                error_inputs = test_np[idx]
                error_mlx = mlx_np[idx]
                error_ref = ref_results[idx]
                error_diff = np.abs(error_mlx - error_ref)
                error_rel = error_diff / (np.abs(error_ref) + 1e-10)
                
                error_msg = f"\nFunction {name} failed comparison:\n"
                for i in range(len(idx)):
                    error_msg += f"  Input: {error_inputs[i]}, MLX: {error_mlx[i]}, Ref: {error_ref[i]}, "
                    error_msg += f"Abs Diff: {error_diff[i]}, Rel Diff: {error_rel[i]}\n"
                
                raise AssertionError(error_msg) from e
    
    def test_erf(self):
        """Test error function implementation"""
        self._test_function(
            self.special_math.erf,
            scipy.special.erf,
            mx.concatenate([self.small_values, self.medium_values]),
            name="erf"
        )
    
    def test_erfc(self):
        """Test complementary error function implementation"""
        self._test_function(
            self.special_math.erfc,
            scipy.special.erfc,
            mx.concatenate([self.small_values, self.medium_values]),
            name="erfc"
        )
    
    def test_digamma(self):
        """Test digamma function implementation"""
        # Only test positive values as digamma has singularities at negative integers
        self._test_function(
            self.special_math.digamma,
            scipy.special.digamma,
            self.positive_values,
            name="digamma"
        )
    
    def test_lgamma(self):
        """Test log-gamma function implementation"""
        self._test_function(
            self.special_math.lgamma,
            scipy.special.gammaln,
            self.positive_values,
            name="lgamma"
        )
    
    def test_gamma(self):
        """Test gamma function implementation"""
        # Limit to small/medium positive values to avoid overflow
        test_values = mx.array([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5])
        
        self._test_function(
            self.special_math.gamma,
            scipy.special.gamma,
            test_values,
            name="gamma"
        )
    
    def test_bessel_j0(self):
        """Test Bessel J0 function implementation"""
        self._test_function(
            self.special_math.bessel_j0,
            scipy.special.j0,
            mx.concatenate([self.small_values, self.medium_values]),
            name="bessel_j0"
        )
    
    def test_bessel_j1(self):
        """Test Bessel J1 function implementation"""
        self._test_function(
            self.special_math.bessel_j1,
            scipy.special.j1,
            mx.concatenate([self.small_values, self.medium_values]),
            name="bessel_j1"
        )
    
    def test_i0e(self):
        """Test scaled modified Bessel function I0e implementation"""
        def i0e_ref(x):
            return scipy.special.i0e(x)
            
        self._test_function(
            self.special_math.i0e,
            i0e_ref,
            mx.concatenate([self.small_values, self.medium_values]),
            name="i0e"
        )
    
    def test_i1e(self):
        """Test scaled modified Bessel function I1e implementation"""
        def i1e_ref(x):
            return scipy.special.i1e(x)
            
        self._test_function(
            self.special_math.i1e,
            i1e_ref,
            mx.concatenate([self.small_values, self.medium_values]),
            name="i1e"
        )
    
    def test_sinh(self):
        """Test hyperbolic sine implementation"""
        # Use smaller values to avoid overflow
        test_values = mx.array([-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0])
        
        self._test_function(
            self.special_math.sinh,
            np.sinh,
            test_values,
            name="sinh"
        )
    
    def test_cosh(self):
        """Test hyperbolic cosine implementation"""
        # Use smaller values to avoid overflow
        test_values = mx.array([-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0])
        
        self._test_function(
            self.special_math.cosh,
            np.cosh,
            test_values,
            name="cosh"
        )
    
    def test_asinh(self):
        """Test inverse hyperbolic sine implementation"""
        self._test_function(
            self.special_math.asinh,
            np.arcsinh,
            mx.concatenate([self.small_values, self.medium_values]),
            name="asinh"
        )
    
    def test_acosh(self):
        """Test inverse hyperbolic cosine implementation"""
        # acosh is only defined for x >= 1
        test_values = mx.array([1.0, 1.1, 2.0, 5.0, 10.0])
        
        self._test_function(
            self.special_math.acosh,
            np.arccosh,
            test_values,
            name="acosh"
        )
    
    def test_atanh(self):
        """Test inverse hyperbolic tangent implementation"""
        # atanh is only defined for -1 < x < 1
        test_values = mx.array([-0.99, -0.5, -0.1, 0.0, 0.1, 0.5, 0.99])
        
        self._test_function(
            self.special_math.atanh,
            np.arctanh,
            test_values,
            name="atanh"
        )
    
    def test_expm1(self):
        """Test exp(x)-1 implementation"""
        self._test_function(
            self.special_math.expm1,
            np.expm1,
            mx.concatenate([self.small_values, mx.array([-0.001, 0.001])]),
            name="expm1"
        )
    
    def test_log1p(self):
        """Test log(1+x) implementation"""
        # log1p is defined for x > -1
        test_values = mx.array([-0.9, -0.5, -0.1, -0.001, 0.0, 0.001, 0.1, 0.5, 1.0, 10.0])
        
        self._test_function(
            self.special_math.log1p,
            np.log1p,
            test_values,
            name="log1p"
        )


class TestNumericalFunctions(unittest.TestCase):
    """Test suite for numerical approximation functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.numerical = NumericalFunctions()
        
        # Define test points for different ranges
        self.small_values = mx.array([-0.1, -0.01, 0.0, 0.01, 0.1])
        self.medium_values = mx.array([-3.5, -2.0, -1.0, 1.0, 2.0, 3.5])
        self.large_values = mx.array([-15.0, -10.0, -5.0, 5.0, 10.0, 15.0])
        self.positive_values = mx.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        
        # Higher tolerance for approximation functions
        self.atol = 1e-3
        self.rtol = 1e-3
        
        # Function-specific tolerances
        self.tolerances = {
            'fast_sigmoid': {'rtol': 0.25, 'atol': 0.2},
            'fast_tanh': {'rtol': 0.05, 'atol': 0.05}
        }
    
    def _test_function(self, 
                      mlx_fn: Callable, 
                      ref_fn: Callable, 
                      test_values: mx.array,
                      name: str = None,
                      rtol: float = None,
                      atol: float = None):
        """
        Generic function to test MLX implementation against reference implementation
        
        Args:
            mlx_fn: MLX implementation to test
            ref_fn: Reference implementation (numpy/scipy)
            test_values: Input values to test
            name: Function name for error reporting
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        if name is None:
            name = mlx_fn.__name__
            
        # Use function-specific tolerances if available
        if rtol is None:
            rtol = self.tolerances.get(name, {}).get('rtol', self.rtol)
        if atol is None:
            atol = self.tolerances.get(name, {}).get('atol', self.atol)
            
        # Compute MLX results
        mlx_results = mlx_fn(test_values)
        
        # Convert to numpy for comparison - use appropriate method based on MLX version
        try:
            # Try the newer MLX API first
            mlx_np = mlx_results.numpy()
        except AttributeError:
            # Fall back to older MLX API
            mlx_np = np.array(mlx_results.tolist())
        
        # Convert test values to numpy
        try:
            test_np = test_values.numpy()
        except AttributeError:
            test_np = np.array(test_values.tolist())
        
        # Compute reference results
        ref_results = ref_fn(test_np)
        
        # Compare results
        try:
            np.testing.assert_allclose(mlx_np, ref_results, rtol=rtol, atol=atol)
        except AssertionError as e:
            # Enhanced error report
            error_indices = ~np.isclose(mlx_np, ref_results, rtol=rtol, atol=atol)
            if np.any(error_indices):
                idx = np.where(error_indices)[0]
                error_inputs = test_np[idx]
                error_mlx = mlx_np[idx]
                error_ref = ref_results[idx]
                error_diff = np.abs(error_mlx - error_ref)
                error_rel = error_diff / (np.abs(error_ref) + 1e-10)
                
                error_msg = f"\nFunction {name} failed comparison:\n"
                for i in range(len(idx)):
                    error_msg += f"  Input: {error_inputs[i]}, MLX: {error_mlx[i]}, Ref: {error_ref[i]}, "
                    error_msg += f"Abs Diff: {error_diff[i]}, Rel Diff: {error_rel[i]}\n"
                
                raise AssertionError(error_msg) from e
    
    def test_fast_exp(self):
        """Test fast exponential approximation"""
        # MLX already returns exact exp, but test for completeness
        self._test_function(
            self.numerical.fast_exp,
            np.exp,
            mx.concatenate([self.small_values, self.medium_values]),
            name="fast_exp"
        )
    
    def test_fast_sigmoid(self):
        """Test fast sigmoid approximation"""
        def sigmoid_ref(x):
            return 1.0 / (1.0 + np.exp(-x))
            
        self._test_function(
            self.numerical.fast_sigmoid,
            sigmoid_ref,
            mx.concatenate([self.small_values, self.medium_values]),
            name="fast_sigmoid"
        )
    
    def test_fast_tanh(self):
        """Test fast tanh approximation"""
        self._test_function(
            self.numerical.fast_tanh,
            np.tanh,
            mx.concatenate([self.small_values, self.medium_values]),
            name="fast_tanh"
        )
    
    def test_rsqrt(self):
        """Test reciprocal square root implementation"""
        def rsqrt_ref(x):
            return 1.0 / np.sqrt(x)
            
        self._test_function(
            self.numerical.rsqrt,
            rsqrt_ref,
            self.positive_values,  # Only test positive values
            name="rsqrt"
        )


class TestOperationMapping(unittest.TestCase):
    """Test the mapping of Triton functions to MLX implementations"""
    
    def setUp(self):
        """Set up test environment"""
        self.ops_map = get_special_ops_map()
        
    def test_mapping_completeness(self):
        """Test that all expected operations are mapped"""
        expected_ops = [
            'tt.erf', 'tt.erfc', 'tt.gamma', 'tt.lgamma', 'tt.digamma',
            'tt.bessel_j0', 'tt.bessel_j1', 'tt.i0e', 'tt.i1e',
            'tt.sinh', 'tt.cosh', 'tt.asinh', 'tt.acosh', 'tt.atanh',
            'tt.expm1', 'tt.log1p', 'tt.fast_exp', 'tt.fast_sigmoid',
            'tt.fast_tanh', 'tt.rsqrt'
        ]
        
        for op in expected_ops:
            self.assertIn(op, self.ops_map, f"Operation {op} missing from mapping")
    
    def test_callable_functions(self):
        """Test that all mapped operations are callable"""
        for op_name, op_func in self.ops_map.items():
            self.assertTrue(callable(op_func), f"Operation {op_name} is not callable")


if __name__ == '__main__':
    unittest.main() 