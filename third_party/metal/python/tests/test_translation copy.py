#!/usr/bin/env python
"""
Test for the translated special_ops.py file.
This verifies that the English translation is correctly loadable and functional.
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Check if MLX is available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    print("Warning: MLX not found. Some tests will be skipped.")
    HAS_MLX = False

class TestSpecialOpsTranslation(unittest.TestCase):
    """Test the translation of special_ops.py"""
    
    def test_module_import(self):
        """Test that the translated module can be imported"""
        try:
            from MLX.special_ops import SpecialMathFunctions, NumericalFunctions, get_special_ops_map
            self.assertTrue(True, "Module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import translated module: {e}")
    
    @unittest.skipIf(not HAS_MLX, "MLX not available")
    def test_basic_function(self):
        """Test a basic function from the translated module"""
        try:
            from MLX.special_ops import SpecialMathFunctions
            
            # Create an instance
            special_math = SpecialMathFunctions()
            
            # Create a simple test input
            x = mx.array([0.0, 0.5, 1.0])
            
            # Call a simple function
            result = special_math.erf(x)
            
            # Verify that result is an MLX array
            self.assertEqual(type(result), type(x))
            
            # Verify that the shape is correct
            self.assertEqual(result.shape, x.shape)
            
            print(f"Result of erf function: {result}")
            
        except Exception as e:
            self.fail(f"Error testing basic function: {e}")
    
    def test_function_mapping(self):
        """Test the operation mapping from the translated module"""
        try:
            from MLX.special_ops import get_special_ops_map
            
            # Get the operation map
            op_map = get_special_ops_map()
            
            # Verify that it's a dictionary
            self.assertIsInstance(op_map, dict)
            
            # Verify that it has the expected operations
            expected_ops = [
                'tt.erf', 'tt.erfc', 'tt.gamma', 'tt.lgamma', 
                'tt.bessel_j0', 'tt.fast_sigmoid'
            ]
            
            for op in expected_ops:
                self.assertIn(op, op_map, f"Operation {op} not found in mapping")
                self.assertTrue(callable(op_map[op]), f"Operation {op} is not callable")
                
        except Exception as e:
            self.fail(f"Error testing function mapping: {e}")

def main():
    unittest.main()

if __name__ == "__main__":
    main() 