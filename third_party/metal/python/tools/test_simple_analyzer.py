#!/usr/bin/env python
"""
Test script for simple_analyzer.py

This script tests the parsing and analysis capabilities of simple_analyzer.py,
with a focus on multi-axis reduction operations and JSON serialization.
"""

import os
import sys
import json
import tempfile
import unittest

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import functions from simple_analyzer.py
from simple_analyzer import (
    parse_axis, 
    parse_shape, 
    parse_operation_string, 
    analyze_operation,
    serialize_results
)

class TestSimpleAnalyzer(unittest.TestCase):
    """Test cases for simple_analyzer.py"""
    
    def test_parse_axis_single(self):
        """Test parsing a single axis value"""
        # Test various formats of single axis
        self.assertEqual(parse_axis("0"), 0)
        self.assertEqual(parse_axis(" 1 "), 1)
        self.assertEqual(parse_axis("[2]"), 2)
        
        # Test negative axis
        self.assertEqual(parse_axis("-1"), -1)
        
        # Test invalid axis
        with self.assertRaises(ValueError):
            parse_axis("not_a_number")
    
    def test_parse_axis_multi(self):
        """Test parsing multi-axis values"""
        # Test various formats of multi-axis
        self.assertEqual(parse_axis("0,1"), [0, 1])
        self.assertEqual(parse_axis(" 0, 2 "), [0, 2])
        self.assertEqual(parse_axis("[0,1,2]"), [0, 1, 2])
        self.assertEqual(parse_axis(" [ 0 , 3 ] "), [0, 3])
        
        # Test with negative axes
        self.assertEqual(parse_axis("-1,0"), [-1, 0])
        
        # Test invalid multi-axis
        with self.assertRaises(ValueError):
            parse_axis("[0,invalid]")
    
    def test_parse_shape(self):
        """Test parsing shapes"""
        # Test various formats of shape
        self.assertEqual(parse_shape("32"), [32])
        self.assertEqual(parse_shape("32,64"), [32, 64])
        self.assertEqual(parse_shape("[32,64,128]"), [32, 64, 128])
        self.assertEqual(parse_shape(" [ 32 , 64 ] "), [32, 64])
        
        # Test invalid shape
        with self.assertRaises(ValueError):
            parse_shape("[32,invalid]")
    
    def test_parse_operation_string(self):
        """Test parsing operation strings"""
        # Test standard formats
        op1 = parse_operation_string("tt.reduce:[1024]:0")
        self.assertEqual(op1["type"], "tt.reduce")
        self.assertEqual(op1["input_shapes"][0], [1024])
        self.assertEqual(op1["args"]["axis"], 0)
        
        # Test multi-axis
        op2 = parse_operation_string("tt.mean:[32,64,128]:[0,1]")
        self.assertEqual(op2["type"], "tt.mean")
        self.assertEqual(op2["input_shapes"][0], [32, 64, 128])
        self.assertEqual(op2["args"]["axis"], [0, 1])
    
    def test_analyze_operation(self):
        """Test analyzing operations"""
        # Test a reduction operation
        op1 = {
            "type": "tt.reduce",
            "input_shapes": [[1024]],
            "args": {"axis": 0}
        }
        result1 = analyze_operation(op1)
        self.assertTrue(result1["is_reduction"])
        self.assertEqual(result1["memory_layout"], "COALESCED")
        
        # Test a non-reduction operation
        op2 = {
            "type": "tt.matmul",
            "input_shapes": [[512, 512], [512, 512]]
        }
        result2 = analyze_operation(op2)
        self.assertFalse(result2["is_reduction"])
        self.assertEqual(result2["memory_layout"], "DEFAULT")
        
        # Test a multi-axis reduction
        op3 = {
            "type": "tt.mean",
            "input_shapes": [[32, 64, 128]],
            "args": {"axis": [0, 1]}
        }
        result3 = analyze_operation(op3)
        self.assertTrue(result3["is_reduction"])
        self.assertEqual(result3["memory_layout"], "COALESCED")
        
        # Test a large reduction (should trigger two-stage reduction)
        op4 = {
            "type": "tt.sum",
            "input_shapes": [[2048]],
            "args": {"axis": 0}
        }
        result4 = analyze_operation(op4)
        self.assertTrue(result4["is_reduction"])
        self.assertEqual(result4["memory_layout"], "COALESCED")
        self.assertTrue(result4["optimizations"]["two_stage_reduction"])
    
    def test_serialize_results(self):
        """Test serializing results to JSON"""
        # Create a sample result with non-serializable objects
        from enum import Enum
        
        class TestEnum(Enum):
            TEST = 42
        
        results = [
            {
                "type": "tt.reduce",
                "input_shapes": [[1024]],
                "is_reduction": True,
                "enum_value": TestEnum.TEST,
                "nested_dict": {
                    "enum_value": TestEnum.TEST
                },
                "nested_list": [TestEnum.TEST, 1, 2]
            }
        ]
        
        # Serialize the results
        serialized = serialize_results(results)
        
        # Check if serialization worked correctly
        self.assertEqual(serialized[0]["enum_value"], 42)
        self.assertEqual(serialized[0]["nested_dict"]["enum_value"], 42)
        self.assertEqual(serialized[0]["nested_list"][0], 42)
        
        # Test that the serialized results can be properly JSON dumped
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            json.dump({"results": serialized}, f)
            temp_filename = f.name
        
        try:
            # Read back the JSON file
            with open(temp_filename, 'r') as f:
                data = json.load(f)
            
            # Check if the data was preserved
            self.assertEqual(data["results"][0]["type"], "tt.reduce")
            self.assertEqual(data["results"][0]["enum_value"], 42)
        finally:
            # Clean up
            os.unlink(temp_filename)

if __name__ == "__main__":
    unittest.main() 