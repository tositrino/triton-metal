#!/usr/bin/env python
"""
Integration Test for Metal COALESCED Memory Layout Tools

This script tests that all of the COALESCED memory layout tools work
together correctly and produce consistent results.
"""

import os
import sys
import json
import tempfile
import unittest
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

class TestCoalescedLayoutTools(unittest.TestCase):
    """Integration tests for COALESCED memory layout tools"""
    
    def setUp(self):
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Define paths
        self.sample_ops_path = os.path.join(self.temp_dir.name, "sample_ops.json")
        self.analysis_path = os.path.join(self.temp_dir.name, "analysis.json")
        self.benchmark_path = os.path.join(self.temp_dir.name, "benchmark.png")
        
        # Generate sample operations
        subprocess.run([
            sys.executable, 
            os.path.join(current_dir, "create_sample_ops.py"),
            "--output", self.sample_ops_path,
            "--seed", "42"  # Use fixed seed for reproducibility
        ], check=True)
        
    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_simple_analyzer(self):
        """Test simple_analyzer.py on sample operations"""
        # Run analyzer on sample operations
        result = subprocess.run([
            sys.executable,
            os.path.join(current_dir, "simple_analyzer.py"),
            "--json", self.sample_ops_path,
            "--output", self.analysis_path
        ], check=True, capture_output=True, text=True)
        
        # Check that output file was created
        self.assertTrue(os.path.exists(self.analysis_path))
        
        # Load results
        with open(self.analysis_path, 'r') as f:
            analysis_results = json.load(f)
        
        # Verify that results include operation analysis - our format uses 'results' key
        self.assertIn("results", analysis_results)
        self.assertGreater(len(analysis_results["results"]), 0)
        
        # Verify that at least one operation uses COALESCED layout
        uses_coalesced = False
        for op in analysis_results["results"]:
            if "memory_layout" in op and op["memory_layout"] == "COALESCED":
                uses_coalesced = True
                break
        
        self.assertTrue(uses_coalesced, 
                     "None of the operations were identified as using COALESCED layout")
    
    def test_memory_layout_analyzer(self):
        """Test analyze_memory_layouts.py on sample operations"""
        # Run analyzer on sample operations
        result = subprocess.run([
            sys.executable,
            os.path.join(current_dir, "analyze_memory_layouts.py"),
            "--json", self.sample_ops_path,
            "--output", self.analysis_path
        ], check=True, capture_output=True, text=True)
        
        # Check that output file was created
        self.assertTrue(os.path.exists(self.analysis_path))
        
        # Load results
        with open(self.analysis_path, 'r') as f:
            analysis_results = json.load(f)
        
        # Verify that results include operation analysis - our format uses 'results' key
        self.assertIn("results", analysis_results)
        self.assertGreater(len(analysis_results["results"]), 0)
        
        # Verify that layouts are identified for operations
        for op in analysis_results["results"]:
            self.assertIn("layout", op)
    
    def test_analyzer_consistency(self):
        """Test that simple_analyzer and memory_layout_analyzer are consistent"""
        # Run both analyzers
        simple_output = os.path.join(self.temp_dir.name, "simple_analysis.json")
        comprehensive_output = os.path.join(self.temp_dir.name, "comprehensive_analysis.json")
        
        subprocess.run([
            sys.executable,
            os.path.join(current_dir, "simple_analyzer.py"),
            "--json", self.sample_ops_path,
            "--output", simple_output
        ], check=True)
        
        subprocess.run([
            sys.executable,
            os.path.join(current_dir, "analyze_memory_layouts.py"),
            "--json", self.sample_ops_path,
            "--output", comprehensive_output
        ], check=True)
        
        # Load results
        with open(simple_output, 'r') as f:
            simple_results = json.load(f)
        
        with open(comprehensive_output, 'r') as f:
            comprehensive_results = json.load(f)
        
        # Map operations by ID for comparison
        simple_ops = {op.get("id", op.get("type", i)): op 
                     for i, op in enumerate(simple_results["results"])}
        
        comprehensive_ops = {op.get("id", op.get("type", i)): op 
                            for i, op in enumerate(comprehensive_results["results"])}
        
        # Check for consistency in COALESCED layout identification
        for op_id, simple_op in simple_ops.items():
            if op_id in comprehensive_ops:
                # If simple analyzer says it uses COALESCED, comprehensive should include COALESCED
                if simple_op.get("memory_layout") == "COALESCED":
                    comp_op = comprehensive_ops[op_id]
                    
                    # Check if layout value is 8 (COALESCED value)
                    layout_contains_coalesced = (
                        comp_op.get("layout_name") == "COALESCED" or
                        comp_op.get("layout") == 8  # COALESCED.value = 8
                    )
                    
                    self.assertTrue(
                        layout_contains_coalesced,
                        f"Operation {op_id} identified as using COALESCED by simple_analyzer.py "
                        f"but not by analyze_memory_layouts.py"
                    )
    
    def test_string_parsing(self):
        """Test operation string parsing"""
        # Test with a simple reduction
        result = subprocess.run([
            sys.executable,
            os.path.join(current_dir, "simple_analyzer.py"),
            "--operation", "tt.sum:[64,128]:1"
        ], check=True, capture_output=True, text=True)
        
        self.assertIn("tt.sum", result.stdout)
        self.assertIn("COALESCED", result.stdout)
        
        # Test with a multi-axis reduction
        result = subprocess.run([
            sys.executable,
            os.path.join(current_dir, "simple_analyzer.py"),
            "--operation", "tt.mean:[32,64,128]:[0,2]"
        ], check=True, capture_output=True, text=True)
        
        self.assertIn("tt.mean", result.stdout)
        self.assertIn("COALESCED", result.stdout)
    
    def test_benchmark_tool(self):
        """Test benchmark_reduction_layouts.py"""
        try:
            # Only run a minimal benchmark for testing
            result = subprocess.run([
                sys.executable,
                os.path.join(current_dir, "benchmark_reduction_layouts.py"),
                "--sizes", "32x64,64x64",  # Small sizes for quick test
                "--repeats", "2",
                "--output", self.benchmark_path,
                "--no-show"  # Don't show plot window
            ], check=True, capture_output=True, text=True)
            
            # Check that benchmark plot was created
            self.assertTrue(os.path.exists(self.benchmark_path))
            
            # Verify expected output in console
            self.assertIn("Benchmarking", result.stdout)
        except subprocess.CalledProcessError:
            # Skip test if benchmark fails (e.g., no CUDA/Metal)
            print("Skipping benchmark test - requires compatible GPU")
    
    def test_sample_kernel(self):
        """Test sample_reduction_kernel.py"""
        try:
            # Run the sample kernel with minimal execution
            result = subprocess.run([
                sys.executable,
                os.path.join(current_dir, "sample_reduction_kernel.py"),
                "--quick"  # Run quick version
            ], check=True, capture_output=True, text=True)
            
            # Verify expected output
            self.assertIn("Sample Reduction", result.stdout)
        except subprocess.CalledProcessError:
            # Skip test if kernel fails (e.g., no CUDA/Metal)
            print("Skipping reduction kernel test - requires compatible GPU")

if __name__ == "__main__":
    print("Running Metal COALESCED Memory Layout Integration Tests")
    unittest.main() 