#!/usr/bin/env python
"""
Unit tests for the Metal Backend Compatibility Tutorial.

This test suite verifies the functionality of the tutorial_metal_compatibility.py script
that checks system compatibility with the Triton Metal backend.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import importlib
import platform
from enum import Enum  # Add missing import for Enum

# Add parent directory to path to import tutorial module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import functions from the tutorial for testing
import .tutorial_metal_compatibility as tutorial

class TestTutorialMetalCompatibility(unittest.TestCase):
    """Test cases for tutorial_metal_compatibility.py"""
    
    def test_print_functions(self):
        """Test the print formatting helper functions"""
        # These are mostly visual, so we just make sure they don't crash
        tutorial.print_section("Test Section")
        tutorial.print_status("Test Status", True, "Test Details")
        tutorial.print_status("Test Status", False, "Test Details")
        tutorial.print_code_example("print('Hello World')")
    
    @patch('platform.mac_ver')
    def test_check_macos_version(self, mock_mac_ver):
        """Test macOS version check"""
        # Test compatible version
        mock_mac_ver.return_value = ('13.5.0', '', '')
        is_compatible, details = tutorial.check_macos_version()
        self.assertTrue(is_compatible)
        self.assertIn('13.5.0', details)
        
        # Test newer compatible version
        mock_mac_ver.return_value = ('14.0.0', '', '')
        is_compatible, details = tutorial.check_macos_version()
        self.assertTrue(is_compatible)
        self.assertIn('14.0.0', details)
        
        # Test incompatible version
        mock_mac_ver.return_value = ('13.4.0', '', '')
        is_compatible, details = tutorial.check_macos_version()
        self.assertFalse(is_compatible)
        self.assertIn('required: 13.5+', details)
        
        # Test exception handling
        mock_mac_ver.return_value = ('invalid', '', '')
        is_compatible, details = tutorial.check_macos_version()
        self.assertFalse(is_compatible)
        self.assertIn('Could not determine', details)
    
    @patch('platform.machine')
    def test_check_apple_silicon(self, mock_machine):
        """Test Apple Silicon detection"""
        # Test with Apple Silicon
        mock_machine.return_value = 'arm64'
        is_silicon, details = tutorial.check_apple_silicon()
        self.assertTrue(is_silicon)
        self.assertIn('Apple Silicon detected', details)
        
        # Test with Intel
        mock_machine.return_value = 'x86_64'
        is_silicon, details = tutorial.check_apple_silicon()
        self.assertFalse(is_silicon)
        self.assertIn('Intel CPU detected', details)
    
    @patch('tutorial_metal_compatibility.importlib.util.find_spec')
    @patch('tutorial_metal_compatibility.importlib.import_module')
    def test_check_package_installed(self, mock_import, mock_find_spec):
        """Test package installation checking"""
        # Test installed package with version
        mock_find_spec.return_value = MagicMock()
        mock_module = MagicMock()
        mock_module.__version__ = '1.0.0'
        mock_import.return_value = mock_module
        
        is_installed, details = tutorial.check_package_installed('test_package')
        self.assertTrue(is_installed)
        self.assertIn('1.0.0', details)
        
        # Test not installed package
        mock_find_spec.return_value = None
        is_installed, details = tutorial.check_package_installed('missing_package')
        self.assertFalse(is_installed)
        self.assertIn('Not installed', details)
    
    def test_m_series_generation_mocked(self):
        """Test M-series generation detection with mocks"""
        # Create a mock AppleSiliconGeneration Enum
        class MockAppleSiliconGeneration(Enum):
            M1 = 1
            M2 = 2
            M3 = 3
        
        # Create a mock for hardware capabilities
        mock_hardware_capabilities = MagicMock()
        # For the M3 chip test
        mock_chip_gen = MagicMock()
        mock_chip_gen.name = "M3"
        mock_chip_gen.value = 3
        mock_hardware_capabilities.chip_generation = mock_chip_gen
        
        with patch.dict('sys.modules', {
            'metal_hardware_optimizer': MagicMock(
                hardware_capabilities=mock_hardware_capabilities,
                AppleSiliconGeneration=MockAppleSiliconGeneration
            )
        }):
            has_gen, gen_name, details = tutorial.check_m_series_generation()
            self.assertTrue(has_gen)
            self.assertEqual(gen_name, "M3")
            self.assertIn("Detected M3", details)
        
        # Test for M1 chip (older)
        mock_chip_gen.name = "M1"
        mock_chip_gen.value = 1
        mock_hardware_capabilities.chip_generation = mock_chip_gen
        
        with patch.dict('sys.modules', {
            'metal_hardware_optimizer': MagicMock(
                hardware_capabilities=mock_hardware_capabilities,
                AppleSiliconGeneration=MockAppleSiliconGeneration
            )
        }):
            has_gen, gen_name, details = tutorial.check_m_series_generation()
            self.assertTrue(has_gen)
            self.assertEqual(gen_name, "M1")
            self.assertIn("M3 or newer recommended", details)
    
    def test_m3_specific_optimizations_mocked(self):
        """Test M3-specific optimizations detection with mocks"""
        # Create a mock for M3Feature enum
        class MockM3Feature(Enum):
            DYNAMIC_CACHING = 0
            ENHANCED_MATRIX_COPROCESSOR = 1
        
        # Create a mock for M3Optimizer
        class MockM3Optimizer:
            def __init__(self):
                self.is_m3 = True
                
            def is_feature_available(self, feature):
                return True
        
        with patch.dict('sys.modules', {
            'm3_optimizations': MagicMock(
                M3Optimizer=MockM3Optimizer,
                M3Feature=MockM3Feature
            )
        }):
            is_m3, features = tutorial.check_m3_specific_optimizations()
            self.assertTrue(is_m3)
            self.assertEqual(len(features), 2)  # Two mock features
    
    def test_coalesced_memory_layout_mocked(self):
        """Test COALESCED memory layout detection with mocks"""
        # Create a mock MemoryLayout Enum
        class MockMemoryLayout(Enum):
            DEFAULT = 0
            COALESCED = 1
        
        with patch.dict('sys.modules', {
            'metal_memory_manager': MagicMock(
                MemoryLayout=MockMemoryLayout
            )
        }):
            has_coalesced, details = tutorial.check_coalesced_memory_layout()
            self.assertTrue(has_coalesced)
            self.assertIn("COALESCED layout defined with value 1", details)
    
    def test_optimizer_consistency_mocked(self):
        """Test optimizer consistency check with mocks"""
        # Create mock MemoryLayout Enums
        class MockManagerLayout(Enum):
            DEFAULT = 0
            COALESCED = 1
        
        class MockOptimizerLayout(Enum):
            DEFAULT = 0
            COALESCED = 1
        
        with patch.dict('sys.modules', {
            'metal_memory_manager': MagicMock(
                MemoryLayout=MockManagerLayout
            ),
            'memory_layout_optimizer': MagicMock(
                MemoryLayout=MockOptimizerLayout
            )
        }):
            is_consistent, details = tutorial.check_optimizer_consistency()
            self.assertTrue(is_consistent)
            self.assertIn("Consistent definition", details)
        
        # Test with inconsistent values
        class MockOptimizerLayoutInconsistent(Enum):
            DEFAULT = 0
            COALESCED = 2  # Different value
        
        with patch.dict('sys.modules', {
            'metal_memory_manager': MagicMock(
                MemoryLayout=MockManagerLayout
            ),
            'memory_layout_optimizer': MagicMock(
                MemoryLayout=MockOptimizerLayoutInconsistent
            )
        }):
            is_consistent, details = tutorial.check_optimizer_consistency()
            self.assertFalse(is_consistent)
            self.assertIn("Inconsistent values", details)
    
    def test_get_troubleshooting_tips(self):
        """Test troubleshooting tips generation"""
        # Mock generation name to test both branches
        with patch('tutorial_metal_compatibility.generation_name', None):
            tips = tutorial.get_troubleshooting_tips()
            self.assertTrue(any('For optimal performance' in tip for tip in tips))
        
        # Second test with M3 generation name
        with patch('tutorial_metal_compatibility.generation_name', 'M3'):
            with patch('tutorial_metal_compatibility.has_generation', True):
                tips = tutorial.get_troubleshooting_tips()
                self.assertFalse(any('For optimal performance' in tip for tip in tips))
    
    def test_full_script_execution(self):
        """Test that the full script can execute without errors"""
        # This test just ensures the script doesn't raise exceptions
        # We'll use a number of mocks to control its behavior
        
        # Setup all our mocks for a successful run
        patches = [
            patch('platform.mac_ver', return_value=('13.5.0', '', '')),
            patch('platform.machine', return_value='arm64'),
            patch('subprocess.check_output', return_value=b'Apple M3'),
            patch('tutorial_metal_compatibility.check_package_installed', return_value=(True, 'Version 1.0.0')),
            patch('tutorial_metal_compatibility.check_m_series_generation', 
                 return_value=(True, 'M3', 'Detected M3')),
            patch('tutorial_metal_compatibility.check_m3_specific_optimizations',
                 return_value=(True, ['DYNAMIC_CACHING', 'ENHANCED_MATRIX_COPROCESSOR'])),
            patch('tutorial_metal_compatibility.check_coalesced_memory_layout',
                 return_value=(True, 'COALESCED layout defined')),
            patch('tutorial_metal_compatibility.check_optimizer_consistency',
                 return_value=(True, 'Consistent definition'))
        ]
        
        # Apply all patches
        for p in patches:
            p.start()
        
        try:
            # Call the main function
            # We need to modify the function to return rather than call sys.exit
            with patch('tutorial_metal_compatibility.run_tutorial', return_value=0):
                # Import the module again to run with our patches
                importlib.reload(tutorial)
        except Exception as e:
            self.fail(f"Full script execution raised an exception: {e}")
        finally:
            # Stop all patches
            for p in patches:
                p.stop()

if __name__ == '__main__':
    unittest.main() 