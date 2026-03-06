#!/usr/bin/env python
"""
Unit tests for the system compatibility check script.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import check_system
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import tools.check_system as check_system

class TestCheckSystem(unittest.TestCase):
    """Test cases for the check_system.py script"""
    
    @patch('platform.mac_ver')
    def test_check_macos_version(self, mock_mac_ver):
        """Test macOS version checking"""
        # Test compatible version
        mock_mac_ver.return_value = ('13.5.0', '', '')
        is_compatible, details = check_system.check_macos_version()
        self.assertTrue(is_compatible)
        self.assertIn('13.5.0', details)
        
        # Test newer compatible version
        mock_mac_ver.return_value = ('14.0.0', '', '')
        is_compatible, details = check_system.check_macos_version()
        self.assertTrue(is_compatible)
        self.assertIn('14.0.0', details)
        
        # Test incompatible version
        mock_mac_ver.return_value = ('13.4.0', '', '')
        is_compatible, details = check_system.check_macos_version()
        self.assertFalse(is_compatible)
        self.assertIn('required: 13.5+', details)
    
    @patch('platform.machine')
    def test_check_apple_silicon(self, mock_machine):
        """Test Apple Silicon detection"""
        # Test with Apple Silicon
        mock_machine.return_value = 'arm64'
        is_silicon, details = check_system.check_apple_silicon()
        self.assertTrue(is_silicon)
        self.assertIn('Apple Silicon detected', details)
        
        # Test with Intel
        mock_machine.return_value = 'x86_64'
        is_silicon, details = check_system.check_apple_silicon()
        self.assertFalse(is_silicon)
        self.assertIn('Intel CPU detected', details)
    
    @patch('check_system.importlib.util.find_spec')
    @patch('check_system.importlib.import_module')
    def test_check_package_installed(self, mock_import, mock_find_spec):
        """Test package installation checking"""
        # Test installed package with version
        mock_find_spec.return_value = MagicMock()
        mock_module = MagicMock()
        mock_module.__version__ = '1.0.0'
        mock_import.return_value = mock_module
        
        is_installed, details = check_system.check_package_installed('test_package')
        self.assertTrue(is_installed)
        self.assertIn('1.0.0', details)
        
        # Test not installed package
        mock_find_spec.return_value = None
        is_installed, details = check_system.check_package_installed('missing_package')
        self.assertFalse(is_installed)
        self.assertIn('Not installed', details)
    
    def test_m_series_generation_mocked(self):
        """Test M-series generation detection with mocks"""
        # Create a mock AppleSiliconGeneration Enum
        class MockAppleSiliconGeneration(MagicMock):
            M1 = MagicMock()
            M1.value = 1
            M2 = MagicMock()
            M2.value = 2
            M3 = MagicMock()
            M3.value = 3
        
        # Create a mock for hardware capabilities
        mock_hardware_capabilities = MagicMock()
        
        # Test for M3 chip
        mock_hardware_capabilities.chip_generation = MagicMock()
        mock_hardware_capabilities.chip_generation.name = "M3"
        mock_hardware_capabilities.chip_generation.value = 3
        
        with patch.dict('sys.modules', {
            'metal_hardware_optimizer': MagicMock(
                hardware_capabilities=mock_hardware_capabilities,
                AppleSiliconGeneration=MockAppleSiliconGeneration
            )
        }):
            has_gen, gen_name, details = check_system.check_m_series_generation()
            self.assertTrue(has_gen)
            self.assertEqual(gen_name, "M3")
            self.assertIn("Detected M3", details)
        
        # Test for M1 chip (older)
        mock_hardware_capabilities.chip_generation.name = "M1"
        mock_hardware_capabilities.chip_generation.value = 1
        
        with patch.dict('sys.modules', {
            'metal_hardware_optimizer': MagicMock(
                hardware_capabilities=mock_hardware_capabilities,
                AppleSiliconGeneration=MockAppleSiliconGeneration
            )
        }):
            has_gen, gen_name, details = check_system.check_m_series_generation()
            self.assertTrue(has_gen)
            self.assertEqual(gen_name, "M1")
            self.assertIn("M3 or newer recommended", details)
    
    def test_m3_specific_optimizations_mocked(self):
        """Test M3-specific optimizations detection with mocks"""
        # Create a mock AppleSiliconGeneration Enum
        class MockAppleSiliconGeneration(MagicMock):
            M1 = MagicMock()
            M2 = MagicMock()
            M3 = MagicMock()
        
        # Create a mock for hardware capabilities with M3 optimizations
        mock_hardware_capabilities = MagicMock()
        mock_hardware_capabilities.__dir__ = lambda self: ['get_m3_optimal_settings', 'other_function']
        
        with patch.dict('sys.modules', {
            'metal_hardware_optimizer': MagicMock(
                hardware_capabilities=mock_hardware_capabilities,
                AppleSiliconGeneration=MockAppleSiliconGeneration
            )
        }):
            has_optimizations, details = check_system.check_m3_specific_optimizations()
            self.assertTrue(has_optimizations)
            self.assertIn("get_m3_optimal_settings", details)
        
        # Test without M3 optimizations
        mock_hardware_capabilities.__dir__ = lambda self: ['some_function', 'other_function']
        
        with patch.dict('sys.modules', {
            'metal_hardware_optimizer': MagicMock(
                hardware_capabilities=mock_hardware_capabilities,
                AppleSiliconGeneration=MockAppleSiliconGeneration
            )
        }):
            has_optimizations, details = check_system.check_m3_specific_optimizations()
            self.assertTrue(has_optimizations)
            self.assertIn("but no specific optimizations found", details)
        
        # Test without M3 enum
        class MockAppleSiliconGenerationNoM3(MagicMock):
            M1 = MagicMock()
            M2 = MagicMock()
        
        with patch.dict('sys.modules', {
            'metal_hardware_optimizer': MagicMock(
                hardware_capabilities=mock_hardware_capabilities,
                AppleSiliconGeneration=MockAppleSiliconGenerationNoM3
            )
        }):
            has_optimizations, details = check_system.check_m3_specific_optimizations()
            self.assertFalse(has_optimizations)
            self.assertIn("No M3-specific support detected", details)
    
    @patch('platform.mac_ver')
    def test_get_troubleshooting_tips(self, mock_mac_ver):
        """Test generation of troubleshooting tips"""
        # Set up mock for mac_ver used in troubleshooting tips
        mock_mac_ver.return_value = ('13.4.0', '', '')
        
        # Test with empty failed checks
        tips = check_system.get_troubleshooting_tips({})
        self.assertEqual(len(tips), 0)
        
        # Test with macOS version failure
        tips = check_system.get_troubleshooting_tips({"macOS 13.5+ required": "Found macOS 13.4.0 (required: 13.5+)"})
        self.assertEqual(len(tips), 1)
        self.assertIn("Update your macOS", tips[0])
        self.assertIn("13.4.0", tips[0])
        
        # Test with MLX missing
        tips = check_system.get_troubleshooting_tips({"MLX installed": "Not installed"})
        self.assertEqual(len(tips), 1)
        self.assertIn("pip install mlx", tips[0])
        
        # Test with multiple failures
        failed_checks = {
            "macOS 13.5+ required": "Found macOS 13.4.0 (required: 13.5+)",
            "MLX installed": "Not installed",
            "Metal hardware detection": "Metal hardware detection module not found",
            "COALESCED memory layout defined": "metal_memory_manager module not found"
        }
        tips = check_system.get_troubleshooting_tips(failed_checks)
        self.assertTrue(len(tips) > 3)  # Should have at least 4 specific tips plus the general reinstall tip
        
        # Check for specific recommendations
        has_macos_tip = any("Update your macOS" in tip for tip in tips)
        has_mlx_tip = any("pip install mlx" in tip for tip in tips)
        has_general_tip = any("reinstalling the Triton Metal backend" in tip for tip in tips)
        
        self.assertTrue(has_macos_tip)
        self.assertTrue(has_mlx_tip)
        self.assertTrue(has_general_tip)

if __name__ == '__main__':
    unittest.main() 