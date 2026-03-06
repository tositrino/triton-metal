#!/usr/bin/env python
import os
import time
import json
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Import the module to test
from MLX.metal_instrumentation import (
    MetalInstrumentation,
    ErrorDiagnostics,
    get_metal_instrumentation,
    get_error_diagnostics,
    PerformanceCounter,
    DebugInfo
)

class TestPerformanceCounter(unittest.TestCase):
    """Test the PerformanceCounter class"""

    def test_init(self):
        """Test initialization"""
        counter = PerformanceCounter("test_op")
        self.assertEqual(counter.name, "test_op")
        self.assertEqual(counter.count, 0)
        self.assertEqual(counter.total_time_ns, 0)
        self.assertEqual(counter.min_time_ns, 0)
        self.assertEqual(counter.max_time_ns, 0)
        self.assertEqual(counter.avg_time_ns, 0)

    def test_update(self):
        """Test update method"""
        counter = PerformanceCounter("test_op")

        # First update
        counter.update(100)
        self.assertEqual(counter.count, 1)
        self.assertEqual(counter.total_time_ns, 100)
        self.assertEqual(counter.min_time_ns, 100)
        self.assertEqual(counter.max_time_ns, 100)
        self.assertEqual(counter.avg_time_ns, 100)

        # Second update
        counter.update(200)
        self.assertEqual(counter.count, 2)
        self.assertEqual(counter.total_time_ns, 300)
        self.assertEqual(counter.min_time_ns, 100)
        self.assertEqual(counter.max_time_ns, 200)
        self.assertEqual(counter.avg_time_ns, 150)

        # Third update with smaller value
        counter.update(50)
        self.assertEqual(counter.count, 3)
        self.assertEqual(counter.total_time_ns, 350)
        self.assertEqual(counter.min_time_ns, 50)
        self.assertEqual(counter.max_time_ns, 200)
        self.assertEqual(counter.avg_time_ns, 116)  # 350/3 = 116.666 -> 116 (integer division)

    def test_to_dict(self):
        """Test to_dict method"""
        counter = PerformanceCounter("test_op")
        counter.update(1_000_000)  # 1ms
        counter.update(2_000_000)  # 2ms

        result = counter.to_dict()
        self.assertEqual(result["name"], "test_op")
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["total_time_ms"], 3.0)
        self.assertEqual(result["min_time_ms"], 1.0)
        self.assertEqual(result["max_time_ms"], 2.0)
        self.assertEqual(result["avg_time_ms"], 1.5)


class TestDebugInfo(unittest.TestCase):
    """Test the DebugInfo class"""

    def test_init(self):
        """Test initialization"""
        debug_info = DebugInfo("test_kernel", "test.py", 42)
        self.assertEqual(debug_info.kernel_name, "test_kernel")
        self.assertEqual(debug_info.source_file, "test.py")
        self.assertEqual(debug_info.line_number, 42)
        self.assertEqual(debug_info.variable_values, {})

    def test_init_with_values(self):
        """Test initialization with variable values"""
        values = {"x": 1, "y": 2.5}
        debug_info = DebugInfo("test_kernel", "test.py", 42, values)
        self.assertEqual(debug_info.variable_values, values)

    def test_to_dict(self):
        """Test to_dict method"""
        values = {"x": 1, "y": 2.5}
        debug_info = DebugInfo("test_kernel", "test.py", 42, values)

        result = debug_info.to_dict()
        self.assertEqual(result["kernel_name"], "test_kernel")
        self.assertEqual(result["source_file"], "test.py")
        self.assertEqual(result["line_number"], 42)
        self.assertEqual(result["variable_values"], values)


class TestMetalInstrumentation(unittest.TestCase):
    """Test the MetalInstrumentation class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()

        # Create instrumentation with debug level 3 (info)
        self.instrumentation = MetalInstrumentation(debug_level=3)
        self.instrumentation.output_dir = self.temp_dir

    def tearDown(self):
        """Tear down test fixtures"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_init_default(self):
        """Test initialization with default values"""
        instrumentation = MetalInstrumentation()
        self.assertEqual(instrumentation.debug_level, 0)
        self.assertEqual(instrumentation.performance_counters, {})
        self.assertEqual(instrumentation.debug_info_list, [])
        self.assertEqual(instrumentation.active_timers, {})

    def test_init_custom_level(self):
        """Test initialization with custom debug level"""
        instrumentation = MetalInstrumentation(debug_level=2)
        self.assertEqual(instrumentation.debug_level, 2)

    def test_timer_context_manager(self):
        """Test timer context manager"""
        # Use debug level > 0 to enable timing
        instrumentation = MetalInstrumentation(debug_level=1)

        # Mock time.time_ns to return controlled values
        original_time_ns = time.time_ns
        time_values = [1000, 3000]  # Start and end times
        time.time_ns = lambda: time_values.pop(0)

        try:
            # Use the timer
            with instrumentation.timer("test_op"):
                pass

            # Check that counter was created and updated
            self.assertIn("test_op", instrumentation.performance_counters)
            counter = instrumentation.performance_counters["test_op"]
            self.assertEqual(counter.count, 1)
            self.assertEqual(counter.total_time_ns, 2000)  # 3000 - 1000
            self.assertEqual(counter.min_time_ns, 2000)
            self.assertEqual(counter.max_time_ns, 2000)
            self.assertEqual(counter.avg_time_ns, 2000)
        finally:
            # Restore original time.time_ns
            time.time_ns = original_time_ns

    def test_record_debug_info(self):
        """Test record_debug_info method"""
        # Use debug level 3 (info) to enable debug info recording
        instrumentation = MetalInstrumentation(debug_level=3)

        # Record debug info
        instrumentation.record_debug_info("test_kernel", "test.py", 42, {"x": 1})

        # Check that debug info was recorded
        self.assertEqual(len(instrumentation.debug_info_list), 1)
        debug_info = instrumentation.debug_info_list[0]
        self.assertEqual(debug_info.kernel_name, "test_kernel")
        self.assertEqual(debug_info.source_file, "test.py")
        self.assertEqual(debug_info.line_number, 42)
        self.assertEqual(debug_info.variable_values, {"x": 1})

    def test_insert_debug_prints(self):
        """Test insert_debug_prints method"""
        kernel_src = "kernel void test() { int x = 1; }"

        # With debug level 0, it should return the original source
        instrumentation = MetalInstrumentation(debug_level=0)
        result = instrumentation.insert_debug_prints(kernel_src, "test")
        self.assertEqual(result, kernel_src)

        # With debug level 3, it should add debug prints
        instrumentation = MetalInstrumentation(debug_level=3)
        result = instrumentation.insert_debug_prints(kernel_src, "test")
        self.assertNotEqual(result, kernel_src)
        self.assertIn("Debug instrumentation", result)
        self.assertIn("<metal_stdlib>", result)
        self.assertIn("_metal_debug_print", result)
        self.assertIn(kernel_src, result)  # Original source should be included

    def test_generate_performance_report(self):
        """Test generate_performance_report method"""
        instrumentation = MetalInstrumentation(debug_level=2)

        # Add some performance counters
        counter1 = PerformanceCounter("op1")
        counter1.update(1_000_000)  # 1ms
        counter2 = PerformanceCounter("op2")
        counter2.update(2_000_000)  # 2ms
        counter2.update(3_000_000)  # 3ms

        instrumentation.performance_counters = {
            "op1": counter1,
            "op2": counter2,
        }

        # Generate report
        report = instrumentation.generate_performance_report()

        # Check report structure
        self.assertIn("summary", report)
        self.assertIn("counters", report)

        # Check summary
        summary = report["summary"]
        self.assertEqual(summary["total_operations"], 3)  # 1 + 2
        self.assertEqual(summary["total_time_ms"], 6.0)  # 1 + 2 + 3 = 6ms
        self.assertEqual(summary["total_counters"], 2)

        # Check counters
        counters = report["counters"]
        self.assertEqual(len(counters), 2)

        # Check that at least one file was created (debug_level >= 2)
        files = os.listdir(instrumentation.output_dir)
        self.assertGreater(len(files), 0)
        self.assertTrue(any(f.startswith("metal_perf_") for f in files))

    def test_generate_debug_report(self):
        """Test generate_debug_report method"""
        instrumentation = MetalInstrumentation(debug_level=3)

        # Add some debug info
        debug_info1 = DebugInfo("kernel1", "file1.py", 10, {"x": 1})
        debug_info2 = DebugInfo("kernel2", "file2.py", 20, {"y": 2})

        instrumentation.debug_info_list = [debug_info1, debug_info2]

        # Generate report
        report = instrumentation.generate_debug_report()

        # Check report structure
        self.assertIn("summary", report)
        self.assertIn("debug_info", report)

        # Check summary
        summary = report["summary"]
        self.assertEqual(summary["total_debug_records"], 2)

        # Check debug info
        debug_info = report["debug_info"]
        self.assertEqual(len(debug_info), 2)

        # Check that at least one file was created (debug_level >= 3)
        files = os.listdir(instrumentation.output_dir)
        self.assertGreater(len(files), 0)
        self.assertTrue(any(f.startswith("metal_debug_") for f in files))

    def test_clear(self):
        """Test clear method"""
        instrumentation = MetalInstrumentation(debug_level=1)

        # Add some data
        counter = PerformanceCounter("op")
        counter.update(1000)
        instrumentation.performance_counters = {"op": counter}

        debug_info = DebugInfo("kernel", "file.py", 10)
        instrumentation.debug_info_list = [debug_info]

        instrumentation.active_timers = {"timer1": 1000}

        # Clear data
        instrumentation.clear()

        # Check that data was cleared
        self.assertEqual(instrumentation.performance_counters, {})
        self.assertEqual(instrumentation.debug_info_list, [])
        self.assertEqual(instrumentation.active_timers, {})


class TestErrorDiagnostics(unittest.TestCase):
    """Test the ErrorDiagnostics class"""

    def setUp(self):
        """Set up test fixtures"""
        self.diagnostics = ErrorDiagnostics()

    def test_init(self):
        """Test initialization"""
        self.assertGreater(len(self.diagnostics.error_codes), 0)
        self.assertEqual(self.diagnostics.error_log, [])

    def test_diagnose_error_compilation_failed(self):
        """Test diagnose_error with compilation failure"""
        error_message = "Metal compilation failed: syntax error"
        code, desc, suggestions = self.diagnostics.diagnose_error(error_message, "test_kernel")

        self.assertEqual(code, "E001")
        self.assertEqual(desc, "Metal shader compilation failed")
        self.assertGreater(len(suggestions), 0)

        # Check that error was logged
        self.assertEqual(len(self.diagnostics.error_log), 1)
        log_entry = self.diagnostics.error_log[0]
        self.assertEqual(log_entry["error_code"], "E001")
        self.assertEqual(log_entry["kernel_name"], "test_kernel")

    def test_diagnose_error_out_of_memory(self):
        """Test diagnose_error with out of memory error"""
        error_message = "Operation failed: out of memory"
        code, desc, suggestions = self.diagnostics.diagnose_error(error_message)

        self.assertEqual(code, "E004")
        self.assertEqual(desc, "Memory allocation failed")
        self.assertGreater(len(suggestions), 0)

    def test_diagnose_error_generic(self):
        """Test diagnose_error with generic error"""
        error_message = "Unknown error occurred"
        code, desc, suggestions = self.diagnostics.diagnose_error(error_message)

        self.assertEqual(code, "E010")  # Default code
        self.assertEqual(desc, "Metal API error")
        self.assertEqual(suggestions, [])

    def test_log_error(self):
        """Test log_error method"""
        self.diagnostics.log_error("E001", "Test error", "test_kernel", "source code")

        self.assertEqual(len(self.diagnostics.error_log), 1)
        log_entry = self.diagnostics.error_log[0]
        self.assertEqual(log_entry["error_code"], "E001")
        self.assertEqual(log_entry["message"], "Test error")
        self.assertEqual(log_entry["kernel_name"], "test_kernel")
        self.assertIn("source_hash", log_entry)

    def test_get_error_summary(self):
        """Test get_error_summary method"""
        # Log some errors
        self.diagnostics.log_error("E001", "Error 1")
        self.diagnostics.log_error("E001", "Error 2")
        self.diagnostics.log_error("E002", "Error 3")

        # Get summary
        summary = self.diagnostics.get_error_summary()

        # Check summary structure
        self.assertEqual(summary["total_errors"], 3)
        self.assertEqual(summary["unique_error_codes"], 2)
        self.assertEqual(summary["error_counts"], {"E001": 2, "E002": 1})
        self.assertEqual(summary["most_recent"]["message"], "Error 3")


class TestGlobalInstances(unittest.TestCase):
    """Test the global instance getters"""

    @patch('metal_instrumentation.MetalInstrumentation')
    def test_get_metal_instrumentation(self, mock_class):
        """Test get_metal_instrumentation function"""
        # Reset global instance
        import metal_instrumentation
        metal_instrumentation._metal_instrumentation = None

        # Mock METAL_DEBUG_LEVEL environment variable
        with patch.dict('os.environ', {'METAL_DEBUG_LEVEL': '2'}):
            instance = get_metal_instrumentation()
            mock_class.assert_called_once_with(2)

        # Call again - should return the same instance
        instance2 = get_metal_instrumentation()
        self.assertEqual(mock_class.call_count, 1)  # Should not create a new instance

    @patch('metal_instrumentation.ErrorDiagnostics')
    def test_get_error_diagnostics(self, mock_class):
        """Test get_error_diagnostics function"""
        # Reset global instance
        import metal_instrumentation
        metal_instrumentation._error_diagnostics = None

        # Get instance
        instance = get_error_diagnostics()
        mock_class.assert_called_once()

        # Call again - should return the same instance
        instance2 = get_error_diagnostics()
        self.assertEqual(mock_class.call_count, 1)  # Should not create a new instance


if __name__ == "__main__":
    unittest.main()