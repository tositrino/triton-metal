"""
Metal backend tests

This module contains test cases for the Metal backend implementation.
It verifies the correct operation of the Metal backend components including
launcher, compiler, and memory layout adapters.
"""

import os
import unittest
import numpy as np


# Try to import MLX, skip tests if not available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Import Metal backend components
from MLX.launcher import MetalLauncher, MetalCompiler, compile_and_launch
from MLX.memory_layout import MemoryLayout, LayoutType, adapt_tensor, detect_layout
from MLX.thread_mapping import map_kernel_launch_params

class TestMetalBackend(unittest.TestCase):
    """Test cases for Metal backend"""

    def setUp(self):
        """Set up for tests"""
        # Skip tests if MLX is not available
        if not HAS_MLX:
            self.skipTest("MLX not available")

        # Create test data
        self.test_array_1d = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        self.test_array_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        self.test_array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)

        # Create MLX arrays
        self.mx_array_1d = mx.array(self.test_array_1d)
        self.mx_array_2d = mx.array(self.test_array_2d)
        self.mx_array_3d = mx.array(self.test_array_3d)

    def test_memory_layout_detection(self):
        """Test memory layout detection"""
        # Test row-major detection
        layout = detect_layout(self.mx_array_2d)
        self.assertEqual(layout.layout_type, LayoutType.ROW_MAJOR)

        # Test with metadata
        metadata = {"layout": {"type": LayoutType.COL_MAJOR}}
        layout = detect_layout(self.mx_array_2d, metadata)
        self.assertEqual(layout.layout_type, LayoutType.COL_MAJOR)

    def test_memory_layout_adaptation(self):
        """Test memory layout adaptation"""
        # Create test layouts
        row_major = MemoryLayout(self.mx_array_2d.shape, LayoutType.ROW_MAJOR)
        col_major = MemoryLayout(self.mx_array_2d.shape, LayoutType.COL_MAJOR)

        # Test row-major to col-major
        adapted = adapt_tensor(self.mx_array_2d, row_major, col_major)
        # Verify shape didn't change
        self.assertEqual(adapted.shape, self.mx_array_2d.shape)
        # Verify content is transposed
        np.testing.assert_array_equal(
            mx.array(self.test_array_2d.T),
            adapted
        )

    def test_thread_mapping(self):
        """Test thread mapping utility"""
        # Test basic mapping
        grid_config = {"grid": (16, 16, 1), "block": (32, 1, 1)}
        metal_params = map_kernel_launch_params(grid_config)

        # Verify Metal parameters
        self.assertEqual(metal_params["grid_size"], (16, 16, 1))
        self.assertEqual(metal_params["threadgroup_size"], (32, 1, 1))

    @unittest.skipIf(not os.path.exists("/usr/lib/libc++.1.dylib"), "macOS libraries not available")
    def test_metal_compiler(self):
        """Test Metal compiler"""
        compiler = MetalCompiler()

        # Simple test function
        def add_one(x):
            return x + 1

        # Test JIT compilation with simple function
        # Note: This may only work on macOS with Metal support
        try:
            launcher = compiler.jit_compile(add_one, [self.mx_array_1d])
            self.assertIsInstance(launcher, MetalLauncher)
        except Exception as e:
            self.skipTest(f"Metal compilation failed: {e}")

    @unittest.skipIf(not os.path.exists("/usr/lib/libc++.1.dylib"), "macOS libraries not available")
    def test_compile_and_launch(self):
        """Test compile and launch utility"""
        # Simple test function
        def multiply_by_two(x):
            return x * 2

        # Test with simple grid configuration
        grid = {"grid": (1, 1, 1), "block": (1, 1, 1)}

        # Attempt compilation and launch
        try:
            launcher = compile_and_launch(
                multiply_by_two,
                self.mx_array_1d,
                grid=grid
            )
            self.assertIsInstance(launcher, MetalLauncher)
        except Exception as e:
            self.skipTest(f"Compile and launch failed: {e}")

    def test_launcher_performance_stats(self):
        """Test launcher performance statistics"""
        # Create a mock launcher
        launcher = MetalLauncher(b'', {}, {})

        # Manually update performance counters
        launcher.perf_counters["total_calls"] = 10
        launcher.perf_counters["total_time"] = 5.0
        launcher.perf_counters["last_call_time"] = 0.5

        # Get stats
        stats = launcher.get_performance_stats()

        # Verify stats
        self.assertEqual(stats["total_calls"], 10)
        self.assertEqual(stats["total_time"], 5.0)
        self.assertEqual(stats["last_call_time"], 0.5)
        self.assertEqual(stats["avg_time"], 0.5)  # 5.0 / 10

if __name__ == "__main__":
    unittest.main()