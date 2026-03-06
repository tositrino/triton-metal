"""
Metal backend kernel launcher and compilation pipeline implementation
"""

import os
import tempfile
import time
import numpy as np

# Lazy import MLX to avoid unnecessary dependencies
_mx = None

def _get_mlx():
    """Lazy load MLX"""
    global _mx
    if _mx is None:
        import mlx.core as mx
        _mx = mx
    return _mx

# Import thread mapping tools
from .thread_mapping import map_kernel_launch_params

class MetalLauncher:
    """
    Metal backend kernel launcher
    Responsible for executing kernels from the Metal library compiled by Triton
    """

    def __init__(self, metallib_binary, metadata, options):
        """
        Initialize Metal launcher

        Args:
            metallib_binary: Compiled Metal library binary data
            metadata: Kernel metadata
            options: Compilation options
        """
        self.metadata = metadata
        self.options = options
        self.mx = _get_mlx()

        # Load function from metallib binary data
        self.kernel_fn = self._load_metal_function(metallib_binary)

        # Cache performance counters
        self.perf_counters = {
            "total_calls": 0,
            "total_time": 0,
            "last_call_time": 0
        }

    def _load_metal_function(self, metallib_binary):
        """
        Load kernel function from Metal library binary data

        Args:
            metallib_binary: Compiled Metal library binary data

        Returns:
            Loaded Metal function
        """
        # Save binary data to temporary file
        with tempfile.NamedTemporaryFile(suffix='.metallib', delete=False) as f:
            f.write(metallib_binary)
            metallib_path = f.name

        try:
            # Use MLX's Metal API to load the library
            kernel_name = self.metadata.get("kernel_name", "kernel_main")

            # Check if MLX supports direct loading of Metal libraries
            if hasattr(self.mx.metal, "load_metallib"):
                metal_fn = self.mx.metal.load_metallib(metallib_path, kernel_name)
            else:
                # Alternative: Use MLX's compute graph as function
                # This is a simplified implementation, would be more complex in reality
                metal_fn = self._create_mlx_wrapper()

            return metal_fn
        finally:
            # Clean up temporary file
            os.unlink(metallib_path)

    def _create_mlx_wrapper(self):
        """Create MLX function as a wrapper for Metal library"""
        # Extract function signature information
        arg_types = self.metadata.get("arg_types", [])

        # Create wrapper function
        def wrapper(*args, **kwargs):
            # Convert inputs to MLX arrays
            mlx_args = []
            for i, arg in enumerate(args):
                if isinstance(arg, (np.ndarray, list, tuple)) and i < len(arg_types):
                    mlx_args.append(self.mx.array(arg))
                else:
                    mlx_args.append(arg)

            # Create MLX computation
            # In a real implementation, this would directly call the Metal function
            # This is a placeholder implementation
            result = sum(a for a in mlx_args if isinstance(a, type(self.mx.array(0))))

            return result

        return wrapper

    def __call__(self, *args, grid=None, **kwargs):
        """
        Execute kernel

        Args:
            *args: Kernel parameters
            grid: Grid configuration, e.g., {"grid": (16, 16, 1), "block": (32, 32, 1)}
            **kwargs: Other keyword arguments

        Returns:
            Execution result
        """
        # Record start time
        start_time = time.time()

        # Map launch parameters
        if grid is not None:
            metal_params = map_kernel_launch_params(grid)

            # Add additional info from metadata
            if "shared_mem_bytes" in self.metadata:
                metal_params["shared_memory_size"] = self.metadata["shared_mem_bytes"]
        else:
            # Use default parameters
            metal_params = {
                "grid_size": (1, 1, 1),
                "threadgroup_size": (1, 1, 1),
                "shared_memory_size": 0
            }

        # Convert inputs to MLX arrays
        mlx_args = []
        for arg in args:
            if isinstance(arg, (int, float, bool)):
                mlx_args.append(arg)  # Pass scalars directly
            elif isinstance(arg, (np.ndarray, list, tuple)):
                # Convert to MLX array
                mlx_args.append(self.mx.array(arg))
            else:
                # Assume it's already an MLX array or other acceptable type
                mlx_args.append(arg)

        # Execute computation
        try:
            # Apply any special processing from metadata

            # Call kernel function
            result = self.kernel_fn(*mlx_args)

            # Ensure computation completes (synchronous execution)
            self.mx.eval(result)

            # Update performance counters
            self.perf_counters["total_calls"] += 1
            self.perf_counters["last_call_time"] = time.time() - start_time
            self.perf_counters["total_time"] += self.perf_counters["last_call_time"]

            return result
        except Exception as e:
            # Log error
            print(f"Metal kernel execution failed: {e}")
            raise

    def get_performance_stats(self):
        """Get performance statistics"""
        stats = dict(self.perf_counters)

        # Calculate average execution time
        if stats["total_calls"] > 0:
            stats["avg_time"] = stats["total_time"] / stats["total_calls"]
        else:
            stats["avg_time"] = 0

        return stats

class MetalCompiler:
    """Metal kernel compiler"""

    def __init__(self):
        self.mx = _get_mlx()
        self.cache_dir = os.path.expanduser("~/.triton/metal_cache")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def compile_mlx_to_metal(self, mlx_graph, metadata, options):
        """
        Compile MLX compute graph to Metal library

        Args:
            mlx_graph: MLX compute graph
            metadata: Kernel metadata
            options: Compilation options

        Returns:
            Compiled Metal library binary data
        """
        # Generate cache key
        cache_key = self._generate_cache_key(mlx_graph, metadata, options)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.metallib")

        # Check cache
        if os.path.exists(cache_path) and options.get("use_cache", True):
            with open(cache_path, 'rb') as f:
                return f.read()

        # Compile MLX graph to Metal
        # This part requires MLX support, currently a simplified implementation
        if hasattr(self.mx.metal, "compile_to_metallib"):
            # If MLX directly supports export to metallib
            with tempfile.NamedTemporaryFile(suffix='.metallib', delete=False) as f:
                metallib_path = f.name

            # Compile to metallib
            self.mx.metal.compile_to_metallib(mlx_graph, metallib_path)

            # Read compilation result
            with open(metallib_path, 'rb') as f:
                metallib_binary = f.read()

            # Clean up temporary file
            os.unlink(metallib_path)

            # Cache result
            with open(cache_path, 'wb') as f:
                f.write(metallib_binary)

            return metallib_binary
        else:
            # Not directly supported currently, return a placeholder result
            print("Warning: MLX currently doesn't support direct export to Metal library, returning placeholder binary data")
            placeholder = b'METAL_BINARY_PLACEHOLDER'

            # Cache placeholder result
            with open(cache_path, 'wb') as f:
                f.write(placeholder)

            return placeholder

    def _generate_cache_key(self, mlx_graph, metadata, options):
        """Generate cache key"""
        import hashlib

        # Create a string representation
        key_parts = []

        # Add graph ID
        key_parts.append(f"graph_id={id(mlx_graph)}")

        # Add metadata
        for k, v in sorted(metadata.items()):
            key_parts.append(f"{k}={v}")

        # Add options
        for k, v in sorted(options.items()):
            key_parts.append(f"{k}={v}")

        # Add MLX version
        mx_version = getattr(self.mx, "__version__", "unknown")
        key_parts.append(f"mlx_version={mx_version}")

        # Calculate hash
        key_str = ";".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def jit_compile(self, fn, example_inputs, metadata=None, options=None):
        """
        JIT compile Python function to Metal

        Args:
            fn: Python function to compile
            example_inputs: Example inputs for type and shape inference
            metadata: Additional metadata
            options: Compilation options

        Returns:
            Compiled Metal launcher
        """
        # Default values
        metadata = metadata or {}
        options = options or {}

        # Trace function to create MLX compute graph
        if hasattr(self.mx, "compile"):
            mlx_fn = self.mx.compile(fn)
        else:
            # If MLX doesn't have compile functionality, use simple wrapper
            mlx_fn = fn

        # Prepare MLX inputs
        mlx_inputs = []
        for inp in example_inputs:
            if not isinstance(inp, type(self.mx.array(0))):
                mlx_inputs.append(self.mx.array(inp))
            else:
                mlx_inputs.append(inp)

        # Run function once to get compute graph
        result = mlx_fn(*mlx_inputs)

        # Compile MLX graph to Metal
        metallib_binary = self.compile_mlx_to_metal(mlx_fn, metadata, options)

        # Create launcher
        return MetalLauncher(metallib_binary, metadata, options)

# Create global instance
metal_compiler = MetalCompiler()

def compile_and_launch(fn, *example_inputs, grid=None, metadata=None, options=None):
    """
    Convenience function to compile and launch kernel

    Args:
        fn: Python function to compile
        *example_inputs: Example inputs for type and shape inference
        grid: Grid configuration
        metadata: Additional metadata
        options: Compilation options

    Returns:
        Compiled Metal launcher
    """
    # Add grid info to metadata
    metadata = metadata or {}
    if grid is not None:
        metadata["grid"] = grid

    # Compile
    launcher = metal_compiler.jit_compile(fn, example_inputs, metadata, options)

    return launcher