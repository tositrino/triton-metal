"""Metal backend interface for Triton

This module provides the interface functions for the Metal backend
compilation stages, making the existing implementation available to
the Triton Python API integration.
"""


import tempfile
import pathlib
from typing import Dict, Any
from dataclasses import dataclass

# Import backend compiler implementation
from MLX.triton_to_metal_converter import TritonToMLXConverter

# Try to import instrumentation and optimizers
try:
    import MLX.metal_instrumentation
    has_instrumentation = True
except ImportError:
    has_instrumentation = False
    print("Warning: metal_instrumentation module not found. Debug and performance tracking will be disabled.")

try:
    import MLX.mlx_graph_optimizer
    import MLX.metal_memory_manager
    has_optimizers = True
except ImportError:
    has_optimizers = False
    print("Warning: MLX Graph Optimizer modules not available. Advanced optimizations will be disabled.")

try:
    import MLX.metal_hardware_optimizer
    hardware_capabilities = MLX.metal_hardware_optimizer.hardware_capabilities
    AppleSiliconGeneration = MLX.metal_hardware_optimizer.AppleSiliconGeneration
    has_hardware_optimizer = True
except ImportError:
    has_hardware_optimizer = False
    print("Warning: Metal hardware detection not available. Hardware-specific optimizations will be disabled.")

@dataclass
class MetalOptions:
    """Metal backend compilation options"""

    def __init__(self,
                num_warps: int = 4,
                num_ctas: int = 1,
                debug_info: bool = False,
                opt_level: int = 3,
                arch: str = "",
                max_shared_memory: int = 0,
                mlx_shard_size: int = 128,
                enable_fp_fusion: bool = True,
                enable_interleaving: bool = True,
                vectorize: bool = True,
                memory_optimization: str = "auto",
                fusion_optimization: str = "auto",
                metal_optimization_level: str = "auto",
                **kwargs):
        """
        Initialize Metal options

        Args:
            num_warps: Number of warps per threadgroup
            num_ctas: Number of concurrent thread groups
            debug_info: Enable debug info generation
            opt_level: Optimization level (0-3)
            arch: Target architecture (e.g., "m1", "m2")
            max_shared_memory: Maximum shared memory in bytes (0 = use default)
            mlx_shard_size: Shard size for MLX operations
            enable_fp_fusion: Enable fusion of floating-point operations
            enable_interleaving: Enable memory access interleaving
            vectorize: Enable vectorization
            memory_optimization: Memory optimization level ("none", "basic", "hardware_specific", "auto")
            fusion_optimization: Operation fusion level ("none", "basic", "hardware_specific", "auto")
            metal_optimization_level: Overall Metal optimization level ("none", "basic", "standard", "aggressive", "experimental", "auto")
            kwargs: Additional options
        """
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.debug_info = debug_info
        self.opt_level = opt_level
        self.arch = arch
        self.max_shared_memory = max_shared_memory
        self.mlx_shard_size = mlx_shard_size
        self.enable_fp_fusion = enable_fp_fusion
        self.enable_interleaving = enable_interleaving
        self.vectorize = vectorize
        self.memory_optimization = memory_optimization
        self.fusion_optimization = fusion_optimization
        self.metal_optimization_level = metal_optimization_level

        # Store any additional options
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert options to dictionary"""
        return {
            "num_warps": self.num_warps,
            "num_ctas": self.num_ctas,
            "debug_info": self.debug_info,
            "opt_level": self.opt_level,
            "arch": self.arch,
            "max_shared_memory": self.max_shared_memory,
            "mlx_shard_size": self.mlx_shard_size,
            "enable_fp_fusion": self.enable_fp_fusion,
            "enable_interleaving": self.enable_interleaving,
            "vectorize": self.vectorize,
            "memory_optimization": self.memory_optimization,
            "fusion_optimization": self.fusion_optimization,
            "metal_optimization_level": self.metal_optimization_level
        }

# Initialize converter
_converter = TritonToMLXConverter()

# Initialize instrumentation if available
if has_instrumentation:
    _instrumentation = MLX.metal_instrumentation.get_metal_instrumentation()
    _error_diagnostics = MLX.metal_instrumentation.get_error_diagnostics()
else:
    _instrumentation = None
    _error_diagnostics = None

def make_ttir(src, metadata, options: MetalOptions):
    """Optimize Triton IR to canonical form

    Args:
        src: Source Triton IR
        metadata: Metadata dictionary
        options: Compilation options

    Returns:
        Optimized Triton IR
    """
    if has_instrumentation:
        with _instrumentation.timer("make_ttir"):
            return _make_ttir_impl(src, metadata, options)
    return _make_ttir_impl(src, metadata, options)

def _make_ttir_impl(src, metadata, options: MetalOptions):
    """Implementation of Triton IR optimization

    Args:
        src: Source Triton IR
        metadata: Metadata dictionary
        options: Compilation options

    Returns:
        Optimized Triton IR
    """
    try:
        # Apply target-specific optimizations to the IR
        from triton.compiler.compiler import optimize_ir_for_backend
        optimized_src = optimize_ir_for_backend(src, None, options)

        # Store options in metadata for later stages
        if metadata is not None:
            metadata["opt_level"] = options.opt_level
            metadata["arch"] = options.arch

        return optimized_src
    except Exception as e:
        import traceback
        error_msg = f"Triton IR optimization failed: {str(e)}\n{traceback.format_exc()}"

        # Use error diagnostics if available
        if has_instrumentation:
            error_code, description, suggestions = _error_diagnostics.diagnose_error(
                error_msg,
                kernel_name=metadata.get("name", "unknown") if metadata else "unknown",
                source_code=src
            )

            if options.debug_info:
                print(f"Error {error_code}: {description}")
                print("Suggestions:")
                for suggestion in suggestions:
                    print(f"- {suggestion}")

        if options.debug_info:
            print(error_msg)
        raise RuntimeError(error_msg)

def make_ttgir(src, metadata, options: MetalOptions):
    """Convert TTIR to TTGIR

    Args:
        src: Source Triton IR
        metadata: Metadata dictionary
        options: Compilation options

    Returns:
        Triton GPU IR
    """
    if has_instrumentation:
        with _instrumentation.timer("make_ttgir"):
            return _make_ttgir_impl(src, metadata, options)
    return _make_ttgir_impl(src, metadata, options)

def _make_ttgir_impl(src, metadata, options: MetalOptions):
    """Implementation of TTIR to TTGIR conversion

    Args:
        src: Source Triton IR
        metadata: Metadata dictionary
        options: Compilation options

    Returns:
        Triton GPU IR
    """
    try:
        # Include thread/grid dimensions in metadata
        if metadata is not None:
            metadata["num_warps"] = options.num_warps
            metadata["num_ctas"] = options.num_ctas
            metadata["max_shared_memory"] = options.max_shared_memory

            # Add M3-specific parameters if available
            if has_hardware_optimizer and hasattr(hardware_capabilities, "chip_generation"):
                if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
                    # Add M3-specific metadata
                    metadata["chip_generation"] = "M3"
                    metadata["m3_optimizations_enabled"] = True

        return src
    except Exception as e:
        import traceback
        error_msg = f"TTGIR conversion failed: {str(e)}\n{traceback.format_exc()}"

        # Use error diagnostics if available
        if has_instrumentation:
            error_code, description, suggestions = _error_diagnostics.diagnose_error(
                error_msg,
                kernel_name=metadata.get("name", "unknown") if metadata else "unknown",
                source_code=src
            )

            if options.debug_info:
                print(f"Error {error_code}: {description}")
                print("Suggestions:")
                for suggestion in suggestions:
                    print(f"- {suggestion}")

        if options.debug_info:
            print(error_msg)
        raise RuntimeError(error_msg)

def make_mlxir(src, metadata, options: MetalOptions):
    """Convert TTGIR to MLX computation graph representation

    Args:
        src: Source Triton GPU IR
        metadata: Metadata dictionary
        options: Compilation options

    Returns:
        MLX IR
    """
    if has_instrumentation:
        with _instrumentation.timer("make_mlxir"):
            return _make_mlxir_impl(src, metadata, options)
    return _make_mlxir_impl(src, metadata, options)

def _make_mlxir_impl(src, metadata, options: MetalOptions):
    """Implementation of TTGIR to MLX conversion

    Args:
        src: Source Triton GPU IR
        metadata: Metadata dictionary
        options: Compilation options

    Returns:
        MLX IR
    """
    try:
        # Convert to MLX IR using the converter
        if has_instrumentation:
            with _instrumentation.timer("mlx_conversion"):
                mlx_ir = _converter.convert_to_mlx(
                    src,
                    num_warps=options.num_warps,
                    vectorize=options.vectorize,
                    shard_size=options.mlx_shard_size
                )
        else:
            mlx_ir = _converter.convert_to_mlx(
                src,
                num_warps=options.num_warps,
                vectorize=options.vectorize,
                shard_size=options.mlx_shard_size
            )

        # Apply MLX optimizations if available
        if has_optimizers and isinstance(mlx_ir, dict):
            if has_instrumentation:
                with _instrumentation.timer("graph_optimization"):
                    # Apply graph optimizations
                    optimized_mlx_ir, opt_stats = mlx_graph_optimizer.optimize(mlx_ir)

                    # Store optimization stats in metadata
                    if metadata is not None:
                        metadata["graph_optimization"] = opt_stats
            else:
                # Apply graph optimizations without instrumentation
                optimized_mlx_ir, opt_stats = mlx_graph_optimizer.optimize(mlx_ir)

                # Store optimization stats in metadata
                if metadata is not None:
                    metadata["graph_optimization"] = opt_stats

            # Use the optimized graph
            mlx_ir = optimized_mlx_ir

        # Store MLX metadata
        if metadata is not None:
            try:
                import mlx.core as mx
                metadata["mlx_version"] = mx.__version__
            except ImportError:
                metadata["mlx_version"] = "unknown"

            metadata["has_custom_ops"] = _converter.has_custom_ops

            # Add hardware-specific info to metadata if available
            if has_hardware_optimizer and hasattr(hardware_capabilities, "chip_generation"):
                metadata["chip_generation"] = hardware_capabilities.chip_generation.name
                if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
                    metadata["m3_capabilities"] = {
                        "shared_memory_size": hardware_capabilities.shared_memory_size,
                        "simd_width": hardware_capabilities.simd_width,
                        "max_threads_per_threadgroup": hardware_capabilities.max_threads_per_threadgroup
                    }

        return mlx_ir
    except Exception as e:
        import traceback
        error_msg = f"MLX conversion failed: {str(e)}\n{traceback.format_exc()}"

        # Use error diagnostics if available
        if has_instrumentation:
            error_code, description, suggestions = _error_diagnostics.diagnose_error(
                error_msg,
                kernel_name=metadata.get("name", "unknown") if metadata else "unknown",
                source_code=src
            )

            if options.debug_info:
                print(f"Error {error_code}: {description}")
                print("Suggestions:")
                for suggestion in suggestions:
                    print(f"- {suggestion}")

        if options.debug_info:
            print(error_msg)
        raise RuntimeError(error_msg)

def make_metallib(src, metadata, options: MetalOptions):
    """Generate Metal library from MLX computation graph

    Args:
        src: Source MLX IR
        metadata: Metadata dictionary
        options: Compilation options

    Returns:
        Compiled Metal library binary
    """
    if has_instrumentation:
        with _instrumentation.timer("make_metallib"):
            return _make_metallib_impl(src, metadata, options)
    return _make_metallib_impl(src, metadata, options)

def _make_metallib_impl(src, metadata, options: MetalOptions):
    """Implementation of Metal library generation

    Args:
        src: Source MLX IR
        metadata: Metadata dictionary
        options: Compilation options

    Returns:
        Compiled Metal library binary
    """
    try:
        import hashlib

        # Get a temporary directory for Metal library generation
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = pathlib.Path(tmp_dir)

            # Use metadata to generate a unique name
            kernel_name = metadata.get("name", "triton_kernel")
            unique_id = hashlib.md5(src.encode('utf-8')).hexdigest()[:8]
            lib_path = tmp_dir / f"{kernel_name}_{unique_id}.metallib"

            # Store IR for debugging if needed
            if options.debug_info:
                ir_path = tmp_dir / f"{kernel_name}_{unique_id}.mlxir"
                with open(ir_path, "w") as f:
                    f.write(src)

            # Record debug info if instrumentation is available
            if has_instrumentation and metadata and options.debug_info:
                source_file = metadata.get("source_file", "unknown")
                line_number = metadata.get("line_number", 0)
                _instrumentation.record_debug_info(
                    kernel_name=kernel_name,
                    source_file=source_file,
                    line_number=line_number,
                    variable_values={
                        "num_warps": options.num_warps,
                        "vectorize": options.vectorize,
                        "arch": options.arch,
                        "opt_level": options.opt_level
                    }
                )

            # Insert debug prints if needed and enabled
            if has_instrumentation and options.debug_info:
                src = _instrumentation.insert_debug_prints(src, kernel_name)

            # Compile to Metal via MLX's compilation functions
            if has_instrumentation:
                with _instrumentation.timer("mlx_to_binary"):
                    serialized_graph = _converter.mlx_ir_to_binary(src)
            else:
                serialized_graph = _converter.mlx_ir_to_binary(src)

            # Save any compilation metadata
            if metadata is not None:
                metadata["metal_lib_path"] = str(lib_path)
                metadata["metal_kernel_name"] = kernel_name

                # Add hardware-specific metadata if applicable
                if has_hardware_optimizer and hasattr(hardware_capabilities, "chip_generation"):
                    if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
                        metadata["optimized_for_m3"] = True

            return serialized_graph
    except Exception as e:
        import traceback
        error_msg = f"Metal library generation failed: {str(e)}\n{traceback.format_exc()}"

        # Use error diagnostics if available
        if has_instrumentation:
            error_code, description, suggestions = _error_diagnostics.diagnose_error(
                error_msg,
                kernel_name=metadata.get("name", "unknown") if metadata else "unknown",
                source_code=src
            )

            if options.debug_info:
                print(f"Error {error_code}: {description}")
                print("Suggestions:")
                for suggestion in suggestions:
                    print(f"- {suggestion}")

        if options.debug_info:
            print(error_msg)
        raise RuntimeError(error_msg)