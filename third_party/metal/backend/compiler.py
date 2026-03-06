from triton.backends.compiler import BaseBackend, GPUTarget
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from types import ModuleType
import functools
import tempfile
import os
import pathlib
import sys
import json
import subprocess
import hashlib
import time

# Add Python directory to path for Metal-specific modules
metal_python_dir = os.path.join(os.path.dirname(__file__), '..', 'python')
if metal_python_dir not in sys.path:
    sys.path.insert(0, metal_python_dir)

try:
    # Import hardware detection
    from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration

    # Import optimizers
    try:
        # Import our optimizers
        import mlx_graph_optimizer
        import metal_memory_manager

        has_optimizers = True
    except ImportError:
        print("Warning: MLX Graph Optimizer modules not available. Advanced optimizations will be disabled.")
        has_optimizers = False
except ImportError:
    print("Warning: Metal hardware detection not available. Hardware-specific optimizations will be disabled.")
    has_optimizers = False

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

class MetalBackend(BaseBackend):
    """Triton Metal backend implementation"""

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'metal'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "metallib"
        self._mlx = None
        self._converter = None
        self._driver = None

        # Import instrumentation and auto-tuner
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

        # Initialize instrumentation
        try:
            import metal_instrumentation
            self.instrumentation = metal_instrumentation.get_metal_instrumentation()
            self.error_diagnostics = metal_instrumentation.get_error_diagnostics()
            self.has_instrumentation = True
        except ImportError:
            print("Warning: metal_instrumentation module not found. Debug and performance tracking will be disabled.")
            self.has_instrumentation = False
            self.instrumentation = None
            self.error_diagnostics = None

        # Initialize auto-tuner
        try:
            import metal_auto_tuner
            self.auto_tuner_module = metal_auto_tuner
            self.has_auto_tuner = True
        except ImportError:
            print("Warning: metal_auto_tuner module not found. Auto-tuning will be disabled.")
            self.has_auto_tuner = False
            self.auto_tuner_module = None

    @property
    def mlx(self):
        """Lazy load MLX"""
        if self._mlx is None:
            try:
                import mlx.core as mx
                self._mlx = mx
            except ImportError:
                raise ImportError("MLX is required for Metal backend. Install it with 'pip install mlx'")
        return self._mlx

    @property
    def converter(self):
        """Get Triton to MLX converter"""
        if self._converter is None:
            import sys
            # Add metal package to path if needed
            metal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if metal_path not in sys.path:
                sys.path.append(metal_path)

            try:
                from python.triton_to_metal_converter import TritonToMLXConverter
                self._converter = TritonToMLXConverter()
            except ImportError:
                raise ImportError("TritonToMLXConverter not found. Make sure the metal package is installed properly.")
        return self._converter

    @property
    def driver(self):
        """Get Metal driver instance"""
        if self._driver is None:
            from .driver import MetalDriver
            self._driver = MetalDriver()
        return self._driver

    def hash(self) -> str:
        """Get unique backend identifier"""
        try:
            import mlx.core as mx
            version = mx.__version__
            return f'mlx-{version}-metal'
        except ImportError:
            return 'mlx-unknown-metal'

    def parse_options(self, options: dict) -> MetalOptions:
        """Parse compilation options"""
        args = {'arch': self.target.arch if hasattr(self.target, 'arch') else 'apple-silicon'}
        args.update({k: options[k] for k in MetalOptions.__dataclass_fields__.keys()
                    if k in options and options[k] is not None})
        return MetalOptions(**args)

    def add_stages(self, stages, options):
        """Define compilation stages"""
        options = self.parse_options(options)

        # Define the compilation pipeline stages
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["mlxir"] = lambda src, metadata: self.make_mlxir(src, metadata, options)
        stages["metallib"] = lambda src, metadata: self.make_metallib(src, metadata, options)

    def make_ttir(self, src, metadata, options: MetalOptions):
        """Optimize Triton IR to canonical form"""
        if self.has_instrumentation:
            with self.instrumentation.timer("make_ttir"):
                return self._make_ttir_impl(src, metadata, options)
        return self._make_ttir_impl(src, metadata, options)

    def _make_ttir_impl(self, src, metadata, options: MetalOptions):
        """Implementation of Triton IR optimization"""
        try:
            # Apply target-specific optimizations to the IR
            from triton.compiler.compiler import optimize_ir_for_backend
            optimized_src = optimize_ir_for_backend(src, self.target, options)

            # Store options in metadata for later stages
            if metadata is not None:
                metadata["opt_level"] = options.opt_level
                metadata["arch"] = options.arch

            return optimized_src
        except Exception as e:
            import traceback
            error_msg = f"Triton IR optimization failed: {str(e)}\n{traceback.format_exc()}"

            # Use error diagnostics if available
            if self.has_instrumentation:
                error_code, description, suggestions = self.error_diagnostics.diagnose_error(
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

    def make_ttgir(self, src, metadata, options: MetalOptions):
        """Convert TTIR to TTGIR"""
        if self.has_instrumentation:
            with self.instrumentation.timer("make_ttgir"):
                return self._make_ttgir_impl(src, metadata, options)
        return self._make_ttgir_impl(src, metadata, options)

    def _make_ttgir_impl(self, src, metadata, options: MetalOptions):
        """Implementation of TTIR to TTGIR conversion"""
        try:
            # This would typically involve layout planning and optimization
            # For now, we'll make minimal changes and rely on MLX for optimizations

            # Include thread/grid dimensions in metadata
            if metadata is not None:
                metadata["num_warps"] = options.num_warps
                metadata["num_ctas"] = options.num_ctas
                metadata["max_shared_memory"] = options.max_shared_memory

                # Add M3-specific parameters if available
                if has_optimizers and hasattr(hardware_capabilities, "chip_generation"):
                    if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
                        # Add M3-specific metadata
                        metadata["chip_generation"] = "M3"
                        metadata["m3_optimizations_enabled"] = True

            return src
        except Exception as e:
            import traceback
            error_msg = f"TTGIR conversion failed: {str(e)}\n{traceback.format_exc()}"

            # Use error diagnostics if available
            if self.has_instrumentation:
                error_code, description, suggestions = self.error_diagnostics.diagnose_error(
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

    def make_mlxir(self, src, metadata, options: MetalOptions):
        """Convert TTGIR to MLX computation graph representation"""
        if self.has_instrumentation:
            with self.instrumentation.timer("make_mlxir"):
                return self._make_mlxir_impl(src, metadata, options)
        return self._make_mlxir_impl(src, metadata, options)

    def _make_mlxir_impl(self, src, metadata, options: MetalOptions):
        """Implementation of TTGIR to MLX conversion"""
        # Convert Triton IR to MLX computation graph
        try:
            # Import Metal IR transformations for advanced optimizations
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
            try:
                import metal_ir_transforms
                has_ir_transforms = True
            except ImportError:
                has_ir_transforms = False
                print("Warning: metal_ir_transforms module not found. Using basic conversion without Metal-specific optimizations.")

            # Import MLX graph optimizer
            try:
                from mlx_graph_optimizer import MLXGraphOptimizer
                has_graph_optimizer = True
            except ImportError:
                has_graph_optimizer = False
                print("Warning: mlx_graph_optimizer module not found. Graph optimization will be disabled.")

            # Try to import hardware detection
            try:
                from metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
                has_hardware_detection = True
            except ImportError:
                has_hardware_detection = False
                print("Warning: metal_hardware_optimizer module not found. Hardware-specific optimizations will be disabled.")

            # Try to import M3-specific optimizers if on M3 hardware
            has_m3_optimizers = False
            if has_hardware_detection and hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
                try:
                    import m3_graph_optimizer
                    import m3_memory_manager
                    import m3_fusion_optimizer
                    has_m3_optimizers = True
                    print("M3 hardware detected. Using M3-specific optimizations.")
                except ImportError:
                    print("Warning: M3-specific optimization modules not found. Using generic optimizations for M3.")

            # First, parse the IR to a format that can be transformed
            parsed_ir = self._parse_ir_for_transform(src)

            # Apply Metal-specific IR transformations if available
            if has_ir_transforms and parsed_ir:
                if self.has_instrumentation:
                    with self.instrumentation.timer("metal_ir_transforms"):
                        transformed_ir = metal_ir_transforms.optimize_ir(parsed_ir, options)
                else:
                    transformed_ir = metal_ir_transforms.optimize_ir(parsed_ir, options)
            else:
                transformed_ir = parsed_ir

            # Convert to MLX computation graph
            if self.has_instrumentation:
                with self.instrumentation.timer("triton_to_mlx_conversion"):
                    mlx_graph = self.converter.convert(transformed_ir, metadata)
            else:
                mlx_graph = self.converter.convert(transformed_ir, metadata)

            # Apply GPU-specific optimizations
            if has_graph_optimizer and mlx_graph:
                # Apply MLX graph optimizations
                if self.has_instrumentation:
                    with self.instrumentation.timer("mlx_graph_optimization"):
                        graph_optimizer = MLXGraphOptimizer(hardware_capabilities)
                        for optimization_pass in graph_optimizer.passes:
                            mlx_graph, stats = optimization_pass.apply(mlx_graph)
                            if stats and options.debug_info:
                                print(f"Applied {optimization_pass.name}: {stats}")
                else:
                    graph_optimizer = MLXGraphOptimizer(hardware_capabilities)
                    for optimization_pass in graph_optimizer.passes:
                        mlx_graph, stats = optimization_pass.apply(mlx_graph)

                # Apply M3-specific optimizations if available and on M3 hardware
                if has_m3_optimizers and has_hardware_detection and hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
                    if self.has_instrumentation:
                        with self.instrumentation.timer("m3_specific_optimization"):
                            # Apply M3 graph optimizations
                            m3_optimizer = m3_graph_optimizer.get_m3_graph_optimizer()
                            mlx_graph, m3_stats = m3_optimizer.optimize(mlx_graph)
                            if options.debug_info:
                                print(f"Applied M3 graph optimizations: {m3_stats}")

                            # Apply M3 memory optimizations
                            m3_memory_mgr = m3_memory_manager.M3MemoryManager(hardware_capabilities)
                            mlx_graph = m3_memory_mgr.optimize_graph_memory(mlx_graph)

                            # Apply M3 fusion optimizations
                            m3_fusion_opt = m3_fusion_optimizer.get_m3_fusion_optimizer()
                            if m3_fusion_opt and "ops" in mlx_graph:
                                mlx_graph["ops"] = m3_fusion_opt.optimize(mlx_graph["ops"])
                    else:
                        # Apply M3 graph optimizations
                        m3_optimizer = m3_graph_optimizer.get_m3_graph_optimizer()
                        mlx_graph, m3_stats = m3_optimizer.optimize(mlx_graph)

                        # Apply M3 memory optimizations
                        m3_memory_mgr = m3_memory_manager.M3MemoryManager(hardware_capabilities)
                        mlx_graph = m3_memory_mgr.optimize_graph_memory(mlx_graph)

                        # Apply M3 fusion optimizations
                        m3_fusion_opt = m3_fusion_optimizer.get_m3_fusion_optimizer()
                        if m3_fusion_opt and "ops" in mlx_graph:
                            mlx_graph["ops"] = m3_fusion_opt.optimize(mlx_graph["ops"])

                    # Add M3-specific metadata
                    if "metadata" not in mlx_graph:
                        mlx_graph["metadata"] = {}
                    mlx_graph["metadata"]["m3_optimized"] = True
                    mlx_graph["metadata"]["chip_generation"] = "M3"

            # Add metadata about the compilation process
            if "metadata" not in mlx_graph:
                mlx_graph["metadata"] = {}

            mlx_graph["metadata"]["compiler_version"] = "0.1.0"
            mlx_graph["metadata"]["options"] = options.to_dict()
            mlx_graph["metadata"]["num_warps"] = options.num_warps
            mlx_graph["metadata"]["num_ctas"] = options.num_ctas

            if has_hardware_detection:
                mlx_graph["metadata"]["chip_generation"] = hardware_capabilities.chip_generation.name

            return mlx_graph

        except Exception as e:
            import traceback
            error_msg = f"MLX conversion failed: {str(e)}\n{traceback.format_exc()}"

            # Use error diagnostics if available
            if self.has_instrumentation:
                error_code, description, suggestions = self.error_diagnostics.diagnose_error(
                    error_msg,
                    kernel_name=metadata.get("name", "unknown") if metadata else "unknown",
                    source_code=str(src)
                )

                if options.debug_info:
                    print(f"Error {error_code}: {description}")
                    print("Suggestions:")
                    for suggestion in suggestions:
                        print(f"- {suggestion}")

            if options.debug_info:
                print(error_msg)
            raise RuntimeError(error_msg)

    def _parse_ir_for_transform(self, ir):
        """Parse IR for transformation

        Args:
            ir: IR to parse

        Returns:
            Parsed IR
        """
        try:
            # For now, we'll assume the IR is a JSON string
            # In a real implementation, this would parse the Triton IR format
            # to a format that can be transformed

            # This is a placeholder implementation
            import json
            return json.loads(ir) if isinstance(ir, str) else ir
        except Exception as e:
            print(f"Warning: Failed to parse IR for transformation: {e}")
            return None

    def _serialize_ir_after_transform(self, transformed_ir):
        """Serialize transformed IR back to the format expected by the converter

        Args:
            transformed_ir: Transformed IR operations

        Returns:
            Serialized IR
        """
        try:
            # For now, we'll assume the converter expects a JSON string
            # In a real implementation, this would convert the transformed IR
            # to the format expected by the converter

            # This is a placeholder implementation
            import json
            return json.dumps(transformed_ir)
        except Exception as e:
            print(f"Warning: Failed to serialize transformed IR: {e}")
            # Return the original transformed_ir as a fallback
            return transformed_ir

    def make_metallib(self, src, metadata, options: MetalOptions):
        """Generate Metal library from MLX computation graph"""
        if self.has_instrumentation:
            with self.instrumentation.timer("make_metallib"):
                return self._make_metallib_impl(src, metadata, options)
        return self._make_metallib_impl(src, metadata, options)

    def _make_metallib_impl(self, src, metadata, options: MetalOptions):
        """Implementation of Metal library generation"""
        try:
            # MLX handles the Metal library generation
            # We'll create a serialized representation of the computation

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
                if self.has_instrumentation and metadata and options.debug_info:
                    source_file = metadata.get("source_file", "unknown")
                    line_number = metadata.get("line_number", 0)
                    self.instrumentation.record_debug_info(
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
                if self.has_instrumentation and options.debug_info:
                    src = self.instrumentation.insert_debug_prints(src, kernel_name)

                # Compile to Metal via MLX's compilation functions
                # This is simplified - we would need to integrate with MLX's actual compilation
                if self.has_instrumentation:
                    with self.instrumentation.timer("mlx_to_binary"):
                        serialized_graph = self.converter.mlx_ir_to_binary(src)
                else:
                    serialized_graph = self.converter.mlx_ir_to_binary(src)

                # In the actual implementation, we would call MLX's Metal compiler
                # For now, we'll simulate the resulting binary

                # Save any compilation metadata
                if metadata is not None:
                    metadata["metal_lib_path"] = str(lib_path)
                    metadata["metal_kernel_name"] = kernel_name

                    # Add M3-specific metadata if applicable
                    if has_optimizers and hasattr(hardware_capabilities, "chip_generation"):
                        if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
                            metadata["optimized_for_m3"] = True

                return serialized_graph
        except Exception as e:
            import traceback
            error_msg = f"Metal library generation failed: {str(e)}\n{traceback.format_exc()}"

            # Use error diagnostics if available
            if self.has_instrumentation:
                error_code, description, suggestions = self.error_diagnostics.diagnose_error(
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

    def get_module_map(self) -> Dict[str, ModuleType]:
        """Return module mapping"""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            return {
                "mlx": mx,
                "mlx.core": mx,
                "mlx.nn": nn
            }
        except ImportError:
            return {}

    def load_dialects(self, ctx):
        """Load dialects"""
        # Load Metal-specific dialects when we have them
        pass

    def get_runtime_library(self):
        """Return runtime library with Metal-specific functions"""
        lib_dir = os.path.join(os.path.dirname(__file__), "lib")
        if os.path.exists(lib_dir):
            return lib_dir
        return None

    def get_device_properties(self):
        """Get Metal device properties"""
        props = self.driver.metal_info.copy()

        # Add M3-specific capabilities if applicable
        if has_optimizers and hasattr(hardware_capabilities, "chip_generation"):
            if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
                props["m3_features"] = {
                    "dynamic_caching": True,
                    "hardware_ray_tracing": True,
                    "hardware_mesh_shading": True,
                    "tensor_cores": True,
                    "simdgroup_width": 32,
                    "shared_memory_size": 65536  # 64KB
                }

        return props

    def auto_tune_kernel(self, kernel_name, signature, config_args, config_kwargs, n_trials=100,
                       search_strategy="random", kernel_fn=None, parallel=True, num_workers=4):
        """Auto-tune a kernel with Metal backend

        Args:
            kernel_name: Name of the kernel
            signature: Kernel signature
            config_args: Config args
            config_kwargs: Config kwargs
            n_trials: Number of trials to run
            search_strategy: Search strategy ("random", "grid", or "bayesian")
            kernel_fn: Kernel function object
            parallel: Whether to run evaluations in parallel (default: True)
            num_workers: Number of worker threads when parallel is True (default: 4)

        Returns:
            Best configuration found
        """
        if not self.has_auto_tuner:
            print("Auto-tuning not available. Using default configuration.")
            return None

        # Import the module here to avoid circular imports
        try:
            from metal_hardware_optimizer import hardware_capabilities

            # Determine operation type from kernel name and signature
            operation_type = 'general'

            # Check for matmul/gemm kernels
            if any(x in kernel_name.lower() for x in ["matmul", "gemm", "mm", "matrix"]):
                operation_type = 'matmul'
                print(f"Detected matrix multiplication kernel: {kernel_name}")
            # Check for convolution kernels
            elif any(x in kernel_name.lower() for x in ["conv", "filter", "corr"]):
                operation_type = 'conv'
                print(f"Detected convolution kernel: {kernel_name}")
            # Try to infer from signature
            elif signature:
                # Look for patterns in function signature indicating matmul
                if signature.count('matrix') >= 2 or signature.count('mat') >= 2:
                    operation_type = 'matmul'
                    print(f"Inferred matrix multiplication from signature: {signature}")
                # Look for patterns indicating convolution
                elif 'conv' in signature or ('filter' in signature and 'kernel' in signature):
                    operation_type = 'conv'
                    print(f"Inferred convolution from signature: {signature}")

            # Get appropriate tunable parameters based on operation type
            if operation_type == 'matmul':
                tunable_params = self.auto_tuner_module.get_matmul_metal_params()
            elif operation_type == 'conv':
                tunable_params = self.auto_tuner_module.get_conv_metal_params()
            else:
                tunable_params = self.auto_tuner_module.get_common_metal_params()

            # Display hardware information for debugging
            if self.has_instrumentation:
                self.instrumentation.record_debug_info(
                    kernel_name=kernel_name,
                    source_file="auto_tuning",
                    line_number=0,
                    variable_values={
                        "operation_type": operation_type,
                        "chip_generation": hardware_capabilities.chip_generation.name,
                        "n_trials": n_trials,
                        "search_strategy": search_strategy,
                        "parallel": parallel,
                        "num_workers": num_workers
                    }
                )

            # Create auto-tuner
            tuner = self.auto_tuner_module.MetalAutoTuner(
                kernel_name,
                tunable_params,
                n_trials=n_trials,
                search_strategy=search_strategy,
                operation_type=operation_type,
                use_hardware_optimizer=True
            )

            # Define evaluation function that tests the kernel with different configs
            def evaluate_config(config):
                # Update kwargs with config
                updated_kwargs = config_kwargs.copy()
                updated_kwargs.update(config)

                try:
                    # If we have instrumentation, measure performance
                    if self.has_instrumentation:
                        with self.instrumentation.timer(f"kernel_{kernel_name}"):
                            # We'd run the kernel here with the provided config
                            if kernel_fn:
                                kernel_fn(*config_args, **updated_kwargs)

                        # Get the timing
                        kernel_time = self.instrumentation.get_last_timer(f"kernel_{kernel_name}")
                        runtime_ms = kernel_time * 1000 if kernel_time else float("inf")
                    else:
                        # Use a simple timer if no instrumentation
                        start = time.time()
                        if kernel_fn:
                            kernel_fn(*config_args, **updated_kwargs)
                        end = time.time()
                        runtime_ms = (end - start) * 1000

                    # Create configuration result
                    return self.auto_tuner_module.ConfigurationResult(
                        config=config,
                        runtime_ms=runtime_ms,
                        success=True
                    )
                except Exception as e:
                    # If the kernel fails, return failure
                    return self.auto_tuner_module.ConfigurationResult(
                        config=config,
                        runtime_ms=float("inf"),
                        success=False,
                        metrics={"error": str(e)}
                    )

            # Run tuning if kernel_fn is provided
            if kernel_fn:
                print(f"Auto-tuning {operation_type} kernel '{kernel_name}' with {search_strategy} strategy")
                print(f"Hardware: {hardware_capabilities.chip_generation.name}, Running {n_trials} trials")

                # Adjust num_workers based on hardware
                if num_workers is None or num_workers <= 0:
                    # Use 1/2 of available cores by default
                    num_workers = max(1, hardware_capabilities.num_cores // 2)

                best_config = tuner.tune(
                    evaluate_config,
                    parallel=parallel,
                    num_workers=num_workers
                )

                # Log the best configuration if instrumentation is available
                if self.has_instrumentation:
                    self.instrumentation.record_debug_info(
                        kernel_name=kernel_name,
                        source_file="auto_tuned",
                        line_number=0,
                        variable_values=best_config
                    )

                print(f"Best configuration for {kernel_name}: {best_config}")
                return best_config
            else:
                # If no kernel function is provided, just return hardware-optimized default config
                default_config = tuner.get_hardware_optimized_default_config()
                print(f"No kernel function provided, using hardware-optimized default: {default_config}")
                return default_config

        except Exception as e:
            print(f"Error during auto-tuning: {e}")
            return None