"""Metal compiler for Triton

This module implements the compiler interface for Metal backend on Apple Silicon GPUs.
It provides a bridge to the Metal backend implementation.
"""

import os
import sys
import tempfile
import pathlib
from typing import Dict, Union, Optional, List, Tuple, Any, Type
from types import ModuleType

from triton.backends.compiler import BaseBackend, GPUTarget

# Add Metal package to path
metal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                          '..', '..', '..', '..', 
                                          'third_party', 'metal'))
if metal_path not in sys.path:
    sys.path.insert(0, metal_path)

class MetalBackend(BaseBackend):
    """Backend implementation for Metal backend on Apple Silicon GPUs"""
    
    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        """Check if this backend supports the given target
        
        Args:
            target: Target device specification
            
        Returns:
            True if the target is supported, False otherwise
        """
        return target.backend == 'metal'
    
    def __init__(self, target: GPUTarget) -> None:
        """Initialize the Metal backend
        
        Args:
            target: Target device specification
        """
        super().__init__(target)
        
        # Import Metal backend implementation
        try:
            # Import metal-specific libraries
            # We follow a lazy initialization pattern to only load what's needed
            self._mlx = None
            self._driver = None
            self._converter = None
            self._instrumentation = None
            
            # Initialize version information
            self._version = self._get_version()
            
            # Set file extension for compiled binaries
            self.binary_ext = "metallib"
            
            # Check for capabilities
            self._has_instrumentation = self._check_instrumentation_available()
            
        except ImportError as e:
            print(f"Error initializing Metal backend: {e}")
            raise
    
    def _get_version(self) -> str:
        """Get Metal backend version
        
        Returns:
            Version string
        """
        try:
            # Try to get version from MLX
            import mlx.core as mx
            return getattr(mx, "__version__", "unknown")
        except ImportError:
            return "unknown"
    
    def _check_instrumentation_available(self) -> bool:
        """Check if Metal instrumentation is available
        
        Returns:
            True if instrumentation is available, False otherwise
        """
        try:
            from python.metal_instrumentation import get_metal_instrumentation
            return True
        except ImportError:
            return False
    
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
    def driver(self):
        """Get Metal driver instance"""
        if self._driver is None:
            from .driver import MetalDriver
            self._driver = MetalDriver()
        return self._driver
    
    @property
    def converter(self):
        """Get Metal converter instance"""
        if self._converter is None:
            try:
                from python.triton_to_metal_converter import TritonToMLXConverter
                self._converter = TritonToMLXConverter()
            except ImportError:
                raise ImportError("Metal converter is required. Please ensure the Metal backend is properly installed.")
        return self._converter
    
    @property
    def instrumentation(self):
        """Get Metal instrumentation instance"""
        if not self._has_instrumentation:
            print("Warning: Metal instrumentation not available.")
            return None
        
        if self._instrumentation is None:
            try:
                from python.metal_instrumentation import get_metal_instrumentation
                self._instrumentation = get_metal_instrumentation()
            except ImportError:
                self._has_instrumentation = False
                return None
        return self._instrumentation
    
    def hash(self) -> str:
        """Get unique backend identifier
        
        Returns:
            String identifier for this backend
        """
        return f'mlx-{self._version}-metal'
    
    def parse_options(self, options: dict) -> object:
        """Parse compilation options
        
        Args:
            options: Dictionary of options
            
        Returns:
            Parsed options object
        """
        # Import MetalOptions
        try:
            from python.metal_backend import MetalOptions
        except ImportError:
            # If MetalOptions isn't available, use a simple dictionary
            class MetalOptions:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
            
        # Get target architecture
        arch = self.target.arch if hasattr(self.target, 'arch') else 'apple-silicon'
        
        # Create options dictionary
        args = {'arch': arch}
        
        # Update with options
        option_keys = [
            'num_warps', 'num_ctas', 'debug_info', 'opt_level', 'max_shared_memory',
            'mlx_shard_size', 'enable_fp_fusion', 'enable_interleaving', 'vectorize',
            'memory_optimization', 'fusion_optimization', 'metal_optimization_level'
        ]
        
        for k in option_keys:
            if k in options and options[k] is not None:
                args[k] = options[k]
        
        # Create and return options object
        return MetalOptions(**args)
    
    def add_stages(self, stages: dict, options: object) -> None:
        """Define compilation stages
        
        Args:
            stages: Dictionary to populate with compilation stages
            options: Parsed options object
        """
        try:
            # Import Metal backend
            from python.metal_backend import (
                make_ttir, make_ttgir, make_mlxir, make_metallib
            )
            
            # Define the compilation pipeline stages
            stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: make_ttgir(src, metadata, options)
            stages["mlxir"] = lambda src, metadata: make_mlxir(src, metadata, options)
            stages["metallib"] = lambda src, metadata: make_metallib(src, metadata, options)
        except ImportError as e:
            # Fallback to default implementation
            print(f"Warning: Could not import Metal backend modules: {e}")
            print("Using stub implementations for compilation stages.")
            
            # Stub implementations
            stages["ttir"] = lambda src, metadata: src
            stages["ttgir"] = lambda src, metadata: src
            stages["mlxir"] = lambda src, metadata: src
            stages["metallib"] = lambda src, metadata: bytes(src, 'utf-8') if isinstance(src, str) else src
    
    def load_dialects(self, context) -> None:
        """Load additional MLIR dialects into the provided context
        
        Args:
            context: MLIR context
        """
        # Try to load Metal-specific dialects
        try:
            from python.metal_mlir_ext import load_metal_dialects
            load_metal_dialects(context)
        except ImportError:
            # No Metal dialects to load
            pass
    
    def get_module_map(self) -> Dict[str, ModuleType]:
        """Return module mapping
        
        Returns:
            Dictionary mapping module names to module objects
        """
        modules = {}
        
        # Try to add MLX modules
        try:
            import mlx.core as mx
            modules["mlx"] = mx
            modules["mlx.core"] = mx
            
            try:
                import mlx.nn as nn
                modules["mlx.nn"] = nn
            except ImportError:
                pass
        except ImportError:
            pass
        
        # Try to add Metal-specific modules
        try:
            from third_party.metal.language import metal
            modules["metal"] = metal
        except ImportError:
            pass
        
        return modules 