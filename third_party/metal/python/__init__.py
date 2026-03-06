"""
Triton Metal Backend

This package provides support for running Triton kernels on Apple Silicon GPUs
through MLX and Metal.
"""

__version__ = "0.1.0"

import os
import sys
import platform

# Check if running on macOS with Apple Silicon
if sys.platform != "darwin" or platform.machine() != "arm64":
    is_available = False
else:
    # Check for MLX availability
    try:
        import mlx.core as mx
        is_available = hasattr(mx, "metal") or "metal" in str(mx.default_device())
    except ImportError:
        is_available = False

# Import current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Export public API
from MLX.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
from mlx.operation_mapping import MLXDispatcher, OpCategory, op_conversion_registry
from MLX.metal_fusion_optimizer import FusionOptimizer, fusion_optimizer

__all__ = [
    "is_available",
    "hardware_capabilities",
    "AppleSiliconGeneration",
    "MLXDispatcher",
    "OpCategory", 
    "op_conversion_registry",
    "FusionOptimizer",
    "fusion_optimizer",
]

# Additional version and metadata
__author__ = "Triton Contributors"
__description__ = "Metal backend for Triton on Apple Silicon GPUs"

# Register the Metal backend with Triton - must be after the backend modules are installed
def register_backend():
    """Register the Metal backend with Triton"""
    try:
        # Add the Metal backend directory to the Python path
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
        if backend_dir not in sys.path:
            sys.path.append(backend_dir)
        
        # Import the backend module
        from backend import register_backend as reg_backend
        
        # Register the backend
        return reg_backend()
    except ImportError as e:
        print(f"Warning: Could not register Metal backend: {e}")
        return False
    except Exception as e:
        print(f"Error registering Metal backend: {e}")
        return False

# Try to register the backend if we're the main module
if __name__ != "__main__":
    try:
        register_backend()
    except Exception as e:
        print(f"Error during Metal backend registration: {e}") 