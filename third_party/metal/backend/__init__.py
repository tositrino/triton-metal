"""
Triton Metal Backend for Apple Silicon GPUs

This backend leverages MLX to run Triton kernels on Apple Silicon GPUs via Metal.
"""

from .driver import MetalDriver
from .compiler import MetalBackend, MetalOptions

__all__ = ["MetalDriver", "MetalBackend", "MetalOptions"]

# Register the backend with Triton
def register_backend():
    """Register the Metal backend with Triton"""
    try:
        from triton.backends import register_backend
        from triton.backends.compiler import GPUTarget
        
        # Check if the Metal backend can be activated
        if MetalDriver.is_active():
            # Register Metal backend
            register_backend("metal", MetalBackend, MetalDriver)
            
            # Log registration
            import logging
            logger = logging.getLogger("triton.metal")
            logger.info("Metal backend registered for Apple Silicon GPUs")
            
            # Register device detection
            return True
        else:
            return False
    except ImportError as e:
        print(f"Could not register Metal backend: {e}")
        return False
        
# Automatically register if this module is imported directly
register_backend() 