"""Metal backend for Triton

This package provides integration between Triton and the Metal backend for
Apple Silicon GPUs.
"""

# Import and register Metal backend components
from .compiler import MetalBackend
from .driver import MetalDriver

__all__ = ["MetalBackend", "MetalDriver"] 