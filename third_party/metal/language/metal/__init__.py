"""Metal language extension for Triton

This module provides Metal-specific functions and utilities for Triton.
"""

from . import libdevice
from .utils import (
    threadgroup_id, thread_id, num_threads, num_threadgroups,
    threadgroup_barrier, device_barrier,
    get_chip_generation, get_max_threads_per_threadgroup, get_max_shared_memory
)

__all__ = [
    "libdevice",
    # Thread and Grid Functions
    "threadgroup_id",
    "thread_id",
    "num_threads",
    "num_threadgroups",
    # Memory Barrier Functions
    "threadgroup_barrier",
    "device_barrier",
    # Apple Silicon Specific Functions
    "get_chip_generation",
    "get_max_threads_per_threadgroup",
    "get_max_shared_memory",
]

# Initialize Metal device library
libdevice.init_libdevice() 