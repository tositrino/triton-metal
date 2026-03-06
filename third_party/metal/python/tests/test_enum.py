"""
Debug script to check what's in the MemoryLayout enum
"""

import os
import sys
from enum import Enum



# First try to import directly from memory_layout_optimizer
print("Directly from memory_layout_optimizer:")
from mlx.memory_layout_optimizer import MemoryLayout as MLOMemoryLayout
print("Items in MLOMemoryLayout:")
for layout in MLOMemoryLayout:
    print(f"  {layout.name} = {layout.value}")

# Then try to import from metal_memory_manager if available
try:
    print("\nFrom metal_memory_manager:")
    from MLX.metal_memory_manager import MemoryLayout as MMMemoryLayout
    print("Items in MMMemoryLayout:")
    for layout in MMMemoryLayout:
        print(f"  {layout.name} = {layout.value}")
except ImportError:
    print("\nCouldn't import MemoryLayout from metal_memory_manager")

# Check if the COALESCED value exists
print("\nChecking for COALESCED:")
try:
    print(f"MLOMemoryLayout.COALESCED = {MLOMemoryLayout.COALESCED.value}")
except AttributeError:
    print("COALESCED not found in MLOMemoryLayout")

# Check our fallback implementation
print("\nFallback implementation in memory_layout_optimizer.py:")
fallback_code = """
class MemoryLayout(Enum):
    \"""Enum for memory layouts\"""
    DEFAULT = 0
    ROW_MAJOR = 1
    COLUMN_MAJOR = 2
    BLOCK_BASED = 3
    TILED = 4
    INTERLEAVED = 5
    SIMD_ALIGNED = 6
    TEXTURE_OPTIMIZED = 7
    COALESCED = 8
"""
print(fallback_code) 