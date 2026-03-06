"""
Test suite for special_ops.py mathematical functions.
Validates accuracy of MLX implementations against reference values.
"""


import numpy as np
import os
import sys
from typing import Callable, Dict, List, Tuple

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the functions to test
from MLX.special_ops import (
    SpecialMathFunctions,
    NumericalFunctions,
    get_special_ops_map
)

# Import MLX for testing
import mlx.core as mx