"""
Metal Driver for Triton

This module provides the driver for Triton to use Apple Silicon GPUs 
through Metal and MLX.
"""

import os
import sys
import platform
import importlib.util
from typing import Dict, List, Any, Optional, Tuple, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import hardware capabilities
try:
    from python.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
except ImportError as e:
    print(f"Error importing hardware_capabilities: {e}")
    
class Target:
    """
    Target descriptor for Metal backend
    """
    
    def __init__(self, backend: str, arch: str):
        """
        Initialize a target descriptor
        
        Args:
            backend: Backend name (e.g., "metal")
            arch: Target architecture (e.g., "apple8", "apple9")
        """
        self.backend = backend
        self.arch = arch
        
    def __str__(self) -> str:
        """String representation"""
        return f"{self.backend}:{self.arch}"

class MetalDriver:
    """
    Driver for Metal backend execution
    
    This class handles the initialization and management of Metal 
    devices for Triton.
    """
    
    def __init__(self):
        """Initialize Metal driver"""
        self.detected = self._detect_metal()
        self.targets = self._get_all_targets() if self.detected else []
        self.current_target_idx = 0 if self.targets else -1
        
        # Try to load MLX
        self.mlx = None
        if self.detected:
            self._load_mlx()
    
    def _detect_metal(self) -> bool:
        """
        Detect if Metal is available
        
        Returns:
            True if Metal is available, False otherwise
        """
        # Check platform
        if sys.platform != "darwin":
            return False
        
        # Check if running on Apple Silicon
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.UNKNOWN:
            return False
        
        # Check if MLX can be imported 
        try:
            if importlib.util.find_spec("mlx"):
                return True
        except ImportError:
            pass
        
        return False
    
    def _get_all_targets(self) -> List[Target]:
        """
        Get all available Metal targets
        
        Returns:
            List of available targets
        """
        targets = []
        
        # Add current Metal device
        if hardware_capabilities.chip_generation != AppleSiliconGeneration.UNKNOWN:
            arch = f"apple{hardware_capabilities.chip_generation.value + 6}"  # M1=7, M2=8, M3=9
            targets.append(Target("metal", arch))
        
        return targets
    
    def _load_mlx(self):
        """Load MLX module"""
        try:
            import mlx.core as mx
            self.mlx = mx
        except ImportError as e:
            print(f"Error loading MLX: {e}")
            self.mlx = None
    
    def is_active(self) -> bool:
        """
        Check if Metal driver is active
        
        Returns:
            True if Metal driver is active, False otherwise
        """
        return self.detected and self.mlx is not None
    
    def get_current_target(self) -> Optional[Target]:
        """
        Get current Metal target
        
        Returns:
            Current target or None if no target available
        """
        if not self.targets or self.current_target_idx < 0:
            return None
        
        return self.targets[self.current_target_idx]
    
    def set_target(self, target_idx: int) -> bool:
        """
        Set current target
        
        Args:
            target_idx: Target index
            
        Returns:
            True if successful, False otherwise
        """
        if target_idx >= 0 and target_idx < len(self.targets):
            self.current_target_idx = target_idx
            return True
        
        return False
    
    def num_devices(self) -> int:
        """
        Get number of available Metal devices
        
        Returns:
            Number of devices
        """
        return len(self.targets)
    
    def get_device_properties(self, device_idx: int = 0) -> Dict[str, Any]:
        """
        Get properties of a Metal device
        
        Args:
            device_idx: Device index
            
        Returns:
            Dictionary of device properties
        """
        # Check if device index is valid
        if device_idx < 0 or device_idx >= len(self.targets):
            return {}
        
        # Get device properties
        properties = {
            "name": f"Apple {hardware_capabilities.chip_generation.name} GPU",
            "max_threads_per_block": hardware_capabilities.max_threads_per_threadgroup,
            "max_blocks_per_grid": hardware_capabilities.max_threadgroups_per_grid,
            "shared_memory_per_block": hardware_capabilities.shared_memory_size,
            "simd_width": hardware_capabilities.simd_width,
            "unified_memory": True,
            "metal_gpu_family": hardware_capabilities.gpu_family,
            "metal_feature_set": hardware_capabilities.feature_set.name,
        }
        
        return properties
    
    def get_current_device(self) -> int:
        """
        Get current device index
        
        Returns:
            Current device index
        """
        return self.current_target_idx
    
    def synchronize(self):
        """Synchronize Metal device"""
        # In MLX, GPU operations are asynchronous, and we can call
        # mx.eval_async to ensure operations complete. However, there's
        # no direct synchronize method, so we create a dummy operation
        if self.mlx:
            try:
                # Create a small tensor and evaluate it to synchronize
                dummy = self.mlx.zeros(1)
                result = dummy + 0  # Dummy operation
                self.mlx.eval(result)  # Force evaluation
            except Exception as e:
                print(f"Error synchronizing device: {e}")
    
    def memory_info(self) -> Tuple[int, int]:
        """
        Get memory info for current device
        
        Returns:
            Tuple of (free memory, total memory) in bytes
        """
        # Metal doesn't provide direct memory info through MLX
        # For now, return conservative estimates based on device type
        
        # Determine approximate memory from chip generation
        # These are conservative estimates, actual values may be higher
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
            # M3 typically has 8-16GB unified memory, assume 8GB and 1/4 for GPU
            total_memory = 2 * 1024 * 1024 * 1024  # 2GB
        elif hardware_capabilities.chip_generation == AppleSiliconGeneration.M2:
            # M2 typically has 8-24GB unified memory, assume 8GB and 1/4 for GPU
            total_memory = 2 * 1024 * 1024 * 1024  # 2GB
        elif hardware_capabilities.chip_generation == AppleSiliconGeneration.M1:
            # M1 typically has 8-16GB unified memory, assume 8GB and 1/4 for GPU
            total_memory = 2 * 1024 * 1024 * 1024  # 2GB
        else:
            # Unknown device, assume 1GB
            total_memory = 1 * 1024 * 1024 * 1024  # 1GB
        
        # Assume 75% of total memory is free
        free_memory = int(0.75 * total_memory)
        
        return (free_memory, total_memory)
    
    def compute_capability(self) -> str:
        """
        Get NVIDIA compute capability equivalent
        
        This maps Apple Silicon generations to NVIDIA compute capability
        for compatibility with Triton's CUDA code paths.
        
        Returns:
            String representation of compute capability
        """
        # Map Apple Silicon generations to NVIDIA compute capabilities
        if hardware_capabilities.chip_generation == AppleSiliconGeneration.M3:
            return "8.6"  # Approximate to Ampere (SM86)
        elif hardware_capabilities.chip_generation == AppleSiliconGeneration.M2:
            return "8.0"  # Approximate to Ampere (SM80)
        elif hardware_capabilities.chip_generation == AppleSiliconGeneration.M1:
            return "7.5"  # Approximate to Turing (SM75)
        else:
            return "7.0"  # Default to Volta 