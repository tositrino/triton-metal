"""Metal driver for Triton

This module implements the driver interface for Metal backend on Apple Silicon GPUs.
It provides a bridge to the Metal device through MLX.
"""

import os
import sys
import ctypes
import importlib
from pathlib import Path
from typing import Dict, Union, Optional, List, Set, Callable, Any, Tuple

from triton.backends.driver import DriverBase, Benchmarker

# Add Metal package to path
metal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                          '..', '..', '..', '..', 
                                          'third_party', 'metal'))
if metal_path not in sys.path:
    sys.path.insert(0, metal_path)

class MetalDriver(DriverBase):
    """Triton driver for Metal backend on Apple Silicon GPUs"""
    
    def __init__(self):
        """Initialize the Metal driver, loading MLX and Metal dependencies"""
        super().__init__()
        
        # Import MLX and Metal-specific modules
        try:
            import mlx.core as mx
            self.mlx = mx
            
            # Import hardware capabilities
            try:
                from python.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
                self.hardware_capabilities = hardware_capabilities
            except ImportError:
                print("Warning: Could not import Metal hardware capabilities")
                self.hardware_capabilities = self._create_default_hardware_capabilities()
            
            # Set device properties
            self.metal_info = self._get_metal_device_info()
            
            # Set default device
            self._current_device = 0
            
            # Initialize benchmarker
            self._benchmarker = None
            
            # Flag to check if Metal is available
            self.is_available = True
            
        except ImportError as e:
            print(f"Warning: Failed to initialize Metal driver: {e}")
            self.mlx = None
            self.hardware_capabilities = self._create_default_hardware_capabilities()
            self.metal_info = self._get_default_metal_info()
            self.is_available = False
    
    def _create_default_hardware_capabilities(self) -> object:
        """Create a default hardware capabilities object when imports fail
        
        Returns:
            Default hardware capabilities object
        """
        class DefaultHardwareCapabilities:
            def __init__(self):
                self.shared_memory_size = 32768  # 32KB
                self.max_threads_per_threadgroup = 1024
                self.simd_width = 32
                self.chip_generation = type('', (), {'name': 'UNKNOWN'})()
        
        return DefaultHardwareCapabilities()
    
    def _get_default_metal_info(self) -> Dict[str, Any]:
        """Get default Metal device information when hardware detection fails
        
        Returns:
            Dictionary with default Metal device information
        """
        return {
            "name": "Apple Metal GPU",
            "device_count": 1,
            "compute_capability": "metal",
            "max_shared_memory": 32768,  # 32KB
            "max_threads_per_block": 1024,
            "warp_size": 32,
            "simd_width": 32,
            "chip_generation": "UNKNOWN",
        }
    
    def _get_metal_device_info(self) -> Dict[str, Any]:
        """Get Metal device information
        
        Returns:
            Dictionary with Metal device information
        """
        try:
            # Get hardware capabilities
            hc = self.hardware_capabilities
            
            info = {
                "name": "Apple Metal GPU",
                "device_count": 1,  # Metal presents a unified view of all GPUs
                "compute_capability": "metal",
                "max_shared_memory": hc.shared_memory_size,
                "max_threads_per_block": hc.max_threads_per_threadgroup,
                "warp_size": hc.simd_width,
                "simd_width": hc.simd_width,
                "chip_generation": hc.chip_generation.name,
            }
            
            # Add more hardware-specific details
            if hasattr(hc, "chip_model"):
                info["chip_model"] = hc.chip_model
                
            if hasattr(hc, "unified_memory_size"):
                info["unified_memory_size"] = hc.unified_memory_size
            
            return info
            
        except Exception as e:
            print(f"Warning: Failed to get Metal device info: {e}")
            return self._get_default_metal_info()
    
    @classmethod
    def is_active(cls) -> bool:
        """Check if this driver is active (Metal is available)
        
        Returns:
            True if Metal is available, False otherwise
        """
        # Check for metal capability
        try:
            import mlx.core as mx
            return True
        except ImportError:
            return False
    
    def get_current_target(self) -> Dict[str, Any]:
        """Get the current target device
        
        Returns:
            Dictionary with target information
        """
        return {
            "backend": "metal",
            "arch": self.metal_info.get("chip_generation", "apple-silicon"),
            "warp_size": self.metal_info.get("warp_size", 32),
        }
    
    def get_active_torch_device(self) -> Optional[str]:
        """Get the active PyTorch device, if available
        
        Returns:
            Active PyTorch device name or None if not available
        """
        # PyTorch may not be available or configured for Metal
        try:
            import torch
            return "mps" if torch.backends.mps.is_available() else None
        except (ImportError, AttributeError):
            return None
    
    def get_current_device(self) -> int:
        """Get the current device identifier
        
        Returns:
            Current device index
        """
        return self._current_device
    
    def set_current_device(self, device: int) -> None:
        """Set the current device
        
        Args:
            device: Device index (should be 0 for Metal)
        """
        if device != 0:
            raise ValueError("Metal backend only supports device index 0")
        self._current_device = device
    
    def get_current_stream(self, device: Optional[int] = None) -> int:
        """Get the current stream for the specified device
        
        Args:
            device: Device index (default: current device)
            
        Returns:
            Stream ID (always 0 for Metal backend)
        """
        return 0
    
    def get_driver_version(self) -> int:
        """Get the driver version
        
        Returns:
            Driver version as an integer
        """
        try:
            if hasattr(self.mlx, "__version__"):
                # Extract version components from MLX version (e.g., 0.3.0 -> 300)
                version_str = self.mlx.__version__
                components = version_str.split('.')
                if len(components) >= 3:
                    return int(components[0]) * 10000 + int(components[1]) * 100 + int(components[2])
            return 0
        except Exception:
            return 0
    
    def get_device_count(self) -> int:
        """Get the number of available devices
        
        Returns:
            Number of devices (always 1 for Metal backend)
        """
        return 1 if self.is_available else 0
    
    def get_device_properties(self, device: int) -> Dict[str, Any]:
        """Get properties of the specified device
        
        Args:
            device: Device index
            
        Returns:
            Dictionary of device properties
        """
        if device != 0:
            raise ValueError("Metal backend only supports device index 0")
        return self.metal_info
    
    def synchronize(self, device: Optional[int] = None) -> None:
        """Synchronize the specified device
        
        Args:
            device: Device index (default: current device)
        """
        # MLX operations are synchronized automatically
        pass
    
    def get_benchmarker(self) -> Benchmarker:
        """Get the benchmarking function for this backend
        
        Returns:
            Benchmarking function
        """
        if self._benchmarker is None:
            # Import benchmarker
            try:
                from python.metal_benchmarker import metal_benchmarker
                self._benchmarker = metal_benchmarker
            except ImportError:
                # Fallback to a simple benchmark implementation
                def default_benchmarker(kernel_call, *, quantiles=None, **kwargs):
                    """Simple benchmarker implementation"""
                    import time
                    if quantiles is None:
                        quantiles = [0.5, 0.2, 0.8]
                    
                    # Run the kernel a few times to warm up
                    for _ in range(3):
                        kernel_call()
                    
                    # Benchmark
                    times = []
                    for _ in range(10):
                        start = time.perf_counter()
                        kernel_call()
                        end = time.perf_counter()
                        times.append((end - start) * 1000)  # Convert to ms
                    
                    times.sort()
                    result = [times[int(q * len(times))] for q in quantiles]
                    return result
                
                self._benchmarker = default_benchmarker
        
        return self._benchmarker
    
    def load_binary(self, binary, name, device) -> int:
        """Load a compiled binary onto the device
        
        Args:
            binary: Compiled binary data
            name: Kernel name
            device: Device index
            
        Returns:
            Handle to the loaded binary
        """
        # The Metal driver will implement this later based on the binary format
        # For now, it's just a placeholder that returns a dummy handle
        return id(binary)

    def unload_binary(self, handle) -> None:
        """Unload a previously loaded binary
        
        Args:
            handle: Handle to the loaded binary
        """
        # No need to implement for now
        pass 