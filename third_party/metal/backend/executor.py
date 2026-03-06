"""
Metal Kernel Executor for Triton

This module provides the executor for running Triton kernels on Apple Silicon GPUs
through MLX and Metal.
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
try:
    from python.metal_hardware_optimizer import hardware_capabilities, AppleSiliconGeneration
    from python.operation_mapping import MLXDispatcher, OpCategory, op_conversion_registry
    from python.metal_fusion_optimizer import FusionOptimizer, fusion_optimizer
    from .driver import MetalDriver
except ImportError as e:
    print(f"Error importing required modules: {e}")

# Global executor instance
_EXECUTOR = None

def get_executor():
    """
    Get the global executor instance
    
    Returns:
        Global executor instance
    """
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = MetalKernelExecutor()
    return _EXECUTOR

class KernelInfo:
    """
    Information about a loaded kernel
    """
    
    def __init__(self, kernel_id: str, function_dict: Dict[str, Any], metadata: Dict[str, Any]):
        """
        Initialize kernel info
        
        Args:
            kernel_id: Kernel ID
            function_dict: Dictionary containing the function and its operations
            metadata: Kernel metadata
        """
        self.kernel_id = kernel_id
        self.function = function_dict
        self.metadata = metadata
        self.last_used = time.time()
    
    def update_last_used(self):
        """Update last used timestamp"""
        self.last_used = time.time()

class MetalKernelExecutor:
    """
    Metal kernel executor for Triton
    
    This class handles the execution of compiled Triton kernels on
    Apple Silicon GPUs using MLX.
    """
    
    def __init__(self):
        """Initialize Metal kernel executor"""
        self.driver = MetalDriver()
        self.dispatcher = MLXDispatcher()
        self.fusion_optimizer = fusion_optimizer
        self.conversion_registry = op_conversion_registry
        
        # Try to load MLX
        try:
            import mlx.core as mx
            self.mx = mx
        except ImportError:
            print("MLX not found. Please install with 'pip install mlx'")
            self.mx = None
        
        # Dictionary of loaded kernels
        self._kernels = {}
    
    def _parse_binary(self, binary: bytes) -> Dict[str, Any]:
        """
        Parse kernel binary
        
        Args:
            binary: Kernel binary
            
        Returns:
            Parsed kernel dictionary
        """
        try:
            # Parse the binary, which is expected to be a JSON string
            kernel_dict = json.loads(binary.decode())
            
            # Make sure we have required fields
            if "version" not in kernel_dict:
                kernel_dict["version"] = "1.0"
            
            if "operations" not in kernel_dict:
                kernel_dict["operations"] = []
            
            if "metal_code" not in kernel_dict:
                kernel_dict["metal_code"] = ""
            
            if "metadata" not in kernel_dict:
                kernel_dict["metadata"] = {}
            
            return kernel_dict
        except Exception as e:
            print(f"Error parsing kernel binary: {e}")
            # Return a minimal valid kernel
            return {
                "version": "1.0",
                "operations": [],
                "metal_code": "",
                "metadata": {},
            }
    
    def load_kernel(self, binary: bytes, metadata: Dict[str, Any]) -> str:
        """
        Load a kernel
        
        Args:
            binary: Kernel binary
            metadata: Kernel metadata
            
        Returns:
            Kernel ID
        """
        # Parse the binary
        kernel_dict = self._parse_binary(binary)
        
        # Merge metadata
        kernel_dict["metadata"].update(metadata)
        
        # Optimize operations if available
        if "operations" in kernel_dict and kernel_dict["operations"]:
            kernel_dict["operations"] = self.fusion_optimizer.optimize(kernel_dict["operations"])
        
        # Create a unique kernel ID
        kernel_id = f"kernel_{len(self._kernels)}_{int(time.time())}"
        
        # Store the kernel
        self._kernels[kernel_id] = KernelInfo(kernel_id, kernel_dict, metadata)
        
        return kernel_id
    
    def get_kernel(self, kernel_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a loaded kernel
        
        Args:
            kernel_id: Kernel ID
            
        Returns:
            Kernel info or None if not found
        """
        if kernel_id not in self._kernels:
            return None
        
        # Update last used timestamp
        self._kernels[kernel_id].update_last_used()
        
        return {
            "function": self._kernels[kernel_id].function,
            "metadata": self._kernels[kernel_id].metadata,
            "last_used": self._kernels[kernel_id].last_used,
        }
    
    def unload_kernel(self, kernel_id: str) -> bool:
        """
        Unload a kernel
        
        Args:
            kernel_id: Kernel ID
            
        Returns:
            True if successful, False otherwise
        """
        if kernel_id in self._kernels:
            del self._kernels[kernel_id]
            return True
        
        return False
    
    def _prepare_args(self, kernel_info: Dict[str, Any], args: List[Any]) -> List[Any]:
        """
        Prepare arguments for kernel execution
        
        Args:
            kernel_info: Kernel info
            args: List of arguments
            
        Returns:
            Prepared arguments
        """
        prepared_args = []
        
        for arg in args:
            if isinstance(arg, np.ndarray):
                # Convert NumPy arrays to MLX arrays
                prepared_args.append(self.mx.array(arg))
            elif isinstance(arg, (int, float, bool)):
                # Pass scalars directly
                prepared_args.append(arg)
            elif hasattr(arg, "__array__"):
                # Convert array-like objects
                prepared_args.append(self.mx.array(arg.__array__()))
            else:
                # Pass other objects directly
                prepared_args.append(arg)
        
        return prepared_args
    
    def execute_kernel(self, kernel_id: str, args: List[Any], grid: Tuple[int, ...]) -> None:
        """
        Execute a kernel
        
        Args:
            kernel_id: Kernel ID
            args: Kernel arguments
            grid: Grid dimensions
        """
        # Get kernel info
        kernel_info = self.get_kernel(kernel_id)
        if not kernel_info:
            raise ValueError(f"Kernel not found: {kernel_id}")
        
        # Prepare arguments
        prepared_args = self._prepare_args(kernel_info, args)
        
        # Execute kernel
        self._execute_operations(kernel_info["function"], prepared_args, grid)
    
    def _execute_operations(self, kernel_dict: Dict[str, Any], args: List[Any], grid: Tuple[int, ...]) -> None:
        """
        Execute operations in a kernel
        
        Args:
            kernel_dict: Kernel dictionary
            args: Kernel arguments
            grid: Grid dimensions
        """
        # Create execution context
        context = {}
        
        # Add arguments to context
        for i, arg in enumerate(args):
            context[f"arg_{i}"] = arg
        
        # Add grid dimensions to context
        for i, dim in enumerate(grid):
            context[f"grid_{i}"] = dim
        
        # Execute operations
        operations = kernel_dict.get("operations", [])
        
        if not operations:
            # If no operations, try to execute the Metal code directly
            self._execute_metal_code(kernel_dict.get("metal_code", ""), context)
            return
        
        # Execute each operation
        for op in operations:
            # Get operation type
            op_type = op.get("type", "")
            
            # Check if it's a fused operation
            if "fused" in op_type:
                # Execute fused operation
                result = self.fusion_optimizer.execute_fused_op(op, context)
            else:
                # Get converter for the operation
                converter = self.conversion_registry.get_converter(op_type)
                
                if converter is None:
                    print(f"Warning: No converter found for operation: {op_type}")
                    continue
                
                # Execute the operation
                result = converter(op, context)
            
            # Store the result in the context
            if "id" in op:
                context[op["id"]] = result
        
        # Synchronize device
        self.driver.synchronize()
    
    def _execute_metal_code(self, metal_code: str, context: Dict[str, Any]) -> None:
        """
        Execute Metal shader code directly
        
        Args:
            metal_code: Metal shader code
            context: Execution context
        """
        # This is a placeholder. A proper implementation would compile and execute
        # the Metal shader code directly. For now, we just print a warning.
        print("Warning: Direct Metal code execution not implemented")
        
        # Synchronize device
        self.driver.synchronize() 