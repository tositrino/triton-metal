"""
Synchronization primitives converter for Metal backend

This module provides utilities for converting Triton synchronization primitives
to Metal synchronization primitives.
"""

from typing import Dict, Optional, Any
import metal_hardware_optimizer

class SyncPrimitivesConverter:
    """Converter for synchronization primitives in Metal"""
    
    def __init__(self, hardware_capabilities=None):
        """Initialize synchronization primitives converter
        
        Args:
            hardware_capabilities: Optional hardware capabilities instance
        """
        self.hardware = hardware_capabilities or getattr(metal_hardware_optimizer, 'hardware_capabilities', None)
    
    def convert_debug_barrier(self, memory_scope: str = "threadgroup") -> str:
        """Convert Triton debug_barrier to Metal barrier
        
        Args:
            memory_scope: Memory scope for synchronization
            
        Returns:
            Metal code for barrier synchronization
        """
        if memory_scope == "threadgroup":
            return "threadgroup_barrier(mem_flags::mem_threadgroup);"
        elif memory_scope == "device":
            return "threadgroup_barrier(mem_flags::mem_device);"
        else:
            return f"// Unsupported barrier scope: {memory_scope}"
    
    def map_atomic_operation(self, op_type: str, target_type: str, 
                             address: str, value: str, expected: Optional[str] = None) -> str:
        """Map Triton atomic operation to Metal atomic operation
        
        Args:
            op_type: Type of atomic operation
            target_type: Target data type
            address: Address expression
            value: Value expression
            expected: Expected value expression (only for CAS)
            
        Returns:
            Metal code for atomic operation
        """
        # Convert Triton type to Metal type
        metal_type = self._convert_type(target_type)
        
        # Map atomic operation
        if op_type == "add":
            return f"atomic_fetch_add_explicit({address}, {value}, memory_order_relaxed);"
        
        elif op_type == "max":
            if target_type.startswith("float"):
                # For floating-point, use atomic_compare_exchange in a loop
                return f"""
                {metal_type} old_val = *{address};
                {metal_type} new_val;
                do {{
                    new_val = max(old_val, {value});
                    if (new_val == old_val) break;
                }} while (!atomic_compare_exchange_weak_explicit(
                    {address}, &old_val, new_val, 
                    memory_order_relaxed, memory_order_relaxed));
                """
            else:
                # For integers, use atomic_fetch_max
                return f"atomic_fetch_max_explicit({address}, {value}, memory_order_relaxed);"
        
        elif op_type == "min":
            if target_type.startswith("float"):
                # For floating-point, use atomic_compare_exchange in a loop
                return f"""
                {metal_type} old_val = *{address};
                {metal_type} new_val;
                do {{
                    new_val = min(old_val, {value});
                    if (new_val == old_val) break;
                }} while (!atomic_compare_exchange_weak_explicit(
                    {address}, &old_val, new_val, 
                    memory_order_relaxed, memory_order_relaxed));
                """
            else:
                # For integers, use atomic_fetch_min
                return f"atomic_fetch_min_explicit({address}, {value}, memory_order_relaxed);"
        
        elif op_type == "xchg":
            return f"atomic_exchange_explicit({address}, {value}, memory_order_relaxed);"
        
        elif op_type == "cas":
            if expected is None:
                return f"// Error: CAS operation requires 'expected' parameter"
            return f"atomic_compare_exchange_weak_explicit({address}, &{expected}, {value}, memory_order_relaxed, memory_order_relaxed);"
        
        elif op_type == "and":
            return f"atomic_fetch_and_explicit({address}, {value}, memory_order_relaxed);"
        
        elif op_type == "or":
            return f"atomic_fetch_or_explicit({address}, {value}, memory_order_relaxed);"
        
        elif op_type == "xor":
            return f"atomic_fetch_xor_explicit({address}, {value}, memory_order_relaxed);"
        
        else:
            return f"// Unsupported atomic operation: {op_type}"
    
    def _convert_type(self, triton_type: str) -> str:
        """Convert Triton type to Metal type
        
        Args:
            triton_type: Triton type
            
        Returns:
            Metal type
        """
        if triton_type == "float" or triton_type == "float32":
            return "float"
        elif triton_type == "float16":
            return "half"
        elif triton_type == "int" or triton_type == "int32":
            return "int"
        elif triton_type == "int16":
            return "short"
        elif triton_type == "int8":
            return "char"
        elif triton_type == "uint" or triton_type == "uint32":
            return "uint"
        elif triton_type == "uint16":
            return "ushort"
        elif triton_type == "uint8":
            return "uchar"
        elif triton_type == "bool":
            return "bool"
        else:
            return "/* Unknown type: " + triton_type + " */ float"
    
    def generate_sync_primitive_declarations(self) -> str:
        """Generate declarations for synchronization primitives
        
        Returns:
            Metal code for synchronization primitive declarations
        """
        # This would include any declarations needed in the Metal shader
        return """
        // Metal synchronization primitives
        using namespace metal;
        
        // Memory flags for barriers
        namespace mem_flags {
            constexpr auto mem_none = memory_order_relaxed;
            constexpr auto mem_threadgroup = memory_order_relaxed;
            constexpr auto mem_device = memory_order_relaxed;
            constexpr auto mem_texture = memory_order_relaxed;
        }
        """
    
    def convert_sync_operation(self, op: Dict[str, Any]) -> str:
        """Convert a synchronization operation
        
        Args:
            op: Synchronization operation dictionary
            
        Returns:
            Metal code for the synchronization operation
        """
        op_type = op.get("type")
        
        if op_type == "barrier":
            memory_scope = op.get("memory_scope", "threadgroup")
            return self.convert_debug_barrier(memory_scope)
        
        elif op_type == "atomic":
            atomic_op = {
                "op_type": op.get("op_type", "add"),
                "target_type": op.get("target_type", "float"),
                "address": op.get("address", "address"),
                "value": op.get("value", "value"),
                "expected": op.get("expected")
            }
            return self.map_atomic_operation(
                atomic_op["op_type"], atomic_op["target_type"],
                atomic_op["address"], atomic_op["value"], atomic_op["expected"])
        
        else:
            return f"// Unsupported sync operation: {op_type}"
    
    def generate_atomic_wrapper_functions(self) -> str:
        """Generate wrapper functions for atomic operations
        
        Returns:
            Metal code for atomic wrapper functions
        """
        # Generate wrapper functions for common atomic operations to make them easier to use
        return """
        // Atomic wrapper functions
        template <typename T>
        T atomic_add(device T* address, T value) {
            return atomic_fetch_add_explicit(address, value, memory_order_relaxed);
        }
        
        template <typename T>
        T atomic_max(device T* address, T value) {
            return atomic_fetch_max_explicit(address, value, memory_order_relaxed);
        }
        
        template <typename T>
        T atomic_min(device T* address, T value) {
            return atomic_fetch_min_explicit(address, value, memory_order_relaxed);
        }
        
        template <typename T>
        T atomic_exchange(device T* address, T value) {
            return atomic_exchange_explicit(address, value, memory_order_relaxed);
        }
        
        // Special case for float atomics
        float atomic_add(device float* address, float value) {
            uint* address_as_uint = (device uint*)address;
            uint old = *address_as_uint;
            uint assumed;
            do {
                assumed = old;
                old = atomic_compare_exchange_weak_explicit(
                    address_as_uint, 
                    &assumed, 
                    as_type<uint>(as_type<float>(assumed) + value),
                    memory_order_relaxed, 
                    memory_order_relaxed);
            } while (assumed != old);
            return as_type<float>(old);
        }
        
        // Utility function for float atomic max
        float atomic_max(device float* address, float value) {
            float old_val = *address;
            float new_val;
            do {
                new_val = max(old_val, value);
                if (new_val == old_val) break;
            } while (!atomic_compare_exchange_weak_explicit(
                (device uint*)address, 
                (device uint*)&old_val, 
                as_type<uint>(new_val), 
                memory_order_relaxed, 
                memory_order_relaxed));
            return old_val;
        }
        
        // Utility function for float atomic min
        float atomic_min(device float* address, float value) {
            float old_val = *address;
            float new_val;
            do {
                new_val = min(old_val, value);
                if (new_val == old_val) break;
            } while (!atomic_compare_exchange_weak_explicit(
                (device uint*)address, 
                (device uint*)&old_val, 
                as_type<uint>(new_val), 
                memory_order_relaxed, 
                memory_order_relaxed));
            return old_val;
        }
        """

# Create global instance for convenience
sync_converter = SyncPrimitivesConverter() 