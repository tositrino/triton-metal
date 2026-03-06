"""
Test module for Metal synchronization primitives
"""

import unittest
import thread_mapping
import sync_converter

class TestSyncPrimitives(unittest.TestCase):
    """Test cases for synchronization primitives"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sync_primitives = thread_mapping.SyncPrimitives()
        self.converter = sync_converter.SyncPrimitivesConverter()
    
    def test_barrier_code_generation(self):
        """Test barrier code generation with various memory scopes"""
        # Test threadgroup memory scope
        barrier_code = self.sync_primitives.generate_barrier_code("threadgroup")
        self.assertEqual(barrier_code, "threadgroup_barrier(mem_flags::mem_threadgroup);")
        
        # Test device memory scope
        barrier_code = self.sync_primitives.generate_barrier_code("device")
        self.assertEqual(barrier_code, "threadgroup_barrier(mem_flags::mem_device);")
        
        # Test no memory scope
        barrier_code = self.sync_primitives.generate_barrier_code("none")
        self.assertEqual(barrier_code, "threadgroup_barrier(mem_flags::mem_none);")
    
    def test_warp_sync_code_generation(self):
        """Test SIMD group synchronization code generation"""
        # Test threadgroup memory scope
        warp_sync_code = self.sync_primitives.generate_warp_sync_code("threadgroup")
        self.assertEqual(warp_sync_code, "simdgroup_barrier(mem_flags::mem_threadgroup);")
        
        # Test device memory scope
        warp_sync_code = self.sync_primitives.generate_warp_sync_code("device")
        self.assertEqual(warp_sync_code, "simdgroup_barrier(mem_flags::mem_device);")
    
    def test_atomic_operations(self):
        """Test atomic operations code generation"""
        # Test atomic add
        add_code = self.sync_primitives.generate_atomic_add_code("float", "addr", "val")
        self.assertEqual(add_code, "atomic_fetch_add_explicit((_Atomic float*)&addr, val, memory_order_relaxed);")
        
        # Test atomic max for float
        max_code = self.sync_primitives.generate_atomic_max_code("float", "addr", "val")
        self.assertEqual(max_code, "mlx_atomic_fetch_max_explicit((device mlx_atomic<float>*)addr, val, 0);")
        
        # Test atomic exchange
        exchange_code = self.sync_primitives.generate_atomic_exchange_code("int", "addr", "val")
        self.assertEqual(exchange_code, "atomic_exchange_explicit((_Atomic int*)&addr, val, memory_order_relaxed);")
        
        # Test atomic CAS
        cas_code = self.sync_primitives.generate_atomic_cas_code("int", "addr", "exp", "des")
        self.assertEqual(cas_code, "atomic_compare_exchange_weak_explicit((_Atomic int*)&addr, &exp, des, memory_order_relaxed, memory_order_relaxed);")
    
    def test_converter_integration(self):
        """Test converter integration"""
        # Test debug barrier conversion
        barrier_code = self.converter.convert_debug_barrier()
        self.assertEqual(barrier_code, "threadgroup_barrier(mem_flags::mem_threadgroup);")
        
        # Test atomic add conversion
        add_code = self.converter.convert_atomic_add("int", "addr", "val")
        self.assertEqual(add_code, "atomic_fetch_add_explicit((_Atomic int*)&addr, val, memory_order_relaxed);")
        
        # Test general atomic operation mapping
        op_code = self.converter.map_atomic_operation("max", "int", "addr", "val")
        self.assertEqual(op_code, "atomic_fetch_max_explicit((_Atomic int*)&addr, val, memory_order_relaxed);")
        
        # Test CAS with expected value
        cas_code = self.converter.map_atomic_operation("cas", "int", "addr", "des", "exp")
        self.assertEqual(cas_code, "atomic_compare_exchange_weak_explicit((_Atomic int*)&addr, &exp, des, memory_order_relaxed, memory_order_relaxed);")

if __name__ == "__main__":
    unittest.main() 