"""
Test module for Metal synchronization primitives edge cases
"""

import unittest
import os
import sys
import itertools
import thread_mapping
import sync_converter
import metal_hardware_optimizer

class TestSyncPrimitivesEdgeCases(unittest.TestCase):
    """Test cases for synchronization primitives edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sync_primitives = thread_mapping.SyncPrimitives()
        self.converter = sync_converter.SyncPrimitivesConverter()
        self.shared_memory = thread_mapping.SharedMemory()
        self.hardware = metal_hardware_optimizer.hardware_capabilities
    
    def test_barrier_with_different_memory_scopes(self):
        """Test barrier with all possible memory scopes"""
        memory_scopes = ["threadgroup", "device", "none", "simdgroup"]
        for scope in memory_scopes:
            barrier_code = self.sync_primitives.generate_barrier_code(scope)
            self.assertTrue("threadgroup_barrier" in barrier_code)
            expected_scope = self.sync_primitives.memory_scope_map.get(scope, "mem_flags::mem_threadgroup")
            self.assertTrue(expected_scope in barrier_code)
    
    def test_atomic_operations_with_unsupported_types(self):
        """Test atomic operations with unsupported types"""
        unsupported_types = ["double", "char"]
        
        for op_type, data_type in itertools.product(["add", "max", "min"], unsupported_types):
            with self.assertRaises(ValueError):
                self.converter.map_atomic_operation(op_type, data_type, "addr", "val")
    
    def test_atomic_operations_with_expressions(self):
        """Test atomic operations with complex expressions"""
        # Test with complex expressions
        expressions = [
            "ptr[i * stride + j]",
            "output_ptr[blockIdx.x * blockDim.x + threadIdx.x]",
            "*reinterpret_cast<float*>(&shared_mem[offset])",
            "result[threadIdx.x + threadIdx.y * blockDim.x]"
        ]
        
        for expr in expressions:
            code = self.converter.convert_atomic_add("float", expr, "1.0f")
            self.assertTrue(expr in code)
            
            code = self.converter.convert_atomic_max("int", expr, "val")
            self.assertTrue(expr in code)
    
    def test_shared_memory_named_allocations(self):
        """Test named allocations in shared memory"""
        # Clear previous allocations
        self.shared_memory.clear()
        
        # Allocate arrays with names
        offset1 = self.shared_memory.allocate_array("float", 32, name="data1")
        offset2 = self.shared_memory.allocate_array("int", 64, name="data2")
        offset3 = self.shared_memory.allocate(128, name="raw_data")
        
        # Verify offsets are correct and increasing
        self.assertEqual(offset1, 0)
        self.assertTrue(offset2 > offset1)
        self.assertTrue(offset3 > offset2)
        
        # Test named access generation
        access1 = self.shared_memory.generate_named_access("data1", 5)
        self.assertTrue("float" in access1)
        self.assertTrue("shared_memory" in access1)
        
        access2 = self.shared_memory.generate_named_access("data2")
        self.assertTrue("int" in access2)
        self.assertTrue("shared_memory" in access2)
        
        # Test named declarations
        declarations = self.shared_memory.generate_named_declarations()
        self.assertTrue("data1" in declarations)
        self.assertTrue("data2" in declarations)
        self.assertTrue("raw_data_OFFSET" in declarations)
    
    def test_shared_memory_array_bound_checking(self):
        """Test array bound checking in shared memory access"""
        # Clear previous allocations
        self.shared_memory.clear()
        
        # Enable debug mode for bounds checking
        self.shared_memory.set_debug_mode(True)
        
        # Allocate array
        self.shared_memory.allocate_array("float", 32, name="data")
        
        # Test valid index
        valid_access = self.shared_memory.generate_named_access("data", 31)
        self.assertTrue("float" in valid_access)
        
        # Test invalid index
        with self.assertRaises(ValueError):
            self.shared_memory.generate_named_access("data", 32)
            
        with self.assertRaises(ValueError):
            self.shared_memory.generate_named_access("data", -1)
        
        # Test dynamic index with bounds checking
        dynamic_access = self.shared_memory.generate_named_access("data", "tid")
        self.assertTrue("Debug: runtime bounds checking" in dynamic_access)
        self.assertTrue("assert" in dynamic_access)
    
    def test_non_array_access_checking(self):
        """Test non-array access checking"""
        # Clear previous allocations
        self.shared_memory.clear()
        
        # Allocate raw memory
        self.shared_memory.allocate(128, name="raw_data")
        
        # Test indexed access to non-array allocation
        with self.assertRaises(ValueError):
            self.shared_memory.generate_named_access("raw_data", 0)
    
    def test_hardware_specific_optimizations(self):
        """Test hardware-specific optimizations"""
        # Test optimal work group size
        work_group_size = self.converter.get_optimal_work_group_size()
        self.assertEqual(len(work_group_size), 3)
        
        # Test optimal reduction strategy
        small_strategy = self.converter.get_optimal_reduction_strategy(512)
        self.assertEqual(small_strategy, "shared_memory")
        
        large_strategy = self.converter.get_optimal_reduction_strategy(10000)
        self.assertIn(large_strategy, ["direct_atomic", "hierarchical"])
        
        # Test optimized reduction code generation
        reduction_small = self.converter.generate_optimized_reduction_code(512)
        self.assertEqual(reduction_small["strategy"], "shared_memory")
        self.assertTrue("threadgroup" in reduction_small["code"])
        
        reduction_large = self.converter.generate_optimized_reduction_code(10000)
        self.assertIn(reduction_large["strategy"], ["direct_atomic", "hierarchical"])
    
    def test_dynamic_index_array_access(self):
        """Test dynamic index array access in shared memory"""
        # Clear previous allocations
        self.shared_memory.clear()
        
        # Allocate array
        self.shared_memory.allocate_array("float", 32, name="data")
        
        # Test dynamic index
        dynamic_access = self.shared_memory.generate_named_access("data", "tid")
        self.assertTrue("float" in dynamic_access)
        self.assertTrue("tid" in dynamic_access)
        
        # Test complex expression
        complex_access = self.shared_memory.generate_named_access("data", "base_idx + offset * stride")
        self.assertTrue("float" in complex_access)
        self.assertTrue("base_idx + offset * stride" in complex_access)
    
    def test_alignment_padding(self):
        """Test memory alignment padding"""
        # Clear previous allocations
        self.shared_memory.clear()
        
        # Allocate with different alignments
        offset1 = self.shared_memory.allocate(10, alignment=16)
        self.assertEqual(offset1, 0)
        
        offset2 = self.shared_memory.allocate(5, alignment=32)
        # Check if padding was applied
        self.assertTrue(offset2 % 32 == 0)
        self.assertTrue(offset2 >= 16)
        
        offset3 = self.shared_memory.allocate(7, alignment=64)
        # Check if padding was applied
        self.assertTrue(offset3 % 64 == 0)
        self.assertTrue(offset3 >= 32)
    
    def test_matrix_allocation_and_access(self):
        """Test matrix allocation and access in shared memory"""
        # Clear previous allocations
        self.shared_memory.clear()
        
        # Allocate matrix with row-major layout
        rows, cols = 4, 8
        self.shared_memory.allocate_matrix("float", rows, cols, name="matrix_rm", row_major=True)
        
        # Allocate matrix with column-major layout
        self.shared_memory.allocate_matrix("int", rows, cols, name="matrix_cm", row_major=False)
        
        # Test row-major matrix access with static indices
        rm_access = self.shared_memory.generate_named_access("matrix_rm", (2, 3))
        self.assertTrue("float" in rm_access)
        
        # Test column-major matrix access with static indices
        cm_access = self.shared_memory.generate_named_access("matrix_cm", (1, 2))
        self.assertTrue("int" in cm_access)
        
        # Test row-major matrix access with dynamic indices
        rm_dynamic = self.shared_memory.generate_named_access("matrix_rm", ("row", "col"))
        self.assertTrue("row" in rm_dynamic)
        self.assertTrue("col" in rm_dynamic)
        
        # Test access validation
        self.shared_memory.set_debug_mode(True)
        with self.assertRaises(ValueError):
            self.shared_memory.generate_named_access("matrix_rm", (rows, 0))
            
        with self.assertRaises(ValueError):
            self.shared_memory.generate_named_access("matrix_rm", (0, cols))
    
    def test_half_precision_support(self):
        """Test half-precision support in atomic operations"""
        # Check if half-precision is supported
        supported_types = self.converter.get_supported_atomic_types()
        self.assertIn("half", supported_types)
        
        # Test half-precision atomic add
        half_add = self.converter.convert_atomic_add("half", "addr", "val")
        self.assertTrue("float temp_val" in half_add)
        
        # Test half-precision atomic max
        half_max = self.converter.convert_atomic_max("half", "addr", "val")
        self.assertTrue("float temp_val" in half_max)
        
        # Test half-precision CAS
        half_cas = self.converter.convert_atomic_cas("half", "addr", "expected", "desired")
        self.assertTrue("float expected_val" in half_cas)
        self.assertTrue("float desired_val" in half_cas)
    
    def test_simdgroup_reduction(self):
        """Test SIMD group reduction operations"""
        # Test float addition reduction
        float_add = self.converter.convert_simdgroup_reduction("float", "add", "value")
        self.assertTrue("simd_sum(value)" in float_add)
        
        # Test int max reduction
        int_max = self.converter.convert_simdgroup_reduction("int", "max", "value")
        self.assertTrue("simd_max(value)" in int_max)
        
        # Test invalid reduction op
        with self.assertRaises(ValueError):
            self.converter.convert_simdgroup_reduction("float", "invalid_op", "value")
    
    def test_threadgroup_reduction(self):
        """Test threadgroup reduction operations"""
        # Test float addition reduction
        float_add = self.converter.convert_threadgroup_reduction("float", "add", "value")
        
        # Test int max reduction
        int_max = self.converter.convert_threadgroup_reduction("int", "max", "value")
        
        # For M3 chips, should use optimized reduction
        if self.hardware.chip_generation.value >= metal_hardware_optimizer.AppleSiliconGeneration.M3.value:
            self.assertTrue("threadgroup_reduction_add" in float_add or "simd_sum" in float_add)
    
    def test_memory_order_support(self):
        """Test memory ordering support in atomic operations"""
        # Test relaxed memory ordering
        relaxed = self.converter.convert_atomic_add("int", "addr", "val", "relaxed")
        self.assertTrue("memory_order_relaxed" in relaxed)
        
        # Test acquire memory ordering
        acquire = self.converter.convert_atomic_add("int", "addr", "val", "acquire")
        self.assertTrue("memory_order_acquire" in acquire)
        
        # Test release memory ordering
        release = self.converter.convert_atomic_add("int", "addr", "val", "release")
        self.assertTrue("memory_order_release" in release)
        
        # Test acquire-release memory ordering
        acq_rel = self.converter.convert_atomic_add("int", "addr", "val", "acq_rel")
        self.assertTrue("memory_order_acq_rel" in acq_rel)
    
    def test_memcpy_generation(self):
        """Test memory copy code generation"""
        # Test aligned memcpy
        aligned = self.converter.convert_memcpy("dst", "src", "size", True)
        self.assertTrue("memcpy" in aligned)
        
        # Test unaligned memcpy
        unaligned = self.converter.convert_memcpy("dst", "src", "size", False)
        self.assertTrue("for" in unaligned)  # Should use a loop for safety
    
    def test_optimized_barrier_code(self):
        """Test optimized barrier code generation"""
        # Test single barrier
        single = self.converter.generate_optimized_barrier_code(1, True)
        self.assertTrue("threadgroup_barrier" in single)
        
        # Test multiple barriers
        multiple = self.converter.generate_optimized_barrier_code(3, True)
        
        # Count number of barriers
        if self.hardware.chip_generation.value >= metal_hardware_optimizer.AppleSiliconGeneration.M3.value:
            # On M3, should have multiple fine-grained barriers
            self.assertEqual(multiple.count("threadgroup_barrier"), 3)
        else:
            # On M1/M2, should use a single device barrier
            self.assertEqual(multiple.count("threadgroup_barrier"), 1)
            self.assertTrue("mem_flags::mem_device" in multiple)
    
    def test_shared_memory_declarations(self):
        """Test shared memory declarations generation"""
        # Clear previous allocations
        self.shared_memory.clear()
        
        # Allocate various types of memory
        self.shared_memory.allocate_array("float", 32, name="float_array")
        self.shared_memory.allocate_matrix("int", 4, 8, name="int_matrix")
        self.shared_memory.allocate(128, name="raw_buffer")
        
        # Generate declarations
        declarations = self.shared_memory.generate_declaration()
        named_decl = self.shared_memory.generate_named_declarations()
        
        # Check that declarations include all allocations
        self.assertTrue("threadgroup char shared_memory" in declarations)
        self.assertTrue("float_array" in named_decl)
        self.assertTrue("int_matrix" in named_decl)
        self.assertTrue("raw_buffer_OFFSET" in named_decl)
        
        # Test with debug mode on
        self.shared_memory.set_debug_mode(True)
        debug_decl = self.shared_memory.generate_named_declarations()
        self.assertTrue("float_array_SIZE" in debug_decl)
        self.assertTrue("int_matrix_ROWS" in debug_decl)
        self.assertTrue("int_matrix_COLS" in debug_decl)
    
    def test_shared_memory_via_converter(self):
        """Test shared memory management through SyncPrimitivesConverter"""
        # Clear any existing shared memory state
        self.shared_memory.clear()
        
        # Allocate arrays and matrices through converter
        self.converter.allocate_shared_array("float", 64, "data")
        self.converter.allocate_shared_matrix("int", 8, 16, "matrix")
        
        # Test access generation
        array_access = self.converter.generate_shared_memory_access("data", 10)
        self.assertTrue("float" in array_access)
        
        matrix_access = self.converter.generate_shared_memory_access("matrix", (3, 4))
        self.assertTrue("int" in matrix_access)
        
        # Test shared memory declarations
        declarations = self.converter.generate_shared_memory_declarations()
        self.assertTrue("threadgroup char shared_memory" in declarations)
        self.assertTrue("data" in declarations)
        self.assertTrue("matrix" in declarations)

if __name__ == "__main__":
    unittest.main() 