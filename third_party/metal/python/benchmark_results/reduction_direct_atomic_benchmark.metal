
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Benchmark kernel for direct atomic reduction
kernel void benchmark_reduction_direct_atomic(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uint& iterations [[buffer(2)]],
    device float* timing [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    
    // Start timing
    uint start = 0;
    if (gid == 0) {
        start = as_type<uint>(input[0]);
        // Initialize output to 0
        output[0] = 0;
    }
    
    // Ensure initialization is complete
    threadgroup_barrier(mem_flags::mem_device);
    
    // Execute reduction iterations
    for (uint iter = 0; iter < iterations; iter++) {
        // Reset result for this iteration
        if (gid == 0) {
            output[0] = 0;
        }
        
        // Ensure reset is complete
        threadgroup_barrier(mem_flags::mem_device);
        
        // Direct atomic reduction
        float val = input[gid];
        atomic_fetch_add_explicit((_Atomic float*)&output[0], val, memory_order_relaxed);
    }
    
    // End timing
    threadgroup_barrier(mem_flags::mem_device);
    if (gid == 0) {
        uint end = as_type<uint>(input[1]);
        timing[0] = as_type<float>(end - start);
    }
}
