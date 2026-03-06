
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Benchmark kernel for hierarchical reduction
kernel void benchmark_reduction_hierarchical(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uint& iterations [[buffer(2)]],
    device float* timing [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint blockIdx [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
    
    // Shared memory for the benchmark
    threadgroup float shared_data[256];
    
    // Start timing
    uint start = 0;
    if (tid == 0 && blockIdx == 0) {
        start = as_type<uint>(input[0]);
        // Initialize output to 0
        output[0] = 0;
    }
    
    // Ensure initialization is complete
    threadgroup_barrier(mem_flags::mem_device);
    
    // Execute reduction iterations
    for (uint iter = 0; iter < iterations; iter++) {
        // Reset result for this iteration
        if (tid == 0 && blockIdx == 0) {
            output[0] = 0;
        }
        
        // Ensure reset is complete
        threadgroup_barrier(mem_flags::mem_device);
        
        // Load data to shared memory
        shared_data[tid] = input[gid];
        
        // Synchronize threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Local reduction in shared memory
        for (uint stride = threads_per_threadgroup/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Global reduction using atomics (only one thread per threadgroup)
        if (tid == 0) {
            atomic_fetch_add_explicit((_Atomic float*)&output[0], shared_data[0], memory_order_relaxed);
        }
    }
    
    // End timing
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0 && blockIdx == 0) {
        uint end = as_type<uint>(input[1]);
        timing[0] = as_type<float>(end - start);
    }
}
