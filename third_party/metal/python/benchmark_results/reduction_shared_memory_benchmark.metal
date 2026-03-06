
#include <metal_stdlib>
using namespace metal;

// Benchmark kernel for shared memory reduction
kernel void benchmark_reduction_shared_memory(
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
    }
    
    // Execute reduction iterations
    for (uint iter = 0; iter < iterations; iter++) {
        // Load data to shared memory
        shared_data[tid] = input[gid];
        
        // Synchronize threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Reduction in shared memory
        for (uint stride = threads_per_threadgroup/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Write result
        if (tid == 0) {
            output[blockIdx] = shared_data[0];
        }
    }
    
    // End timing
    if (tid == 0 && blockIdx == 0) {
        uint end = as_type<uint>(input[1]);
        timing[0] = as_type<float>(end - start);
    }
}
