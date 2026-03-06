
#include <metal_stdlib>
using namespace metal;

// Benchmark kernel for barrier synchronization
kernel void benchmark_barrier(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uint& iterations [[buffer(2)]],
    device float* timing [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint blockIdx [[threadgroup_position_in_grid]]) {
    
    // Shared memory for the benchmark
    threadgroup float shared_data[256];
    
    // Start timing
    uint start = 0;
    if (tid == 0 && blockIdx == 0) {
        start = as_type<uint>(input[0]);
    }
    
    // Load data to shared memory
    shared_data[tid] = input[gid];
    
    // Execute barrier iterations
    for (uint i = 0; i < iterations; i++) {
        // Call barrier
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Do some work to prevent compiler optimizations
        if (tid < 128) {
            shared_data[tid] += shared_data[tid + 1];
        }
        
        // Call barrier again
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Do more work
        if (tid > 0) {
            shared_data[tid] += shared_data[tid - 1];
        }
    }
    
    // End timing
    if (tid == 0 && blockIdx == 0) {
        uint end = as_type<uint>(input[1]);
        timing[0] = as_type<float>(end - start);
    }
    
    // Write result to prevent work from being optimized away
    output[gid] = shared_data[tid];
}
