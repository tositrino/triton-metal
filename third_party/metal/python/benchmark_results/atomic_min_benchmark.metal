
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Benchmark kernel for atomic min operation
kernel void benchmark_atomic_min(
    device float* result [[buffer(0)]],
    device int* result_int [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const uint& iterations [[buffer(3)]],
    device float* timing [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]) {
    
    // Start timing
    uint start = 0;
    if (gid == 0) {
        start = as_type<uint>(input[0]);
    }
    
    // Execute atomic operation iterations
    float val = input[gid % 1024];
    for (uint i = 0; i < iterations; i++) {
        // Call atomic operation
        atomic_fetch_min_explicit((_Atomic int*)&result_int[0], as_type<int>(val), memory_order_relaxed);
    }
    
    // End timing
    threadgroup_barrier(mem_flags::mem_none);
    if (gid == 0) {
        uint end = as_type<uint>(input[1]);
        timing[0] = as_type<float>(end - start);
    }
}
