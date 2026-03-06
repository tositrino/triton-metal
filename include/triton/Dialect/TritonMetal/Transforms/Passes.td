#ifndef TRITON_DIALECT_TRITONMETAL_TRANSFORMS_PASSES
#define TRITON_DIALECT_TRITONMETAL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def M3MemoryOptimizationPass : Pass<"triton-metal-m3-memory-optimization", "mlir::ModuleOp"> {
  let summary = "Optimize memory layout for M3 chips";
  let description = [{
    This pass optimizes memory access patterns for Apple M3 chips, leveraging the 64KB shared memory
    and improving access patterns for better performance.
  }];
}

def M3VectorizationPass : Pass<"triton-metal-m3-vectorization", "mlir::ModuleOp"> {
  let summary = "Vectorize operations for M3 chips";
  let description = [{
    This pass vectorizes operations to utilize the 8-wide SIMD capabilities of M3 chips.
  }];
}

def M3SIMDOptimizationPass : Pass<"triton-metal-m3-simd-optimization", "mlir::ModuleOp"> {
  let summary = "Optimize SIMD operations for M3 chips";
  let description = [{
    This pass optimizes operations to take advantage of the 32-wide SIMD groups in M3 chips.
  }];
}

def TritonToMLXPass : Pass<"triton-to-mlx", "mlir::ModuleOp"> {
  let summary = "Convert Triton dialect to MLX operations";
  let description = [{
    This pass converts Triton operations to their MLX equivalents for execution on Metal.
  }];
}

#endif // TRITON_DIALECT_TRITONMETAL_TRANSFORMS_PASSES 