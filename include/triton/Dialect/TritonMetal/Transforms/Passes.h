#ifndef TRITON_DIALECT_TRITONMETAL_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONMETAL_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonMetal/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace metal {

std::unique_ptr<Pass> createM3MemoryOptimizationPass();
std::unique_ptr<Pass> createM3VectorizationPass();
std::unique_ptr<Pass> createM3SIMDOptimizationPass();
std::unique_ptr<Pass> createTritonToMLXPass();

/// Register all Metal dialect passes.
void registerPasses();

} // namespace metal
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONMETAL_TRANSFORMS_PASSES_H_ 