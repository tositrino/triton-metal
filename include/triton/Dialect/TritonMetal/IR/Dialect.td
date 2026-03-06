#ifndef TRITON_DIALECT_TRITONMETAL_IR_DIALECT
#define TRITON_DIALECT_TRITONMETAL_IR_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def TritonMetal_Dialect : Dialect {
  let name = "tt.metal";
  let cppNamespace = "::mlir::triton::metal";
  let summary = "Metal backend for Triton on Apple Silicon (M1/M2/M3) GPUs";

  let description = [{
    The Metal dialect includes Metal-specific operations, types, and optimization passes
    for targeting Apple Silicon GPUs with the Metal API.

    This dialect provides M3-specific optimizations, including support for:
    - 64KB shared memory (vs 32KB on M1/M2)
    - 8-wide vectorization
    - 32-wide SIMD groups
    - Tensor cores and hardware ray tracing
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // TRITON_DIALECT_TRITONMETAL_IR_DIALECT 