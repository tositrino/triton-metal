#ifndef TRITON_DIALECT_TRITONMETAL_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONMETAL_IR_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

// Forward declarations
namespace mlir {
namespace triton {
namespace metal {

class TritonMetalDialect;

} // namespace metal
} // namespace triton
} // namespace mlir

#include "triton/Dialect/TritonMetal/IR/TritonMetalDialect.h.inc"

namespace mlir {
namespace triton {
namespace metal {

class TritonMetalDialect : public ::mlir::Dialect {
public:
  explicit TritonMetalDialect(::mlir::MLIRContext *context);
  static ::llvm::StringRef getDialectNamespace() { return "tt.metal"; }

  void initialize();
};

} // namespace metal
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONMETAL_IR_DIALECT_H_ 