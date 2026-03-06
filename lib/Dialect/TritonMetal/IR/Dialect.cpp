#include "triton/Dialect/TritonMetal/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace triton {
namespace metal {

TritonMetalDialect::TritonMetalDialect(::mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<TritonMetalDialect>()) {
  // Initialize the dialect
  initialize();
}

void TritonMetalDialect::initialize() {
  // Add types
  // addTypes<...>();
  
  // Add operations
  // addOperations<...>();
}

} // namespace metal
} // namespace triton
} // namespace mlir 