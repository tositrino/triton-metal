// This is a stub implementation for the TableGen-generated file
// It will be replaced by the actual TableGen output during the build process

namespace mlir {
namespace triton {
namespace metal {
namespace impl {

// Base classes for the passes
template <typename DerivedT>
class M3MemoryOptimizationPassBase : public ::mlir::PassWrapper<DerivedT, ::mlir::OperationPass<::mlir::ModuleOp>> {
public:
  M3MemoryOptimizationPassBase() = default;
  M3MemoryOptimizationPassBase(const M3MemoryOptimizationPassBase &) = default;

  static llvm::StringRef getArgument() { return "triton-metal-m3-memory-optimization"; }
  static llvm::StringRef getDescription() { return "Optimize memory layout for M3 chips"; }
};

template <typename DerivedT>
class M3VectorizationPassBase : public ::mlir::PassWrapper<DerivedT, ::mlir::OperationPass<::mlir::ModuleOp>> {
public:
  M3VectorizationPassBase() = default;
  M3VectorizationPassBase(const M3VectorizationPassBase &) = default;

  static llvm::StringRef getArgument() { return "triton-metal-m3-vectorization"; }
  static llvm::StringRef getDescription() { return "Vectorize operations for M3 chips"; }
};

template <typename DerivedT>
class M3SIMDOptimizationPassBase : public ::mlir::PassWrapper<DerivedT, ::mlir::OperationPass<::mlir::ModuleOp>> {
public:
  M3SIMDOptimizationPassBase() = default;
  M3SIMDOptimizationPassBase(const M3SIMDOptimizationPassBase &) = default;

  static llvm::StringRef getArgument() { return "triton-metal-m3-simd-optimization"; }
  static llvm::StringRef getDescription() { return "Optimize SIMD operations for M3 chips"; }
};

template <typename DerivedT>
class TritonToMLXPassBase : public ::mlir::PassWrapper<DerivedT, ::mlir::OperationPass<::mlir::ModuleOp>> {
public:
  TritonToMLXPassBase() = default;
  TritonToMLXPassBase(const TritonToMLXPassBase &) = default;

  static llvm::StringRef getArgument() { return "triton-to-mlx"; }
  static llvm::StringRef getDescription() { return "Convert Triton dialect to MLX operations"; }
};

} // namespace impl
} // namespace metal
} // namespace triton
} // namespace mlir 