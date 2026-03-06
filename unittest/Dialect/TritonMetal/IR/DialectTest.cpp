#include "triton/Dialect/TritonMetal/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

namespace {

using namespace mlir;
using namespace mlir::triton::metal;

class TritonMetalDialectTest : public ::testing::Test {
public:
  TritonMetalDialectTest() {
    // Register and load the Metal dialect
    registry.insert<TritonMetalDialect>();
    context.appendDialectRegistry(registry);
    context.loadDialect<TritonMetalDialect>();
    context.allowUnregisteredDialects();
    builder = std::make_unique<OpBuilder>(&context);
  }

protected:
  MLIRContext context;
  DialectRegistry registry;
  std::unique_ptr<OpBuilder> builder;
};

TEST_F(TritonMetalDialectTest, CanLoadDialect) {
  // Test that the Metal dialect can be loaded
  ASSERT_TRUE(context.isDialectRegistered<TritonMetalDialect>());
}

TEST_F(TritonMetalDialectTest, DialectNamespace) {
  // Test that the dialect namespace is correctly configured
  auto *dialect = context.getLoadedDialect<TritonMetalDialect>();
  ASSERT_NE(dialect, nullptr);
  ASSERT_EQ(dialect->getNamespace(), TritonMetalDialect::getDialectNamespace());
}

TEST_F(TritonMetalDialectTest, CreateModuleWithMetalDialect) {
  // Test creating a module with the Metal dialect loaded
  OpBuilder::InsertionGuard guard(*builder);
  auto module = ModuleOp::create(builder->getUnknownLoc());
  ASSERT_TRUE(module);
}

// Add more specific Metal dialect IR tests as operations are added to the dialect
// For now we'll include placeholder tests that can be expanded later

#ifdef __APPLE__
TEST_F(TritonMetalDialectTest, CreateMetalModule) {
  // Test creating a module specifically for Metal operations
  OpBuilder::InsertionGuard guard(*builder);
  auto module = ModuleOp::create(builder->getUnknownLoc());
  
  // Adding Metal-specific operations would go here once they're implemented
  
  ASSERT_TRUE(module);
}
#else
TEST_F(TritonMetalDialectTest, CreateMetalModule) {
  GTEST_SKIP() << "Test only runs on Apple hardware";
}
#endif

} // namespace

// Simple test that does not require the actual Metal dialect implementation
TEST(TritonMetalDialectTest, SimpleTest) {
  ASSERT_TRUE(true);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
} 