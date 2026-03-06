set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(TritonMetalDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TritonMetalDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(TritonMetalTableGen) 