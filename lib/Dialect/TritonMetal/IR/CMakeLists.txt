add_triton_library(TritonMetalIR
  Dialect.cpp

  DEPENDS
  TritonMetalTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  TritonIR
) 