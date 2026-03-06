add_triton_library(TritonMetalTransforms
  Passes.cpp

  DEPENDS
  TritonMetalTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  TritonIR
  TritonMetalIR
) 