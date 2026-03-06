add_triton_ut(
  NAME TestTritonMetalTransforms
  SRCS TransformsTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
)

add_triton_ut(
  NAME TestTritonMetalMemoryOptimizer
  SRCS MemoryOptimizerTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
)

add_triton_ut(
  NAME TestTritonMetalM3Optimizations
  SRCS M3OptimizationsTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
) 