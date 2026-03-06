add_triton_ut(
  NAME TestMetalBackend
  SRCS MetalBackendTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
)

add_triton_ut(
  NAME TestM3Optimizations
  SRCS M3OptimizationsTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
)

add_triton_ut(
  NAME TestMetalMemoryManager
  SRCS MetalMemoryManagerTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
)

add_triton_ut(
  NAME TestOperationFusion
  SRCS OperationFusionTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
)

add_triton_ut(
  NAME TestHardwareDetection
  SRCS HardwareDetectionTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
)

add_triton_ut(
  NAME TestMLXIntegration
  SRCS MLXIntegrationTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
)

add_triton_ut(
  NAME TestTensorCore
  SRCS TensorCoreTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
) 