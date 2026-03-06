add_subdirectory(IR)
add_subdirectory(Transforms)

add_triton_ut(
  NAME TestTritonMetalMLXIntegration
  SRCS MLXIntegrationTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
)

add_triton_ut(
  NAME TestTritonMetalHardwareDetection
  SRCS HardwareDetectionTest.cpp
  LIBS TritonMetalIR TritonMetalTransforms
) 