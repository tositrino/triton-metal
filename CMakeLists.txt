cmake_minimum_required(VERSION 3.20)
cmake_policy(VERSION 3.20)
cmake_policy(SET CMP0116 OLD)
include(CMakeDependentOption)

project(triton LANGUAGES CXX C
               VERSION 2.0
               DESCRIPTION "A language and compiler for custom Deep Learning operations"
               HOMEPAGE_URL "https://github.com/chenxingqiang/triton-metal")

# Detect Apple Silicon
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(APPLE_SILICON TRUE)
else()
  set(APPLE_SILICON FALSE)
endif()

# Metal backend option
option(TRITON_ENABLE_METAL "Enable Metal backend" ${APPLE_SILICON})
option(TRITON_BUILD_TESTS "Build unit tests" ON)

# Find required packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Set up module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${LLVM_CMAKE_DIR})

# Include LLVM and MLIR CMake modules
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Find Python
find_package(Python 3.6 REQUIRED COMPONENTS Interpreter)

# Add MLIR includes
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Add Triton includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Include Google Test for unit tests
if(TRITON_BUILD_TESTS)
  include(unittest/googletest.cmake)
  enable_testing()
endif()

# Metal backend setup
if(TRITON_ENABLE_METAL)
  message(STATUS "Building Metal backend unit tests for Apple Silicon")
  include(cmake/AddTritonUnitTest.cmake)
  
  # Add run_metal_tests target
  add_executable(run_metal_tests run_metal_tests.cpp)
  target_include_directories(run_metal_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
  )
  
  # Link against GTest
  target_link_libraries(run_metal_tests PRIVATE
    gtest
    gtest_main
    TritonMetalIR
    TritonMetalTransforms
  )
endif()

# Add subdirectories
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)

# Add unit tests
if(TRITON_BUILD_TESTS)
  add_subdirectory(unittest)
endif()
