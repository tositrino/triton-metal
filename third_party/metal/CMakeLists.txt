# Metal后端CMake配置

cmake_minimum_required(VERSION 3.20)

# Metal backend for Triton
set(triton_BACKEND_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Check for macOS with Metal support
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS "Configuring Metal backend for Apple Silicon")
    set(TRITON_ENABLE_METAL_BACKEND ON CACHE BOOL "Enable Metal backend for Apple Silicon")
else()
    message(STATUS "Metal backend requires Apple Silicon Mac")
    set(TRITON_ENABLE_METAL_BACKEND OFF CACHE BOOL "Enable Metal backend for Apple Silicon")
    return()
endif()

# Check for MLX installation
find_package(Python3 COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import mlx; print(mlx.__version__)"
    RESULT_VARIABLE MLX_FOUND
    OUTPUT_VARIABLE MLX_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(MLX_FOUND EQUAL 0)
    message(STATUS "Found MLX version: ${MLX_VERSION}")
else()
    message(WARNING "MLX not found. Metal backend requires MLX to be installed.")
    message(WARNING "Install with: pip install mlx")
endif()

# Metal backend directories
set(METAL_BACKEND_DIRS
    ${triton_BACKEND_ROOT}/backend
    ${triton_BACKEND_ROOT}/include
    ${triton_BACKEND_ROOT}/language
    ${triton_BACKEND_ROOT}/python
)

# Add subdirectories
add_subdirectory(python)

# MLX backend library
add_library(triton_backend STATIC
    backend/driver.py
    backend/mlx_backend.py
    backend/executor.py
    python/metal_hardware_optimizer.py
    python/operation_mapping.py
    python/metal_fusion_optimizer.py
)

set_target_properties(triton_backend PROPERTIES
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
)

# Add dependencies
target_link_libraries(triton_backend PRIVATE
    triton-shared
)

# Include directories
target_include_directories(triton_backend PUBLIC
    ${triton_BACKEND_ROOT}
    ${triton_BACKEND_ROOT}/include
)

# Add install target
install(
    TARGETS triton_backend
    EXPORT triton-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install Python modules
install(
    DIRECTORY ${triton_BACKEND_ROOT}/python/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/python/triton/backends/metal
    PATTERN "*.py"
)

# Enable Metal backend in Triton
message(STATUS "Metal backend enabled")
add_definitions(-DTRITON_ENABLE_METAL_BACKEND=1) 