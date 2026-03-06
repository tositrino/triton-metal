cmake_minimum_required(VERSION 3.20)

# Python modules for Metal backend
set(METAL_PYTHON_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Install Python modules
file(GLOB METAL_PYTHON_FILES "*.py")
file(GLOB METAL_BENCHMARK_FILES "benchmark/*.py")

# Copy Python modules to build directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/triton/backends/metal)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/triton/backends/metal/benchmark)

foreach(file ${METAL_PYTHON_FILES})
    get_filename_component(filename ${file} NAME)
    configure_file(${file} ${CMAKE_CURRENT_BINARY_DIR}/triton/backends/metal/${filename} COPYONLY)
endforeach()

foreach(file ${METAL_BENCHMARK_FILES})
    get_filename_component(filename ${file} NAME)
    configure_file(${file} ${CMAKE_CURRENT_BINARY_DIR}/triton/backends/metal/benchmark/${filename} COPYONLY)
endforeach()

# Create __init__.py files if they don't exist
if(NOT EXISTS ${METAL_PYTHON_DIR}/__init__.py)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/triton/backends/metal/__init__.py "# Metal backend for Triton\n")
endif()

if(NOT EXISTS ${METAL_PYTHON_DIR}/benchmark/__init__.py)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/triton/backends/metal/benchmark/__init__.py "# Metal backend benchmarks\n")
endif()

# Install Python modules
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/triton/backends/metal
        DESTINATION ${CMAKE_INSTALL_PREFIX}/python/triton/backends
        FILES_MATCHING PATTERN "*.py")

# Create Python tests target
add_custom_target(metal_python_tests
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR} -p "test_*.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running Metal backend Python tests"
    DEPENDS triton_backend
)

# Add test to CTest
add_test(NAME metal_python_tests
    COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR} -p "test_*.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
) 