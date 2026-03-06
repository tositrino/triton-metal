/**
 * @file metal_api.h
 * @brief Metal API integration for Triton
 * 
 * This file provides the C++ interface for integrating with Metal APIs
 * in the Triton compiler.
 */

#ifndef TRITON_METAL_API_H
#define TRITON_METAL_API_H

#ifdef __OBJC__
#include <Metal/Metal.h>
#endif

#include <cstdint>
#include <string>
#include <vector>

namespace triton {
namespace metal {

/**
 * @brief Structure representing Metal device properties
 */
struct MetalDeviceProperties {
    std::string name;           // Device name
    int32_t max_threads_per_threadgroup;  // Maximum threads per threadgroup
    int32_t simd_width;         // SIMD width
    int32_t shared_memory_size; // Shared memory size in bytes
    std::string chip_generation; // Chip generation (e.g., "M1", "M2")
};

/**
 * @brief Get properties of the current Metal device
 * 
 * @return Metal device properties
 */
MetalDeviceProperties get_metal_device_properties();

/**
 * @brief Compile Metal shader source to metallib binary
 * 
 * @param source Metal shader source code
 * @param name Function name
 * @param options Compilation options
 * @return Compiled binary data or empty vector on failure
 */
std::vector<uint8_t> compile_metal_shader(const std::string& source, 
                                         const std::string& name,
                                         const std::string& options = "");

/**
 * @brief Load a Metal binary onto the device
 * 
 * @param binary Binary data
 * @param name Kernel name
 * @return Handle to the loaded binary or 0 on failure
 */
uint64_t load_metal_binary(const std::vector<uint8_t>& binary, 
                         const std::string& name);

/**
 * @brief Unload a previously loaded Metal binary
 * 
 * @param handle Handle to the loaded binary
 */
void unload_metal_binary(uint64_t handle);

} // namespace metal
} // namespace triton

#endif // TRITON_METAL_API_H 