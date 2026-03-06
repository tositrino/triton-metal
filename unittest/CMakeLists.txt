add_subdirectory(Analysis)
add_subdirectory(Dialect)
add_subdirectory(Tools)

# Only build Metal backend tests on Apple platforms with Apple Silicon
if(APPLE)
  include(CheckSymbolExists)
  check_symbol_exists(TARGET_OS_MAC "TargetConditionals.h" MACOS)
  
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS "Building Metal backend unit tests for Apple Silicon")
    add_subdirectory(Metal)
  else()
    message(STATUS "Skipping Metal backend unit tests - requires Apple Silicon")
  endif()
endif()
