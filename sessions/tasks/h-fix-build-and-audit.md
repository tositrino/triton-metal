---
name: h-fix-build-and-audit
branch: fix/build-and-audit
status: pending
created: 2026-03-06
---

# Fix Build (Clang/Tahoe) and Code Hygiene Audit

## Problem/Goal
The project has stopped building due to what appears to be a clang/Tahoe (macOS) related version mismatch or API change. The primary goal is to diagnose and fix the build break while preferring the standard system clang/LLVM installed with Xcode/macOS rather than pulling in specific versions unless Triton absolutely requires them.

While investigating the build, perform a single-pass audit of code hygiene: test coverage inventory, code smells, duplicate files, and areas needing refactoring. Anything requiring substantial work should be documented as future tasks rather than addressed inline.

## Success Criteria
- [ ] Project builds successfully on macOS Tahoe with system clang/LLVM
- [ ] Third-party dependency usage is minimized — no unnecessary version pinning
- [ ] Code hygiene audit completed with findings documented
- [ ] Future tasks created for any substantial refactoring or cleanup work identified

## Context Manifest
<!-- Added by context-gathering agent -->

### How the Build System Currently Works: Full Architecture

This project is a fork of OpenAI's Triton compiler, extended with a Metal backend for Apple Silicon GPUs. The build system has a dual-layer architecture: a Python-driven build via `setup.py` (which orchestrates CMake), and a standalone CMake build for the C++ portions. Understanding both layers is essential because they are currently **out of sync**, which is a root cause of build failures.

#### Layer 1: The Python Build (setup.py + pyproject.toml)

The primary build entry point is `pip install -e . --no-build-isolation -v` (or `make dev-install`). This invokes `setup.py`, which defines a custom `CMakeBuild` class inheriting from `setuptools.command.build_ext`. The build flow is:

1. `setup.py` imports `python/build_helpers.py` via a `sys.path.insert(0, os.path.dirname(__file__))` hack at line 44. The `build_helpers.py` module provides `get_base_dir()` (project root) and `get_cmake_dir()` (build output directory at `build/cmake.{platform}-{python_version}`).

2. Before CMake runs, `setup.py` calls `download_and_copy_dependencies()` which attempts to download NVIDIA toolchain binaries (ptxas, cuobjdump, nvdisasm, cudart, cupti). On macOS, this function maps Darwin to "linux" in its platform lookup (`supported = {"Linux": "linux", "Darwin": "linux"}` at line 294), which means it tries to download Linux binaries for macOS. The `open_url()` function has been monkey-patched to skip downloads and print warnings instead (lines 204-210), so this does not actually fail -- it just produces confusing output and creates empty/dummy paths.

3. The `CMakeBuild.build_extension()` method then invokes CMake with these critical flags:
   - `-G Ninja` (requires ninja)
   - `-DTRITON_BUILD_PYTHON_MODULE=ON`
   - `-DTRITON_CODEGEN_BACKENDS=nvidia;amd;metal` (list of backends to build)
   - `-DTRITON_PLUGIN_DIRS=...` (for external plugins)
   - `-DLLVM_ENABLE_WERROR=ON` (fatal warnings -- important for build breaks)
   - Plus pybind11 paths and third-party package paths

4. The build expects the top-level `CMakeLists.txt` to process `TRITON_CODEGEN_BACKENDS` and `TRITON_BUILD_PYTHON_MODULE`, iterate over backend subdirectories, build the `libtriton` pybind11 module (from `python/src/main.cc`), and link everything together. **However, the current top-level `CMakeLists.txt` has been rewritten and does NOT process these variables at all.** This is a fundamental disconnect.

#### Layer 2: The CMake Build (CMakeLists.txt)

The top-level `CMakeLists.txt` is a stripped-down version that:
- Calls `find_package(LLVM REQUIRED CONFIG)` and `find_package(MLIR REQUIRED CONFIG)` -- requiring pre-installed LLVM/MLIR with CMake config files
- Includes LLVM/MLIR CMake modules (TableGen, AddLLVM, AddMLIR)
- Adds subdirectories: `include/`, `lib/`, `test/`, `unittest/`
- Has a `TRITON_ENABLE_METAL` option that attempts to build a `run_metal_tests` target, but **the referenced file `run_metal_tests.cpp` does not exist** in the repository
- Uses GoogleTest fetched from GitHub (release-1.12.1) via `FetchContent`

The C++ libraries under `lib/` use `add_triton_library()` and `add_triton_plugin()` functions that are **not defined anywhere in this repository**. These functions come from the upstream Triton's CMake infrastructure which has been stripped out. This means the CMake build cannot work standalone either.

#### LLVM/MLIR Dependency: The Core Problem

The project has multiple conflicting strategies for finding LLVM/MLIR:

**Strategy A (CMakeLists.txt):** Uses `find_package(LLVM REQUIRED CONFIG)` and `find_package(MLIR REQUIRED CONFIG)`. This requires LLVM/MLIR to be installed with CMake config files (LLVMConfig.cmake, MLIRConfig.cmake). The system Xcode clang does NOT ship these -- only Homebrew LLVM does.

**Strategy B (cmake/FindLLVM.cmake):** A custom FindLLVM module that uses `llvm-config` to discover LLVM. This searches for llvm-config in various paths including Homebrew and MacPorts locations. However, this module is **never actually used** because the top-level CMakeLists.txt uses `find_package(LLVM REQUIRED CONFIG)` which uses CMake's config-mode search, not the custom Find module.

**Strategy C (setup.py):** The `get_llvm_package_info()` function in setup.py returns a dummy package. It checks for `LLVM_DIR` environment variable and passes `LLVM_INCLUDE_DIRS` and `LLVM_LIBRARY_DIR` to CMake. The Makefile target `dev-install-llvm` references a `scripts/build-llvm-project.sh` script that **does not exist** in the repository.

**Strategy D (third_party/llvm/dummy_llvm/):** A dummy LLVM installation with fake headers, a fake `llvm-config` binary, and a fake `libLLVM.so`. The dummy `llvm-config` contains the text "This is a dummy LLVM binary" -- it is not executable/functional. This appears to be a placeholder that was never completed.

#### Current System Environment (macOS Tahoe 26.3)

There are three clang installations on this system:

1. **Xcode/Apple Clang 17.0.0** at `/usr/bin/clang` -- This is the standard system compiler. It does NOT include LLVM/MLIR CMake config files. Apple Clang is a separate distribution from LLVM upstream and intentionally does not expose LLVM/MLIR as a library.

2. **Custom Clang 16.0.0** at `/usr/local/bin/clang` -- Built from LLVM commit `124f90bd89b97066e01274a9bba1068f3a175d66`. This is likely a previous local build. No LLVM/MLIR CMake config files were found at `/usr/local/lib/cmake/`.

3. **Homebrew LLVM 21.1.8** at `/opt/homebrew/opt/llvm/` -- This includes full MLIR support with CMake config files at `/opt/homebrew/opt/llvm/lib/cmake/mlir/`. This is currently the only viable LLVM/MLIR installation on the system that can satisfy `find_package(LLVM REQUIRED CONFIG)` and `find_package(MLIR REQUIRED CONFIG)`.

The `cmake/llvm-hash.txt` file records commit `092b6e73e651469527662443b592f98f442ece72`, which is the upstream LLVM commit this fork was built against. The Homebrew LLVM (21.1.8) is much newer, which could cause API incompatibilities in the MLIR C++ code.

#### Version Discrepancy Between Build Configs

- `setup.py` declares version `3.3.0rc1`
- `pyproject.toml` declares version `3.3.0rc2`
- `pyproject.toml` lists `requires-python = ">=3.9,<3.14"` but `setup.py` computes it as `>=3.9,<3.14` from MIN/MAX constants
- Both files duplicate metadata (author, classifiers, dependencies, entry points) with slight differences (e.g., different email addresses: `chenxingqiang@gmail.com` in setup.py vs `chenxingqiang@turingai.cc` in pyproject.toml)
- `pyproject.toml` has `[tool.setuptools] packages = ["triton"]` and `package-dir = {"" = "python"}` which conflicts with `setup.py`'s dynamic package discovery

### Metal Backend Architecture

The Metal backend exists in two places with significant duplication:

**Location 1: `third_party/metal/`** -- The "source" backend
- `backend/` contains `compiler.py`, `driver.py`, `executor.py`, `mlx_backend.py`, `__init__.py`
- `python/` contains a large tree of Python modules: MLX integration (27 files in `python/MLX/`), M3-specific optimizations (4 files in `python/M3/`), tests (50+ files in `python/tests/`), benchmarks, examples, tutorials, tools, docs
- `language/` contains a `metal/` subdirectory (not yet explored in detail)
- `CMakeLists.txt` has errors: it tries to create a static library from `.py` files (`add_library(triton_backend STATIC backend/driver.py backend/mlx_backend.py ...)` at line 47), which is nonsensical and will cause CMake errors

**Location 2: `python/triton/backends/metal/`** -- The "installed" backend
- Contains a separate `compiler.py` and `driver.py` that act as thin wrappers/bridges
- Both files use `sys.path.insert` to add `third_party/metal` to the Python path
- The `__init__.py` imports `MetalBackend` from `compiler` and `MetalDriver` from `driver`

The nvidia and amd backends follow a different pattern: their source lives in `third_party/{nvidia,amd}/backend/` with `compiler.py` and `driver.py`, and setup.py copies/symlinks them to `python/triton/backends/{nvidia,amd}/` during install. The Metal backend appears to have been created with BOTH a copy under `third_party/metal/backend/` (following the pattern) AND a manually-written bridge under `python/triton/backends/metal/` (breaking the pattern).

### C++ MLIR Dialect for Metal

The TritonMetal MLIR dialect is defined across:
- `include/triton/Dialect/TritonMetal/IR/` -- TableGen definitions (Dialect.td), generated inc files (already checked in as `.h.inc` and `.cpp.inc`), and Dialect.h
- `include/triton/Dialect/TritonMetal/Transforms/` -- Pass definitions in Passes.td (4 passes: M3MemoryOptimization, M3Vectorization, M3SIMDOptimization, TritonToMLX), generated Passes.h.inc, and Passes.h
- `lib/Dialect/TritonMetal/IR/Dialect.cpp` -- Dialect implementation
- `lib/Dialect/TritonMetal/Transforms/Passes.cpp` -- Pass implementations

These files use `add_triton_library()` which is undefined, so they cannot currently build. They also depend on MLIR infrastructure (MLIRIR, MLIRPass, MLIRSupport, TritonIR) which requires a working LLVM/MLIR installation.

Notably, the TritonMetal dialect is NOT registered in `bin/RegisterTritonDialects.h` -- only Triton, TritonGPU, TritonNvidiaGPU, NVGPU, NVWS, TritonAMDGPU, and Proton dialects are registered. This means even if the TritonMetal C++ code compiled, it would not be loaded into the `triton-opt` tool.

### Code Hygiene Issues Identified

**Duplicate / "copy" files:**
- `third_party/metal/python/tests/test_special_ops copy.py` and `test_special_ops copy 2.py`
- `third_party/metal/python/tests/test_m3_optimizations copy.py`
- `third_party/metal/python/tests/test_m3_integration copy.py`
- `third_party/metal/python/tests/test_translation copy.py`
- `third_party/metal/python/docs/M3_OPTIMIZATIONS copy.md`

**Empty test files:**
- `third_party/metal/python/tests/test_m3_optimizations.py` -- 0 bytes
- `third_party/metal/python/tests/test_translation.py` -- 0 bytes

**Build artifacts checked into source:**
- `.DS_Store` files in `third_party/metal/` and `third_party/metal/python/`
- `__pycache__/` directories with `.pyc` files throughout `third_party/metal/python/`
- `python/__pycache__/build_helpers.cpython-313.pyc`
- Benchmark result images (`.png`) in `third_party/metal/python/tests/test_results/` and `third_party/metal/python/benchmark/plots/`
- `third_party/metal/python/benchmark_results/` with `.metal` shader files and `.txt` results

**sys.path hacks (over 30 instances):**
The Metal backend relies extensively on `sys.path.insert()` and `sys.path.append()` to find modules. Key examples:
- `third_party/metal/backend/compiler.py` has 3 separate sys.path hacks (lines 18, 127, 170)
- `third_party/metal/backend/driver.py` inserts parent directory into sys.path
- `python/triton/backends/metal/compiler.py` and `driver.py` both insert `third_party/metal` into sys.path
- Nearly every test file and tool in `third_party/metal/python/` has its own sys.path hack

**Missing files referenced by build system:**
- `run_metal_tests.cpp` -- referenced in top-level `CMakeLists.txt` line 57, does not exist
- `scripts/build-llvm-project.sh` -- referenced in `Makefile` line 92, does not exist

**Nonsensical CMake configurations:**
- `third_party/metal/CMakeLists.txt` tries to create a static C++ library from `.py` files (line 47)
- `third_party/metal/CMakeLists.txt` links against `triton-shared` which is not defined anywhere

**Incomplete .gitignore:**
The `.gitignore` only covers cc-sessions runtime files. It is missing standard entries for:
- `__pycache__/`, `*.pyc`, `*.pyo`
- `.DS_Store`
- `build/`, `dist/`, `*.egg-info`
- `.venv/`
- `*.so`, `*.dylib`

### Test Infrastructure

**C++ Unit Tests (unittest/):**
- `unittest/Analysis/` -- UtilityTest.cpp
- `unittest/Dialect/TritonGPU/` -- DialectTest, DumpLayoutTest, LinearLayoutConversionsTest, SwizzleTest
- `unittest/Dialect/TritonMetal/` -- MLXIntegrationTest, HardwareDetectionTest, IR/DialectTest, Transforms/M3OptimizationsTest, MemoryOptimizerTest, TransformsTest
- `unittest/Metal/` -- MetalBackendTest, M3OptimizationsTest, MetalMemoryManagerTest, OperationFusionTest, HardwareDetectionTest, MLXIntegrationTest, TensorCoreTest
- `unittest/Tools/` -- LayoutUtilsTest, LinearLayoutTest

There is significant duplication between `unittest/Dialect/TritonMetal/` and `unittest/Metal/` -- both contain HardwareDetectionTest, M3OptimizationsTest, and MLXIntegrationTest with similar names but different file sizes. The `unittest/Metal/CMakeLists.txt` references `TritonMetalIR` and `TritonMetalTransforms` libraries which use the undefined `add_triton_ut` function (defined in `cmake/AddTritonUnitTest.cmake` which depends on MLIR/LLVM being available).

**MLIR Lit Tests (test/):**
- Standard Triton lit tests for Analysis, Conversion, TritonGPU, TritonNvidiaGPU
- No Metal-specific lit tests exist yet
- Uses `lit.cfg.py` with `triton-opt` binary

**Python Tests:**
- `python/test/unit/` -- standard Triton unit tests (language, runtime, cuda, etc.)
- `python/test/regression/` -- regression tests
- `third_party/metal/python/tests/` -- 50+ Metal-specific Python tests (this is a large and sprawling test suite)

GoogleTest is fetched via FetchContent from `https://github.com/google/googletest.git` (release-1.12.1) during CMake configuration.

### Technical Reference Details

#### Build Entry Points

```
# Primary build command
pip install -e . --no-build-isolation -v

# Makefile shortcuts
make dev-install        # pip install + requirements
make all               # incremental ninja build
make test              # run all tests (lit + cpp + python)
```

#### Critical Environment Variables

- `LLVM_DIR` -- Path to LLVM CMake config directory
- `LLVM_INCLUDE_DIRS` / `LLVM_LIBRARY_DIR` / `LLVM_SYSPATH` -- Manual LLVM paths
- `TRITON_BUILD_WITH_METAL` -- Force Metal backend (ON/OFF)
- `TRITON_BUILD_WITH_CLANG_LLD` -- Use specific clang/lld
- `TRITON_OFFLINE_BUILD` -- Disable downloads
- `TRITON_PLUGIN_DIRS` -- External plugin directories
- `TRITON_APPEND_CMAKE_ARGS` -- Additional CMake arguments
- `DEBUG` / `REL_WITH_DEB_INFO` -- Build type control
- `MAX_JOBS` -- Parallel build jobs (default: 2 * CPU count)

#### Key File Paths

- Top-level CMake: `/Volumes/Big Data/triton_metal/CMakeLists.txt`
- Setup.py: `/Volumes/Big Data/triton_metal/setup.py`
- pyproject.toml: `/Volumes/Big Data/triton_metal/pyproject.toml`
- Makefile: `/Volumes/Big Data/triton_metal/Makefile`
- Build helpers: `/Volumes/Big Data/triton_metal/python/build_helpers.py`
- Custom FindLLVM: `/Volumes/Big Data/triton_metal/cmake/FindLLVM.cmake`
- LLVM hash: `/Volumes/Big Data/triton_metal/cmake/llvm-hash.txt`
- Metal backend source: `/Volumes/Big Data/triton_metal/third_party/metal/`
- Metal backend bridge: `/Volumes/Big Data/triton_metal/python/triton/backends/metal/`
- TritonMetal dialect: `/Volumes/Big Data/triton_metal/include/triton/Dialect/TritonMetal/` and `/Volumes/Big Data/triton_metal/lib/Dialect/TritonMetal/`
- Dummy LLVM: `/Volumes/Big Data/triton_metal/third_party/llvm/dummy_llvm/`
- Nvidia backend (reference): `/Volumes/Big Data/triton_metal/third_party/nvidia/`
- GTest cmake: `/Volumes/Big Data/triton_metal/unittest/googletest.cmake`
- Nvidia toolchain versions: `/Volumes/Big Data/triton_metal/cmake/nvidia-toolchain-version.json`

#### Homebrew LLVM/MLIR (Viable Installation)

```
Path: /opt/homebrew/opt/llvm/ (version 21.1.8)
CMake configs: /opt/homebrew/opt/llvm/lib/cmake/llvm/LLVMConfig.cmake
               /opt/homebrew/opt/llvm/lib/cmake/mlir/MLIRConfig.cmake
Include dirs:  /opt/homebrew/opt/llvm/include/
Library dirs:  /opt/homebrew/opt/llvm/lib/
```

To make CMake find these: `cmake -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir ...`

#### Undefined CMake Functions That Block the Build

1. `add_triton_library()` -- used by 28 CMakeLists.txt files across lib/ and third_party/
2. `add_triton_plugin()` -- used by nvidia, amd, and proton CMakeLists.txt files
3. These come from upstream Triton's CMake infrastructure that was not carried over to this fork

### Recommended Investigation Order for the Build Fix

1. **Decide the build scope**: Does the project need to build the full C++ Triton compiler (MLIR dialects, triton-opt, libtriton pybind module), or is it primarily a Python-only Metal backend that delegates to MLX? If Python-only, the entire CMake/C++ build can be bypassed.

2. **If C++ build is needed**: The `add_triton_library` and `add_triton_plugin` functions must be defined or obtained from upstream Triton. The top-level `CMakeLists.txt` must be aligned with what `setup.py` expects (processing `TRITON_CODEGEN_BACKENDS`, `TRITON_BUILD_PYTHON_MODULE`, building `libtriton` from `python/src/main.cc`).

3. **LLVM/MLIR version**: Either pin to Homebrew LLVM 21.1.8 (and update C++ code for API changes) or build LLVM from the pinned commit `092b6e73e651469527662443b592f98f442ece72` (and create the missing `scripts/build-llvm-project.sh`).

4. **Metal CMakeLists.txt**: The `third_party/metal/CMakeLists.txt` needs to be rewritten -- creating a C++ static library from `.py` files is nonsensical. If the Metal backend is purely Python, the CMakeLists.txt should only handle file installation, not library creation.

5. **Cleanup**: Address the missing `run_metal_tests.cpp`, duplicate files, build artifacts, `.gitignore`, and the version mismatch between `setup.py` and `pyproject.toml`.

## User Notes
- Prefer system clang/LLVM (Xcode toolchain) unless Triton absolutely requires a specific version
- This is a single pass — fix the build first, audit along the way
- Spin off future tasks for anything that needs deeper work

## Work Log
<!-- Updated as work progresses -->
