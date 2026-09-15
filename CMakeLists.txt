cmake_minimum_required(VERSION 3.24)

project(unlimited_ocr_engine
    VERSION 0.1.0
    DESCRIPTION "Native C++17/CUDA inference engine for Baidu Unlimited-OCR (R-SWA + MoE INT4)"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(ENGINE_BACKEND "CPU" CACHE STRING "Compute backend: CPU or CUDA")
set_property(CACHE ENGINE_BACKEND PROPERTY STRINGS CPU CUDA)

option(ENGINE_BUILD_TESTS "Build unit tests" ON)
option(ENGINE_BUILD_BENCHMARKS "Build benchmarks" ON)
option(ENGINE_BUILD_TOOLS "Build helper tools (weight converter)" ON)
option(ENGINE_ENABLE_SANITIZERS "Build with ASan/UBSan (CPU only)" OFF)
option(ENGINE_NATIVE "Compile CPU host code with -march=native" ON)

# ---------------------------------------------------------------------------
# Language / flags
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(cmake/CUDAArch.cmake)

# Common warning set. Kept conservative so that third-party headers do not
# break the build.
set(ENGINE_WARN_FLAGS -Wall -Wextra -Wno-unused-parameter)

add_library(uocr_warnings INTERFACE)
target_compile_options(uocr_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${ENGINE_WARN_FLAGS}>)
if(ENGINE_NATIVE AND NOT ENGINE_ENABLE_SANITIZERS)
    target_compile_options(uocr_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# nlohmann/json is used for config.json / tokenizer config parsing.
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json CMake package not found, falling back to header include")
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE /usr/include)
endif()

# Threads (scheduler / host runtime)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library (runtime + scheduler + engine + CPU kernels)
# ---------------------------------------------------------------------------
add_library(uocr_core
    src/kernels/cpu_ops.cpp
    src/runtime/config.cpp
    src/runtime/log.cpp
    src/runtime/safetensors.cpp
    src/runtime/tokenizer.cpp
    src/runtime/quant.cpp
    src/runtime/weights.cpp
    src/scheduler/kv_cache.cpp
    src/scheduler/block_manager.cpp
    src/scheduler/memory_pool.cpp
    src/scheduler/continuous_batch.cpp
    src/engine/sampler.cpp
    src/engine/deep_encoder.cpp
    src/engine/moe_decoder.cpp
    src/engine/engine.cpp)

target_include_directories(uocr_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(uocr_core
    PUBLIC nlohmann_json::nlohmann_json Threads::Threads
    PRIVATE uocr_warnings)

target_compile_definitions(uocr_core PUBLIC UOCR_CPU_BACKEND=1)
if(ENGINE_BACKEND STREQUAL "CUDA")
    target_compile_definitions(uocr_core PUBLIC UOCR_CUDA_ENABLED=1)
endif()

if(ENGINE_ENABLE_SANITIZERS)
    target_compile_options(uocr_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(uocr_core PRIVATE -fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# CUDA kernels (only when requested and a toolkit is available)
# ---------------------------------------------------------------------------
if(ENGINE_BACKEND STREQUAL "CUDA")
    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "ENGINE_BACKEND=CUDA but no CUDA toolkit was found")
    endif()
    add_library(uocr_cuda
        src/kernels/cuda/rmsnorm.cu
        src/kernels/cuda/rope_fused.cu
        src/kernels/cuda/rswa_attention.cu
        src/kernels/cuda/moe_gemm_int4.cu
        src/kernels/cuda/backend.cu)
    target_include_directories(uocr_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(uocr_cuda PUBLIC uocr_core CUDA::cudart)
    target_compile_options(uocr_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(uocr_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Convenience alias used by tests/benchmarks/tools so that backend switching is
# a single place.
if(ENGINE_BACKEND STREQUAL "CUDA")
    add_library(uocr_engine ALIAS uocr_cuda)
    set(UOCR_ENGINE_LIB uocr_cuda)
else()
    add_library(uocr_engine ALIAS uocr_core)
    set(UOCR_ENGINE_LIB uocr_core)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ENGINE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(ENGINE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

message(STATUS "-------------------------------------------------------------")
message(STATUS "Unlimited-OCR engine configuration")
message(STATUS "  backend          : ${ENGINE_BACKEND}")
message(STATUS "  C++ standard     : ${CMAKE_CXX_STANDARD}")
message(STATUS "  build type       : ${CMAKE_BUILD_TYPE}")
if(ENGINE_BACKEND STREQUAL "CUDA")
    message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  tests/benchmarks : ${ENGINE_BUILD_TESTS}/${ENGINE_BUILD_BENCHMARKS}")
message(STATUS "-------------------------------------------------------------")
