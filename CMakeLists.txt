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
# Optional: directory containing exported PyTorch reference tensors
# (ref_decoder/, ref_vision/, ref_tokenizer/, ref_layout/, ref_image/, ref_ocr/).
# When set, the compare_* tools are registered as CTest tests.  Empty -> skip.
set(ENGINE_REFERENCE_DIR "" CACHE PATH "Reference tensor directory for compare_* CTest")

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

# OpenMP parallelises the CPU reference kernels (vision encoder especially).
find_package(OpenMP QUIET)

# ---------------------------------------------------------------------------
# Core library (runtime + scheduler + engine + CPU kernels)
# ---------------------------------------------------------------------------
add_library(uocr_core
    src/kernels/cpu_ops.cpp
    src/runtime/config.cpp
    src/runtime/log.cpp
    src/runtime/safetensors.cpp
    src/runtime/tokenizer.cpp
    src/runtime/prompt.cpp
    src/runtime/image.cpp
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

if(OpenMP_CXX_FOUND)
    target_link_libraries(uocr_core PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for CPU kernels")
endif()

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
# The CUDA sources are added directly to `uocr_core` so that the engine (which
# lives in core) can own a device decoder without a circular library dependency.
if(ENGINE_BACKEND STREQUAL "CUDA")
    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "ENGINE_BACKEND=CUDA but no CUDA toolkit was found")
    endif()
    target_sources(uocr_core PRIVATE
        src/kernels/cuda/rmsnorm.cu
        src/kernels/cuda/rope_fused.cu
        src/kernels/cuda/rswa_attention.cu
        src/kernels/cuda/moe_gemm_int4.cu
        src/kernels/cuda/gpu_ops.cu
        src/kernels/cuda/gpu_cache.cu
        src/kernels/cuda/moe_device.cu
        src/kernels/cuda/backend.cu
        src/kernels/cuda/vision_ops.cu
        src/engine/gpu_decoder.cu
        src/engine/gpu_encoder.cu)
    target_link_libraries(uocr_core PUBLIC CUDA::cudart)
    target_compile_options(uocr_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(uocr_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# The engine library is always `uocr_core`; the backend is selected at compile
# time via UOCR_CUDA_ENABLED / UOCR_CPU_BACKEND.
add_library(uocr_engine ALIAS uocr_core)
set(UOCR_ENGINE_LIB uocr_core)

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

# Optional reference-alignment tests.  These need the exported PyTorch tensors
# (tools/reference/export_*.py) and, for the vision/OCR tools, the real model
# weights; they are skipped unless ENGINE_REFERENCE_DIR is provided.
if(ENGINE_BUILD_TOOLS AND ENGINE_REFERENCE_DIR AND ENGINE_BUILD_TESTS)
    set(_ref ${ENGINE_REFERENCE_DIR})
    add_test(NAME compare_reference
             COMMAND compare_reference --model ${CMAKE_SOURCE_DIR}/models --ref ${_ref}/ref_decoder)
    add_test(NAME compare_tokenizer
             COMMAND compare_tokenizer --model ${CMAKE_SOURCE_DIR}/models
                     --ref ${_ref}/ref_tokenizer/tokenizer_cases.json)
    add_test(NAME compare_layout
             COMMAND compare_layout --model ${CMAKE_SOURCE_DIR}/models
                     --ref ${_ref}/ref_layout/layout_cases.json)
    add_test(NAME compare_image
             COMMAND compare_image --ref ${_ref}/ref_image)
    add_test(NAME compare_vision
             COMMAND compare_vision --model ${CMAKE_SOURCE_DIR}/models --ref ${_ref}/ref_vision)
    set_tests_properties(compare_vision PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=8")
    add_test(NAME compare_ocr
             COMMAND compare_ocr --model ${CMAKE_SOURCE_DIR}/models --ref ${_ref}/ref_ocr)
    set_tests_properties(compare_ocr PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=8")
    set_tests_properties(compare_reference compare_tokenizer compare_layout compare_image
                         compare_vision compare_ocr PROPERTIES LABELS reference)
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
