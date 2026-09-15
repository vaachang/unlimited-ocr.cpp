# CUDAArch.cmake -- architecture / toolkit setup for the CUDA backend.
#
# The engine targets sm_120 (consumer Blackwell, RTX 50 series).  sm_100's
# tensor-memory / tcgen05 instructions are intentionally not used because they
# are unavailable on sm_120.
#
# This file is included from the top-level CMakeLists.txt *after* project().

if(ENGINE_BACKEND STREQUAL "CUDA")
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(WARNING "ENGINE_BACKEND=CUDA but no CUDA compiler found; "
                        "configure with -DCMAKE_CUDA_COMPILER=<nvcc>")
        return()
    endif()

    enable_language(CUDA)

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
        message(FATAL_ERROR
            "CUDA >= 12.8 required for sm_120, found ${CMAKE_CUDA_COMPILER_VERSION}")
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING
            "CUDA architectures (sm_120 = consumer Blackwell)" FORCE)
    endif()

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    find_package(CUDAToolkit REQUIRED)

    message(STATUS "CUDA compiler     : ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")
else()
    # Allow CPU builds to know whether a CUDA toolkit exists (useful for tools).
    find_package(CUDAToolkit QUIET)
endif()
