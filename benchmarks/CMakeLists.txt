add_executable(bench_decode bench_decode.cpp)
target_link_libraries(bench_decode PRIVATE ${UOCR_ENGINE_LIB})

add_executable(bench_throughput bench_throughput.cpp)
target_link_libraries(bench_throughput PRIVATE ${UOCR_ENGINE_LIB})

if(ENGINE_BACKEND STREQUAL "CUDA")
    add_executable(bench_cuda_decode bench_cuda_decode.cu)
    target_link_libraries(bench_cuda_decode PRIVATE ${UOCR_ENGINE_LIB})
endif()
