set(UOCR_TEST_SOURCES
    test_main.cpp
    test_kv_ring_buffer.cpp
    test_memory_budget.cpp
    test_scheduler.cpp
    test_decoder.cpp
    test_moe_gate.cpp
    test_tokenizer.cpp)

add_executable(uocr_tests ${UOCR_TEST_SOURCES})
target_include_directories(uocr_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(uocr_tests PRIVATE ${UOCR_ENGINE_LIB})

add_test(NAME uocr_tests COMMAND uocr_tests)
set_tests_properties(uocr_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT "UOCR_MODEL_DIR=${CMAKE_SOURCE_DIR}/models;UOCR_TOKENIZER=${CMAKE_SOURCE_DIR}/models/tokenizer.json")

if(ENGINE_BACKEND STREQUAL "CUDA")
    add_executable(uocr_cuda_tests test_rswa_cuda.cu)
    target_link_libraries(uocr_cuda_tests PRIVATE ${UOCR_ENGINE_LIB})
    add_test(NAME uocr_cuda_tests COMMAND uocr_cuda_tests)
    set_tests_properties(uocr_cuda_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
