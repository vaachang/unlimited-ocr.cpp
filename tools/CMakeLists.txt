add_executable(inspect_model inspect_model.cpp)
target_link_libraries(inspect_model PRIVATE ${UOCR_ENGINE_LIB})
add_executable(compare_reference compare_reference.cpp)
target_link_libraries(compare_reference PRIVATE ${UOCR_ENGINE_LIB})
add_executable(compare_vision compare_vision.cpp)
target_link_libraries(compare_vision PRIVATE ${UOCR_ENGINE_LIB})
add_executable(compare_tokenizer compare_tokenizer.cpp)
target_link_libraries(compare_tokenizer PRIVATE ${UOCR_ENGINE_LIB})
add_executable(compare_layout compare_layout.cpp)
target_link_libraries(compare_layout PRIVATE ${UOCR_ENGINE_LIB})
add_executable(compare_image compare_image.cpp)
target_link_libraries(compare_image PRIVATE ${UOCR_ENGINE_LIB})
add_executable(compare_ocr compare_ocr.cpp)
target_link_libraries(compare_ocr PRIVATE ${UOCR_ENGINE_LIB})

if(ENGINE_BACKEND STREQUAL "CUDA")
    add_executable(compare_vision_gpu compare_vision_gpu.cpp)
    target_link_libraries(compare_vision_gpu PRIVATE ${UOCR_ENGINE_LIB})
    # Synthetic (weight-free) vision-kernel self-test.  Covers the tensor-core
    # flash-attention path with a multi-tile S (=1024) and fully zeroed outputs
    # so that a broken/launch-failed kernel cannot pass by leaving reference
    # values in place.
    if(ENGINE_BUILD_TESTS)
        add_test(NAME compare_vision_gpu_selftest COMMAND compare_vision_gpu --selftest)
        set_tests_properties(compare_vision_gpu_selftest PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
endif()
