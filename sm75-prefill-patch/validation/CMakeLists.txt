cmake_minimum_required(VERSION 3.18)
project(sm75_prefill_tests LANGUAGES CXX)

# Standalone: these tests do not configure llama.cpp or require a model.
option(SM75_TEST_CUDA "Build the SM75 CUDA Q8 equivalence test" OFF)
enable_testing()
set(GGML_SOURCE_DIR "" CACHE PATH "Path to the repository after applying the patch")
if(NOT EXISTS "${GGML_SOURCE_DIR}/ggml/src/ggml-cuda/sm75-tuning.cuh")
    message(FATAL_ERROR "Set GGML_SOURCE_DIR to the patched repository")
endif()

add_executable(test-sm75-host test-sm75-host.cpp)
target_compile_features(test-sm75-host PRIVATE cxx_std_17)
target_include_directories(test-sm75-host PRIVATE "${GGML_SOURCE_DIR}/ggml/src/ggml-cuda")
add_test(NAME sm75-host COMMAND test-sm75-host)

if(SM75_TEST_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    enable_language(CUDA)
    add_executable(test-sm75-q8 test-sm75-q8.cu)
    set_target_properties(test-sm75-q8 PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_include_directories(test-sm75-q8 PRIVATE "${GGML_SOURCE_DIR}/ggml/src/ggml-cuda")
    # Match the backend's fast-math setting; _rn must still preserve equivalence.
    target_compile_options(test-sm75-q8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    add_test(NAME sm75-q8 COMMAND test-sm75-q8)
    set_tests_properties(sm75-q8 PROPERTIES SKIP_RETURN_CODE 77)
endif()
