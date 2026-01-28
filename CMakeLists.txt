cmake_minimum_required(VERSION 4.0)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER "")

project(fft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)


include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)

enable_testing()

add_subdirectory(external/fftw-3.3.10)

find_package(OpenMP REQUIRED)

add_executable(
        fft
        cpp/tests/ft_analytical_tests.cpp
        cpp/src/naive_dft.h
        cpp/src/openmp_cooley_tukey_fft.h
        cpp/tests/ft_random_tests.cpp
        cpp/src/cu_fft.cuh
        cpp/src/cuda_cooley_tukey.h
        cpp/src/cuda_cooley_tukey_device.cu
)

add_executable(
        fft_benchmark
        cpp/tests/ft_performance_tests.cpp
        cpp/src/cuda_cooley_tukey_device.cu
)


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_ARCHITECTURES native)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    target_link_libraries(fft PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cufft)
    target_link_libraries(fft_benchmark PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cufft)
else()
    message(FATAL_ERROR "No CUDA compiler found")
endif()

string(TIMESTAMP TIME "%Y%m%dT%H%M%SZ")
add_custom_target(run_fft_benchmark
        COMMAND $<TARGET_FILE:fft_benchmark>
        --benchmark_format=json
        --benchmark_out=${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fft_benchmark_${TIME}.json
        DEPENDS fft_benchmark
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running FFT benchmark and generating JSON output"
)



target_link_libraries(
        fft
        PRIVATE
        OpenMP::OpenMP_CXX
        GTest::gtest_main
        pthread
        fftw3
        fftw3f
)

target_link_libraries(
        fft_benchmark
        PRIVATE
        OpenMP::OpenMP_CXX
        benchmark::benchmark
        pthread
        fftw3
        fftw3f
)

include(GoogleTest)
gtest_discover_tests(fft)

target_compile_options(fft_benchmark PRIVATE -O3)
