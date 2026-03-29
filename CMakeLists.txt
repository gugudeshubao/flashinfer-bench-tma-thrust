cmake_minimum_required(VERSION 3.24)
project(gdn_kernels LANGUAGES CXX CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Target architectures
# sm_100 = Blackwell (B200)
# sm_90 = Hopper (H100)
# sm_80 = Ampere (A100)
set(CMAKE_CUDA_ARCHITECTURES 100)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# For FP8 support
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src/kernels)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Source files
set(KERNEL_SOURCES
    ${CMAKE_SOURCE_DIR}/src/gdn_kernels.cu
)

# Build shared library
add_library(gdn_kernels SHARED ${KERNEL_SOURCES})

# Link CUDA runtime
target_link_libraries(gdn_kernels CUDA::cudart)

# Install
install(TARGETS gdn_kernels LIBRARY DESTINATION lib)
install(FILES
    ${CMAKE_SOURCE_DIR}/src/kernels/gdn_decode_v7.cuh
    ${CMAKE_SOURCE_DIR}/src/kernels/gdn_prefill_v7.cuh
    DESTINATION include/gdn
)

# Python bindings (optional, using pybind11)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(gdn_kernels_py ${CMAKE_SOURCE_DIR}/src/gdn_bindings.cu)
    target_link_libraries(gdn_kernels_py PRIVATE gdn_kernels)
endif()

# Print configuration
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
