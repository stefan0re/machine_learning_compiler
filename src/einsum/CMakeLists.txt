# CMakeLists.txt for building test_generic_einsum.cpp

cmake_minimum_required(VERSION 3.16)
project(EinsumTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../mini_jit/include
)

# Source files
set(SOURCES
    test_optimize.cpp
    ../mini_jit/generator/Brgemm.cpp
    ../mini_jit/generator/Util.cpp
    ../mini_jit/backend/Kernel.cpp
    ../mini_jit/instructions/neon.cpp
    ../mini_jit/instructions/base.cpp
    backend/TensorOperation.cpp
    ../mini_jit/generator/Unary.cpp
    include/einsum_ref.cpp
    ../mini_jit/include/gemm_ref.cpp
)

# Add executable
add_executable(einsum_test ${SOURCES})
