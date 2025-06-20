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

set(LIB_SOURCES
    ./trees/einsum_trees.cpp
    ./backend/TensorOperation.cpp
    ./include/einsum_ref.cpp
)

add_library(einsum STATIC ${LIB_SOURCES})

target_include_directories(einsum
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backend>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/trees>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(einsum PUBLIC -O3)
endif()

set_target_properties(einsum PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

set(EXTERNAL_SOURCES
    ../tensor/tensor.cpp
    ../mini_jit/generator/Brgemm.cpp
    ../mini_jit/generator/Util.cpp
    ../mini_jit/backend/Kernel.cpp
    ../mini_jit/instructions/neon.cpp
    ../mini_jit/instructions/base.cpp
    ../mini_jit/generator/Unary.cpp
    ../mini_jit/include/gemm_ref.cpp
)

