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
    ../tensor/tensor.cpp
)

add_library(einsum STATIC ${LIB_SOURCES})

target_include_directories(einsum
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backend>
        # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/trees>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mini_jit>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mini_jit/backend>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mini_jit/generator>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mini_jit/instructions>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../mini_jit/include>
)

target_link_libraries(einsum PUBLIC mini_jit OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(einsum PUBLIC -O3)
endif()

set_target_properties(einsum PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)