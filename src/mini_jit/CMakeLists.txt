cmake_minimum_required(VERSION 3.15)

set(LIB_SOURCES
    backend/Kernel.cpp
    generator/Brgemm.cpp
    generator/Util.cpp
    generator/Unary.cpp
    instructions/base.cpp
    instructions/neon.cpp
    include/gemm_ref.cpp
)

add_library(mini_jit STATIC ${LIB_SOURCES})

target_include_directories(mini_jit PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/backend
    ${CMAKE_CURRENT_SOURCE_DIR}/generator
    ${CMAKE_CURRENT_SOURCE_DIR}/instructions
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mini_jit PUBLIC -O3 -Wall -Wextra -pedantic)
endif()

set_target_properties(mini_jit PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(gemm_benchmark benchmark.cpp)

target_link_libraries(gemm_benchmark PRIVATE mini_jit)

if (CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set_target_properties(gemm_benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
endif()
