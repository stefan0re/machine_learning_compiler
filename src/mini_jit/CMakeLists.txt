cmake_minimum_required(VERSION 3.15)
project(mini_jit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


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

target_include_directories(mini_jit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backend>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generator>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/instructions>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mini_jit PUBLIC -O3)
endif()

set_target_properties(mini_jit PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# Little helper binary (optional)
# ---------------------------------------------------------------------------
add_executable(check_brgemm check_brgemm.cpp)
target_link_libraries(check_brgemm PRIVATE mini_jit)
